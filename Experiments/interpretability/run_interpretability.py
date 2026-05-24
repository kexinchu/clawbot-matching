"""Run the weight interpretability experiment over the 20-task tiered testset.

For every task in `simulator/20_Tasks_Testset_tiered.json` we:
  1. Build a requester (with the proposer's true capabilities + needs),
     a task (with synthetic offers derived from required_skills) and a
     candidate pool (init μ/σ from `capability_priors`).
  2. Run OnlineLearning (UCB on, dreaming off, weight SGD on) for
     N_ROUNDS rounds against a GroundTruthFeedback provider that scores
     reward against the candidates' TRUE capabilities (the same trick as
     in `Online_learning/tests/run_20_tasks_evaluation.py`).
  3. Snapshot every round (theta, weights, gap/need details, μ/σ of all
     candidates) via InterpretabilityProbe.
  4. Compute alignment metrics:
       w_j        : Top-1 vs Gap-weighted importance, Kendall τ
       attention  : Top-1 hit per req→cap and per need→offer, mean entropy
       θ          : final (w_c, w_n) vs complement/motivation ratio
       μ, σ       : critical vs non-critical capability convergence
  5. Aggregate across the 20 tasks and dump
     `logs/interpretability_<timestamp>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_ONLINE = _REPO_ROOT / "Online_learning"
_MAPPING_ALGO = _REPO_ROOT / "mapping-algo"
for _p in (str(_HERE), str(_ONLINE), str(_MAPPING_ALGO), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env BEFORE LLM_dreaming import (that module snapshots API_KEY at import).
from env_loader import load_dotenv, check_openrouter_key, dream_config_from_env  # noqa: E402
load_dotenv()

from datatypes import (  # noqa: E402
    CapabilityEntry,
    NeedEntry,
    Task,
    TaskOffer,
    TaskRequirement,
    UserState,
)
from config import MatchConfig  # noqa: E402
from encoder import SimpleEncoder  # noqa: E402
from WorldModel import WorldModel  # noqa: E402
from Online_learning import OnlineLearning  # noqa: E402
from feedback_provider import (  # noqa: E402
    FeedbackProvider, SimulatorFeedback, SkillLevelFeedback,
)
from ol_utils import dummy_user_feedback  # noqa: E402

# LLM_Dreaming lives outside the standard sys.path additions above
_LLM_DREAMING_DIR = _REPO_ROOT / "LLM_Dreaming"
if str(_LLM_DREAMING_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_DREAMING_DIR))
import LLM_dreaming as _dreaming_mod  # noqa: E402
_dreaming_mod.API_KEY = os.environ.get("OPENAI_API_KEY", "")
from LLM_dreaming import DreamSimulator  # noqa: E402

import gt_extractor as gt  # noqa: E402
import metrics as mt  # noqa: E402
from probe import InterpretabilityProbe  # noqa: E402
from experiment_config import ExperimentContext, build_encoder  # noqa: E402


# ---------------------------------------------------------------------------
# OpenRouter-compatible DreamSimulator (wrapper, does not modify source).
# The shipped DreamSimulator targets the Anthropic native response shape
# ({"content": [{"text": ...}]}) and never prepends the system prompt to
# `messages`. Both are fine for Anthropic's direct API but break against
# OpenAI / OpenRouter. We subclass + override `_call_llm` to fix those.
# ---------------------------------------------------------------------------

class _OpenRouterDreamSimulator(DreamSimulator):
    """DreamSimulator that speaks the OpenAI / OpenRouter chat-completions
    protocol. Two differences from the parent:

      * `system` is sent as the first message with role="system" (OpenAI
        style) instead of being dropped on the floor.
      * Response is parsed as data["choices"][0]["message"]["content"]
        rather than data["content"][0]["text"].

    Authentication reads $OPENAI_API_KEY at construction time (after .env load).
    """

    def __init__(
        self,
        n_turns: int,
        base_url: str,
        model: str,
        temperature: float = 0.2,
    ):
        _dreaming_mod.API_KEY = os.environ.get("OPENAI_API_KEY", "")
        super().__init__(
            n_turns=n_turns,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    def _call_llm(self, system: str, messages: list) -> str:
        if self.mock_mode:
            return self._mock_response(system, messages)

        api_key = os.environ.get("OPENAI_API_KEY", "")
        full_messages = [{"role": "system", "content": system}] + list(messages)
        resp = self.client.post(
            self.api_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                # Optional but nice-to-have for OpenRouter analytics
                "HTTP-Referer": "https://github.com/clawbot-matching",
                "X-Title": "clawbot-interpretability",
            },
            json={
                "model": self.model,
                "messages": full_messages,
                "temperature": self.temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            # Surface the raw response so we don't silently fail
            raise RuntimeError(f"Unexpected response shape: {data}")


_BASE_URL_SHORTCUTS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai":     "https://api.openai.com/v1",
}


def _build_dream_simulator(
    base_url: str,
    model: str,
    n_turns: int,
) -> Optional[DreamSimulator]:
    """Construct a DreamSimulator. Returns None if dreaming should run in
    the original mock path (no API_KEY, no base_url).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    resolved_url = _BASE_URL_SHORTCUTS.get(base_url, base_url) if base_url else ""
    if not api_key or not resolved_url:
        return None
    return _OpenRouterDreamSimulator(
        n_turns=n_turns,
        base_url=resolved_url,
        model=model,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TESTSET_PATH = _REPO_ROOT / "simulator" / "20_Tasks_Testset_tiered.json"
LOGS_DIR = _REPO_ROOT / "logs"

# Defaults — overridden by configure_experiment() in main()
_CTX = build_encoder("simple")
ENC = _CTX.enc
EMBED_DIM = _CTX.embed_dim
CFG = _CTX.cfg

N_ROUNDS_DEFAULT = 30
POOL_SIZE_DEFAULT = 5
EVAL_THETA_C = 0.4
EVAL_THETA_N = -0.1
TRUE_SIGMA = 1e-3
SIGMA_INIT_FALLBACK = 0.25
CANDIDATE_UCB_C = 0.5


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def configure_experiment(ctx: ExperimentContext) -> None:
    """Apply encoder + MatchConfig to module-level globals used by builders."""
    global _CTX, ENC, EMBED_DIM, CFG
    _CTX = ctx
    ENC = ctx.enc
    EMBED_DIM = ctx.embed_dim
    CFG = ctx.cfg


def build_task(task_dict: dict) -> Task:
    """Soft requirements straight from required_skills. Synthetic offers
    mirror the requirements (strength = level, description = skill) so
    that need→offer attention has something to attend to. This is the
    natural reading: working on a task that requires skill X offers
    exposure to X.
    """
    reqs = [
        TaskRequirement(ENC(skill), level=float(level),
                        constraint_type="soft", description=skill)
        for skill, level in task_dict.get("required_skills", {}).items()
    ]
    offers_src = task_dict.get("offer_skills") or task_dict.get("required_skills", {})
    offers = [
        TaskOffer(ENC(skill), strength=float(level),
                  source="explicit", description=skill)
        for skill, level in offers_src.items()
    ]
    return Task(
        task_id=task_dict["task_id"],
        goal=task_dict.get("description", task_dict["task_id"]),
        requirements=reqs,
        offers=offers,
        data_clearance=0,
    )


def build_requester(profile: dict) -> UserState:
    """Requester with its true capabilities + needs from the testset.

    Unlike the convergence experiment in run_20_tasks_evaluation, we do
    NOT empty out capabilities here: the whole point of this experiment
    is to surface 'what u still lacks', which requires u to actually
    have some skills.
    """
    caps = [
        CapabilityEntry(ENC(skill), mu=float(level), sigma=TRUE_SIGMA,
                        source="explicit", description=skill)
        for skill, level in profile.get("capabilities", {}).items()
    ]
    needs = [
        NeedEntry(ENC(skill), intensity=float(intensity), description=skill)
        for skill, intensity in profile.get("needs", {}).items()
    ]
    return UserState(
        user_id=profile["user_id"],
        capabilities=caps,
        needs=needs,
        clearance_level=0,
    )


def build_true_candidate(profile: dict) -> UserState:
    """A frozen 'truth' UserState (σ≈0) — used only by the feedback
    provider so the reward signal stays unbiased over rounds.
    """
    caps = [
        CapabilityEntry(ENC(skill), mu=float(level), sigma=TRUE_SIGMA,
                        source="explicit", description=skill)
        for skill, level in profile.get("capabilities", {}).items()
    ]
    needs = [
        NeedEntry(ENC(skill), intensity=float(intensity), description=skill)
        for skill, intensity in profile.get("needs", {}).items()
    ]
    return UserState(
        user_id=profile["user_id"],
        capabilities=caps,
        needs=needs,
        clearance_level=0,
    )


def build_learning_candidate(
    candidate_entry: dict,
    fallback_sigma: float = SIGMA_INIT_FALLBACK,
) -> UserState:
    """Learning-side candidate with priors from capability_priors (tiered
    noise baked in), needs from the true profile. Matches what
    run_20_tasks_evaluation does so results are comparable.
    """
    profile = candidate_entry["candidate_profile"]
    priors = candidate_entry.get("capability_priors", {}) or {}
    true_caps = profile.get("capabilities", {}) or {}
    caps: List[CapabilityEntry] = []
    for skill, mu_true in true_caps.items():
        p = priors.get(skill)
        if p is not None:
            mu_init = float(p["mu_init"])
            sigma_init = float(p["sigma_init"])
        else:
            mu_init = float(mu_true)
            sigma_init = fallback_sigma
        caps.append(
            CapabilityEntry(
                ENC(skill), mu=mu_init, sigma=sigma_init,
                source="explicit", description=skill,
            )
        )
    needs = [
        NeedEntry(ENC(skill), intensity=float(intensity), description=skill)
        for skill, intensity in profile.get("needs", {}).items()
    ]
    return UserState(
        user_id=profile["user_id"],
        capabilities=caps,
        needs=needs,
        clearance_level=0,
    )


# ---------------------------------------------------------------------------
# Ground-truth feedback (reward against TRUE capabilities)
# ---------------------------------------------------------------------------

class GroundTruthFeedback(FeedbackProvider):
    """Compute reward via a neutral WorldModel against true UserStates.

    Without this, when the agent's μ priors are wrong the rewards just
    confirm its mistaken belief and Bayesian updates never recover.
    """

    def __init__(self, true_states_by_id: Dict[str, UserState]):
        self.true_states = true_states_by_id
        self._eval_wm = WorldModel(
            config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N
        )

    def collect(self, requester, candidate, task, match):
        true_v = self.true_states.get(candidate.user_id, candidate)
        true_match = self._eval_wm.compute_match(
            requester, true_v, task, use_ucb=False, round_t=1,
        )
        return dummy_user_feedback(true_match)


# ---------------------------------------------------------------------------
# Per-task metric computation
# ---------------------------------------------------------------------------

def _wj_alignment(
    task: Task,
    requester: UserState,
) -> Dict[str, float]:
    soft = [r for r in task.requirements if r.constraint_type == "soft"]
    if not soft:
        return {"top1_q": 0.0, "top1_gap_weighted": 0.0,
                "kendall_tau_vs_gap": 0.0,
                "wj_top1_share": 0.0, "gap_top1_share": 0.0}

    q_vals = np.array([r.level for r in soft])
    w_j = q_vals / (q_vals.sum() + 1e-8)

    _, gap_scores = gt.critical_req_by_gap_weighted(
        task, requester, CFG.temperature
    )
    gap_arr = np.asarray(gap_scores, dtype=float)

    return {
        "top1_q": float(int(np.argmax(w_j)) == int(np.argmax(q_vals))),
        "top1_gap_weighted": float(
            int(np.argmax(w_j)) == int(np.argmax(gap_arr))
            if gap_arr.sum() > 0 else 0.0
        ),
        "kendall_tau_vs_gap": mt.kendall_tau_b(w_j.tolist(), gap_scores),
        "wj_top1_share": mt.attribution_share(w_j.tolist()),
        "gap_top1_share": mt.attribution_share(gap_scores)
            if gap_arr.sum() > 0 else 0.0,
        "skills": [r.description for r in soft],
        "w_j": w_j.tolist(),
        "gap_weighted": gap_scores,
    }


def _attention_alignment_req_to_cap(
    task: Task,
    candidate: UserState,
) -> Dict[str, object]:
    soft = [r for r in task.requirements if r.constraint_type == "soft"]
    per_req: List[Dict[str, object]] = []
    hits: List[int] = []
    entropies: List[float] = []
    full_matrix: List[List[float]] = []
    cap_labels = [cap.description for cap in candidate.capabilities]

    for req in soft:
        alpha = gt.req_to_cap_attention(req, candidate, CFG.temperature)
        if alpha.size == 0:
            continue
        gt_idx, gt_cap, gt_sim = gt.critical_cap_for_req(req, candidate)
        argmax_idx = int(np.argmax(alpha))
        hit = int(argmax_idx == gt_idx) if gt_idx is not None else 0
        ent = mt.normalised_entropy(alpha.tolist())
        per_req.append({
            "req": req.description,
            "gt_cap": gt_cap.description if gt_cap is not None else None,
            "argmax_cap": cap_labels[argmax_idx] if cap_labels else None,
            "argmax_weight": float(alpha[argmax_idx]),
            "gt_weight": float(alpha[gt_idx])
                if gt_idx is not None else 0.0,
            "hit": hit,
            "normalised_entropy": ent,
        })
        hits.append(hit)
        entropies.append(ent)
        full_matrix.append(alpha.tolist())

    return {
        "per_req": per_req,
        "cap_labels": cap_labels,
        "alpha_matrix": full_matrix,
        "mean_top1": float(np.mean(hits)) if hits else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def _attention_alignment_need_to_offer(
    task: Task,
    candidate: UserState,
) -> Dict[str, object]:
    per_need: List[Dict[str, object]] = []
    hits: List[int] = []
    entropies: List[float] = []
    offer_labels = [o.description for o in task.offers]

    for need in candidate.needs:
        alpha = gt.need_to_offer_attention(need, task, CFG.temperature)
        if alpha.size == 0:
            continue
        gt_idx, gt_off, _ = gt.critical_offer_for_need(need, task)
        argmax_idx = int(np.argmax(alpha))
        hit = int(argmax_idx == gt_idx) if gt_idx is not None else 0
        ent = mt.normalised_entropy(alpha.tolist())
        per_need.append({
            "need": need.description,
            "gt_offer": gt_off.description if gt_off is not None else None,
            "argmax_offer": offer_labels[argmax_idx] if offer_labels else None,
            "argmax_weight": float(alpha[argmax_idx]),
            "gt_weight": float(alpha[gt_idx])
                if gt_idx is not None else 0.0,
            "hit": hit,
            "normalised_entropy": ent,
        })
        hits.append(hit)
        entropies.append(ent)

    return {
        "per_need": per_need,
        "mean_top1": float(np.mean(hits)) if hits else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def _mu_sigma_convergence(
    snapshots: List[Dict],
    true_caps: Dict[str, Dict[str, float]],
    critical_flags: Dict[str, Dict[str, bool]],
) -> Dict[str, object]:
    """Per-round mean |μ - μ_true| split by critical vs non-critical caps.

    Aggregates across ALL candidates in the pool — gives the algorithm
    every chance to learn, not just the selected candidate.
    """
    rounds_axis: List[int] = []
    critical_err: List[float] = []
    irrelevant_err: List[float] = []
    critical_sigma: List[float] = []
    irrelevant_sigma: List[float] = []

    for snap in snapshots:
        rounds_axis.append(snap["round"])
        c_err = []
        ir_err = []
        c_sig = []
        ir_sig = []
        for cid, caps in snap["candidate_caps"].items():
            for cap in caps:
                desc = cap["description"]
                true_mu = true_caps.get(cid, {}).get(desc)
                if true_mu is None:
                    continue
                is_crit = critical_flags.get(cid, {}).get(desc, False)
                err = abs(cap["mu"] - true_mu)
                if is_crit:
                    c_err.append(err)
                    c_sig.append(cap["sigma"])
                else:
                    ir_err.append(err)
                    ir_sig.append(cap["sigma"])
        critical_err.append(float(np.mean(c_err)) if c_err else float("nan"))
        irrelevant_err.append(float(np.mean(ir_err)) if ir_err else float("nan"))
        critical_sigma.append(float(np.mean(c_sig)) if c_sig else float("nan"))
        irrelevant_sigma.append(float(np.mean(ir_sig)) if ir_sig else float("nan"))

    final_c = critical_err[-1] if critical_err else float("nan")
    final_ir = irrelevant_err[-1] if irrelevant_err else float("nan")

    return {
        "rounds": rounds_axis,
        "critical_mu_abs_err": critical_err,
        "irrelevant_mu_abs_err": irrelevant_err,
        "critical_sigma": critical_sigma,
        "irrelevant_sigma": irrelevant_sigma,
        "final_critical_err": final_c,
        "final_irrelevant_err": final_ir,
        "critical_converges_faster": bool(final_c < final_ir)
            if not (np.isnan(final_c) or np.isnan(final_ir)) else False,
    }


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------

def _build_feedback_provider(
    feedback_mode: str,
    true_states_by_id: Dict[str, UserState],
    seed: int,
) -> FeedbackProvider:
    if feedback_mode == "gt":
        return GroundTruthFeedback(true_states_by_id)
    if feedback_mode == "sim":
        import simulator as sim  # local import to avoid hard dep when unused
        cfg_sim = sim.SimulatorConfig(
            backend_type="mock",
            random_seed=seed,
            trace_verbose=False,
        )
        return SimulatorFeedback(config=cfg_sim)
    if feedback_mode == "skill":
        import simulator as sim
        cfg_sim = sim.SimulatorConfig(
            backend_type="mock",
            random_seed=seed,
            trace_verbose=False,
        )
        return SkillLevelFeedback(true_states_by_id, config=cfg_sim)
    raise ValueError(
        f"Unknown feedback mode: {feedback_mode!r} "
        "(use gt, sim, or skill)"
    )


def _layer_rank_overlap(l31: List[str], l32: List[str], top: int) -> Dict[str, float]:
    """Compare L3.1 and L3.2 rankings: how many of the top-k overlap +
    what fraction of position-1 picks agree.
    """
    if not l31 or not l32:
        return {"top_k_overlap": 0.0, "top1_agree": 0.0, "kendall_tau": 0.0}
    k = min(top, len(l31), len(l32))
    s1 = set(l31[:k])
    s2 = set(l32[:k])
    overlap = len(s1 & s2) / float(k) if k > 0 else 0.0
    top1 = float(l31[0] == l32[0])
    # Approximate Kendall τ on common items
    common = [c for c in l31 if c in l32]
    pos1 = {c: i for i, c in enumerate(l31)}
    pos2 = {c: i for i, c in enumerate(l32)}
    common_ranks_a = [pos1[c] for c in common]
    common_ranks_b = [pos2[c] for c in common]
    kt = mt.kendall_tau_b(common_ranks_a, common_ranks_b) if len(common) >= 2 else 0.0
    return {"top_k_overlap": overlap, "top1_agree": top1, "kendall_tau": kt}


def evaluate_task(
    task_entry: dict,
    n_rounds: int,
    pool_size: int,
    seed: int,
    feedback_mode: str = "gt",
    enable_dreaming: bool = False,
    dream_simulator: Optional[DreamSimulator] = None,
) -> Dict:
    rng = np.random.default_rng(seed)

    task_dict = task_entry["task"]
    proposer = task_entry["proposer_profile"]
    candidate_entries = task_entry["candidates"]

    if pool_size > 0:
        candidate_entries = candidate_entries[:pool_size]

    task = build_task(task_dict)
    requester = build_requester(proposer)

    candidate_pool = [build_learning_candidate(c) for c in candidate_entries]
    true_states_by_id: Dict[str, UserState] = {
        c["candidate_profile"]["user_id"]: build_true_candidate(c["candidate_profile"])
        for c in candidate_entries
    }

    feedback_provider = _build_feedback_provider(
        feedback_mode, true_states_by_id, seed=seed,
    )
    world_model = WorldModel(
        config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N,
    )
    engine = OnlineLearning(
        world_model,
        feedback_provider=feedback_provider,
        dream_simulator=dream_simulator,
        enable_dreaming=enable_dreaming,
        enable_candidate_ucb=True,
        enable_weight_update=True,
        candidate_ucb_c=CANDIDATE_UCB_C,
        top_k=min(pool_size, 10),
        top_n=3,
    )
    probe = InterpretabilityProbe(engine)
    probe.initial_snapshot(requester, task, candidate_pool)

    for _ in range(n_rounds):
        probe.step(requester, task, candidate_pool)

    snapshots = probe.snapshots
    last_snap = snapshots[-1]

    # --- w_j alignment (task-static, no rounds dependency) ---
    wj = _wj_alignment(task, requester)

    # --- Attention alignment (use selected candidate at the END) ---
    selected_id = last_snap["selected_candidate_id"]
    selected_cand = next(
        (c for c in candidate_pool if c.user_id == selected_id),
        candidate_pool[0],
    )
    # Also compute against the analytically-optimal (top-1 by M against
    # the true profile) candidate so attention alignment isn't tied to
    # whatever UCB happened to pick.
    eval_wm = WorldModel(config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N)
    M_per_cand: Dict[str, float] = {}
    for c in candidate_pool:
        true_v = true_states_by_id[c.user_id]
        m = eval_wm.compute_match(requester, true_v, task,
                                  use_ucb=False, round_t=1)
        M_per_cand[c.user_id] = float(m.match_score)
    optimum_id = max(M_per_cand, key=M_per_cand.get)
    optimum_cand = next(c for c in candidate_pool if c.user_id == optimum_id)

    attn_req_cap_sel = _attention_alignment_req_to_cap(task, selected_cand)
    attn_req_cap_opt = _attention_alignment_req_to_cap(task, optimum_cand)
    attn_need_off_sel = _attention_alignment_need_to_offer(task, selected_cand)
    attn_need_off_opt = _attention_alignment_need_to_offer(task, optimum_cand)

    # --- θ trajectory + complement ratio ---
    theta_traj = [s["theta"] for s in snapshots]
    weights_traj = [(s["weights"]["w_c"], s["weights"]["w_n"])
                    for s in snapshots]
    complement_info = gt.complement_vs_motivation_ratio(
        task, requester, candidate_pool, CFG.temperature,
    )

    # --- μ, σ convergence ---
    true_caps_map: Dict[str, Dict[str, float]] = {
        c["candidate_profile"]["user_id"]:
            gt.mu_true_lookup(c["candidate_profile"])
        for c in candidate_entries
    }
    critical_flags: Dict[str, Dict[str, bool]] = {
        c.user_id: gt.label_critical_caps(task, c, CFG.tau_update)
        for c in candidate_pool
    }
    convergence = _mu_sigma_convergence(
        snapshots, true_caps_map, critical_flags,
    )

    # --- Layer 3.1 vs Layer 3.2 ranking comparison (dreaming only) ---
    ranking_overlap_history: List[Dict[str, float]] = []
    if enable_dreaming:
        for s in snapshots[1:]:  # skip the round-0 initial snapshot
            ranking_overlap_history.append(
                _layer_rank_overlap(
                    s["layer3_1_ranking"], s["layer3_2_ranking"], top=3,
                )
            )
    if ranking_overlap_history:
        mean_top_k_overlap = float(np.mean(
            [r["top_k_overlap"] for r in ranking_overlap_history]
        ))
        mean_top1_agree = float(np.mean(
            [r["top1_agree"] for r in ranking_overlap_history]
        ))
        mean_kendall = float(np.mean(
            [r["kendall_tau"] for r in ranking_overlap_history]
        ))
    else:
        mean_top_k_overlap = float("nan")
        mean_top1_agree = float("nan")
        mean_kendall = float("nan")

    # --- Dreaming recommendation distribution per task ---
    rec_counts: Dict[str, int] = {}
    for s in snapshots[1:]:
        for rec in s.get("layer3_2_recommendations") or []:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1

    return {
        "task_id": task_dict["task_id"],
        "title": task_dict.get("title", ""),
        "pool_size": len(candidate_pool),
        "selected_candidate_id": selected_id,
        "optimum_candidate_id": optimum_id,
        "M_per_candidate": M_per_cand,
        "w_j_alignment": wj,
        "attention_req_to_cap": {
            "selected": attn_req_cap_sel,
            "optimum": attn_req_cap_opt,
        },
        "attention_need_to_offer": {
            "selected": attn_need_off_sel,
            "optimum": attn_need_off_opt,
        },
        "complement_info": complement_info,
        "theta_trajectory": theta_traj,
        "weights_trajectory": [{"w_c": w_c, "w_n": w_n}
                               for w_c, w_n in weights_traj],
        "mu_sigma_convergence": convergence,
        "final_w": {"w_c": weights_traj[-1][0], "w_n": weights_traj[-1][1]},
        "dreaming": {
            "enabled": bool(enable_dreaming),
            "mean_layer3_top_k_overlap": mean_top_k_overlap,
            "mean_layer3_top1_agree": mean_top1_agree,
            "mean_layer3_kendall_tau": mean_kendall,
            "recommendation_counts": rec_counts,
        },
    }


# ---------------------------------------------------------------------------
# Aggregation across tasks
# ---------------------------------------------------------------------------

def aggregate(results: List[Dict]) -> Dict:
    wj_q = [r["w_j_alignment"]["top1_q"] for r in results]
    wj_gap = [r["w_j_alignment"]["top1_gap_weighted"] for r in results]
    wj_kt = [r["w_j_alignment"]["kendall_tau_vs_gap"] for r in results]

    attn_req_sel = [r["attention_req_to_cap"]["selected"]["mean_top1"]
                    for r in results]
    attn_req_opt = [r["attention_req_to_cap"]["optimum"]["mean_top1"]
                    for r in results]
    attn_need_sel = [r["attention_need_to_offer"]["selected"]["mean_top1"]
                     for r in results]
    attn_need_opt = [r["attention_need_to_offer"]["optimum"]["mean_top1"]
                     for r in results]

    final_wc = [r["final_w"]["w_c"] for r in results]
    ratio = [r["complement_info"]["ratio"] for r in results]

    crit_converges = [
        r["mu_sigma_convergence"]["critical_converges_faster"]
        for r in results
    ]
    final_crit_err = [
        r["mu_sigma_convergence"]["final_critical_err"]
        for r in results
        if not np.isnan(r["mu_sigma_convergence"]["final_critical_err"])
    ]
    final_ir_err = [
        r["mu_sigma_convergence"]["final_irrelevant_err"]
        for r in results
        if not np.isnan(r["mu_sigma_convergence"]["final_irrelevant_err"])
    ]

    dream_enabled = any(r["dreaming"]["enabled"] for r in results)
    if dream_enabled:
        l3_overlap = [r["dreaming"]["mean_layer3_top_k_overlap"]
                      for r in results
                      if not np.isnan(r["dreaming"]["mean_layer3_top_k_overlap"])]
        l3_top1 = [r["dreaming"]["mean_layer3_top1_agree"]
                   for r in results
                   if not np.isnan(r["dreaming"]["mean_layer3_top1_agree"])]
        l3_kt = [r["dreaming"]["mean_layer3_kendall_tau"]
                 for r in results
                 if not np.isnan(r["dreaming"]["mean_layer3_kendall_tau"])]
        rec_counts_agg: Dict[str, int] = {}
        for r in results:
            for k, v in r["dreaming"]["recommendation_counts"].items():
                rec_counts_agg[k] = rec_counts_agg.get(k, 0) + v
    else:
        l3_overlap, l3_top1, l3_kt = [], [], []
        rec_counts_agg = {}

    return {
        "wj_top1_q": float(np.mean(wj_q)),
        "wj_top1_gap_weighted": float(np.mean(wj_gap)),
        "wj_kendall_tau_vs_gap": float(np.mean(wj_kt)),
        "attn_req_to_cap_top1_selected": float(np.mean(attn_req_sel)),
        "attn_req_to_cap_top1_optimum": float(np.mean(attn_req_opt)),
        "attn_need_to_offer_top1_selected": float(np.mean(attn_need_sel)),
        "attn_need_to_offer_top1_optimum": float(np.mean(attn_need_opt)),
        "theta_spearman_wc_vs_complement_ratio": mt.spearman_rho(
            final_wc, ratio,
        ),
        "theta_kendall_wc_vs_complement_ratio": mt.kendall_tau_b(
            final_wc, ratio,
        ),
        "frac_tasks_critical_converges_faster": float(np.mean(crit_converges)),
        "mean_final_critical_err": float(np.mean(final_crit_err))
            if final_crit_err else float("nan"),
        "mean_final_irrelevant_err": float(np.mean(final_ir_err))
            if final_ir_err else float("nan"),
        "dreaming_enabled": bool(dream_enabled),
        "dreaming_mean_layer3_top_k_overlap": float(np.mean(l3_overlap))
            if l3_overlap else float("nan"),
        "dreaming_mean_layer3_top1_agree": float(np.mean(l3_top1))
            if l3_top1 else float("nan"),
        "dreaming_mean_layer3_kendall_tau": float(np.mean(l3_kt))
            if l3_kt else float("nan"),
        "dreaming_recommendation_counts": rec_counts_agg,
    }


def aggregate_multi_seed(per_seed: Dict[int, Dict]) -> Dict:
    """Combine per-seed summaries into mean ± std for numeric metrics."""
    summaries = [v for v in per_seed.values()]
    if not summaries:
        return {}
    keys = summaries[0].keys()
    out: Dict[str, object] = {"n_seeds": len(summaries)}
    for key in keys:
        if key in ("dreaming_enabled", "dreaming_recommendation_counts"):
            out[key] = summaries[0].get(key)
            continue
        vals = [s[key] for s in summaries if isinstance(s.get(key), (int, float))]
        if not vals:
            out[key] = summaries[0].get(key)
            continue
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def _parse_seeds(args) -> List[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.n_seeds and args.n_seeds > 1:
        return list(range(args.seed, args.seed + args.n_seeds))
    return [args.seed]


def _run_one_seed(
    tasks: List[dict],
    args,
    dream_sim,
    seed: int,
) -> Tuple[List[Dict], Dict]:
    results: List[Dict] = []
    for i, task_entry in enumerate(tasks, start=1):
        res = evaluate_task(
            task_entry,
            n_rounds=args.n_rounds,
            pool_size=args.pool_size,
            seed=seed,
            feedback_mode=args.feedback,
            enable_dreaming=args.dreaming,
            dream_simulator=dream_sim,
        )
        cv = res["mu_sigma_convergence"]
        dream_str = ""
        if res["dreaming"]["enabled"]:
            dream_str = (
                f" l3overlap={res['dreaming']['mean_layer3_top_k_overlap']:.2f}"
                f" l3top1={res['dreaming']['mean_layer3_top1_agree']:.2f}"
            )
        print(
            f"  [{i:>2}/{len(tasks)}] {res['task_id']}: "
            f"wj_top1_gap={res['w_j_alignment']['top1_gap_weighted']:.0f} "
            f"w_c={res['final_w']['w_c']:.3f} "
            f"μ_err(c/i)={cv['final_critical_err']:.3f}/"
            f"{cv['final_irrelevant_err']:.3f}" + dream_str
        )
        results.append(res)
    return results, aggregate(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    env_dream = dream_config_from_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=N_ROUNDS_DEFAULT)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE_DEFAULT,
                        help="Candidates per task (0 = all).")
    parser.add_argument("--num-tasks", type=int, default=0,
                        help="If > 0, only run the first N tasks (debug).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma-separated seeds for cross-seed runs.")
    parser.add_argument("--n-seeds", type=int, default=0,
                        help="If >1, run seed, seed+1, ... (overrides --seed).")
    parser.add_argument("--feedback", choices=("gt", "sim", "skill"), default="gt",
                        help="gt | sim | skill (sim + oracle skill observations).")
    parser.add_argument("--encoder", choices=("simple", "sbert"), default="simple",
                        help="Embedding backend: simple (BoW) or sbert (BGE).")
    parser.add_argument("--testset", type=str, default="",
                        help="Path to testset JSON (default: tiered 20-task).")
    parser.add_argument("--dreaming", action="store_true",
                        help="Enable Layer 3.2 LLM Dreaming refinement.")
    parser.add_argument("--dream-base-url", type=str,
                        default=env_dream.get("base_url", ""),
                        help='LLM endpoint shortcut ("openrouter") or full URL.')
    parser.add_argument("--dream-model", type=str,
                        default=env_dream.get("model", "deepseek/deepseek-chat-v3-0324"),
                        help="OpenRouter model id (e.g. deepseek/deepseek-chat-v3-0324).")
    parser.add_argument("--dream-n-turns", type=int,
                        default=int(env_dream.get("n_turns", "2")),
                        help="Agent-agent turns per dream (lower = cheaper).")
    parser.add_argument("--require-api-key", action="store_true",
                        help="Fail if --dreaming + --dream-base-url but no API key.")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag appended to output filename.")
    parser.add_argument("--out", type=str, default="",
                        help="Override output path under logs/.")
    args = parser.parse_args(argv)

    ctx = build_encoder(args.encoder)
    configure_experiment(ctx)

    testset_path = Path(args.testset) if args.testset else TESTSET_PATH
    seeds = _parse_seeds(args)

    dream_sim = None
    if args.dreaming:
        if args.dream_base_url:
            if args.require_api_key and not check_openrouter_key():
                from env_loader import ensure_dreaming_credentials
                ensure_dreaming_credentials()
            dream_sim = _build_dream_simulator(
                args.dream_base_url, args.dream_model, args.dream_n_turns,
            )
            if dream_sim is None:
                print("[interp] WARNING: --dream-base-url given but "
                      "OPENAI_API_KEY is empty. Falling back to mock dreaming.")
        if dream_sim is None:
            try:
                _saved_key = os.environ.pop("OPENAI_API_KEY", None)
                dream_sim = DreamSimulator(n_turns=args.dream_n_turns)
            finally:
                if _saved_key is not None:
                    os.environ["OPENAI_API_KEY"] = _saved_key

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(testset_path.read_text())
    tasks = data["tasks"]
    if args.num_tasks and args.num_tasks > 0:
        tasks = tasks[: args.num_tasks]

    tag = args.tag
    if not tag:
        tag = f"fb-{args.feedback}"
        if args.encoder != "simple":
            tag += f"_{args.encoder}"
        if args.dreaming:
            tag += "_dream"
            if dream_sim is not None:
                tag += "-real"

    backend_label = (
        f"real ({args.dream_base_url}, {args.dream_model}, "
        f"n_turns={args.dream_n_turns})"
        if dream_sim is not None
        else ("mock" if args.dreaming else "off")
    )
    print(
        f"[interp] testset={testset_path.name}  encoder={ctx.encoder_name}  "
        f"n_tasks={len(tasks)}  rounds={args.n_rounds}  pool={args.pool_size}  "
        f"seeds={seeds}  feedback={args.feedback}  dreaming={backend_label}  "
        f"tag={tag}"
    )
    if check_openrouter_key():
        print("[interp] OPENAI_API_KEY: configured (OpenRouter-compatible)")
    elif args.dreaming and args.dream_base_url:
        print("[interp] OPENAI_API_KEY: NOT set — real dreaming unavailable")

    per_seed_summary: Dict[int, Dict] = {}
    results: List[Dict] = []
    for seed in seeds:
        print(f"\n[interp] --- seed={seed} ---")
        seed_results, seed_summary = _run_one_seed(tasks, args, dream_sim, seed)
        per_seed_summary[seed] = seed_summary
        if len(seeds) == 1:
            results = seed_results

    summary = per_seed_summary[seeds[0]] if len(seeds) == 1 else aggregate_multi_seed(
        per_seed_summary,
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = (Path(args.out) if args.out
                else LOGS_DIR / f"interpretability_{tag}_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "testset": str(testset_path),
            "n_rounds": args.n_rounds,
            "pool_size": args.pool_size,
            "seeds": seeds,
            "feedback": args.feedback,
            "dreaming": bool(args.dreaming),
            "dreaming_backend": backend_label,
            "dream_base_url": args.dream_base_url,
            "dream_model": args.dream_model,
            "dream_n_turns": args.dream_n_turns,
            "tag": tag,
            "encoder": ctx.encoder_name,
            "tau_hard": CFG.tau_hard,
            "tau_update": CFG.tau_update,
            "temperature": CFG.temperature,
            "eval_theta": [EVAL_THETA_C, EVAL_THETA_N],
            "candidate_ucb_c": CANDIDATE_UCB_C,
            "openrouter_key_set": check_openrouter_key(),
        },
        "per_task": results,
        "summary": summary,
    }
    if len(seeds) > 1:
        payload["per_seed"] = {
            str(s): {"summary": per_seed_summary[s]} for s in seeds
        }

    out_path.write_text(json.dumps(payload, indent=2, default=str))

    print()
    print("=" * 78)
    print(f"Summary across {len(tasks)} tasks × {len(seeds)} seed(s):")
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k:<46s} {v['mean']:.4f} ± {v['std']:.4f}")
        elif isinstance(v, float):
            print(f"  {k:<46s} {v:.4f}")
        else:
            print(f"  {k:<46s} {v}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
