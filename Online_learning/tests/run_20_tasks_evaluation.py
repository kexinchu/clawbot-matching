"""Evaluate OnlineLearning against baselines on simulator 20-task benchmarks.

Five conditions are run per task:
  1. online_ucb   — full pipeline with candidate-level UCB (no dreaming for speed)
  2. random       — pick a candidate uniformly at random each round (5 seeds)
  3. scap_greedy  — S_cap only (w_n forced to 0, no candidate UCB, no weight SGD)
  4. lappas_coverage — Lappas-style Skill Coverage baseline: static weighted
                       discrete skill coverage only. This is not a full
                       reproduction of Lappas social-network team formation.
  5. mapscore_greedy — static full MapScore top-1 (gate + S_cap + S_need)

For each condition we record:
  - n_rounds  : number of rounds until the same candidate has been selected
                for K_CONVERGE consecutive rounds (capped at N_MAX_ROUNDS)
  - rho       : M_selected / M_optimum, where both are computed using each
                candidate's *true* capabilities (sigma=0, no UCB) under a
                fixed neutral WorldModel theta. rho == 1 means the system
                converged on the analytically optimal candidate.

Outputs:
  - logs/online_learning_20tasks_eval_<timestamp>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


TESTS_DIR = Path(__file__).resolve().parent
ONLINE_LEARNING_DIR = TESTS_DIR.parent
REPO_ROOT = ONLINE_LEARNING_DIR.parent
MAPPING_ALGO_DIR = REPO_ROOT / "mapping-algo"
LOGS_DIR = REPO_ROOT / "logs"
TESTSET_PATH = REPO_ROOT / "simulator" / "20_Tasks_Testset_tiered.json"

for path in (ONLINE_LEARNING_DIR, MAPPING_ALGO_DIR, REPO_ROOT, TESTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from WorldModel import WorldModel  # noqa: E402
from Online_learning import OnlineLearning  # noqa: E402
from config import MatchConfig  # noqa: E402
from datatypes import (  # noqa: E402
    CapabilityEntry,
    NeedEntry,
    Task,
    TaskOffer,
    TaskRequirement,
    UserState,
)
from encoder import SimpleEncoder  # noqa: E402
from feedback_provider import FeedbackProvider  # noqa: E402
from ol_utils import dummy_user_feedback  # noqa: E402
from simulator.bilateral_simulator import BilateralSimulator  # noqa: E402
from simulator.config import SimulatorConfig  # noqa: E402
from simulator.mock_backend import RuleBasedBackend  # noqa: E402
from simulator.outcome_simulator import OutcomeSimulator  # noqa: E402
from simulator.reward import compute_reward  # noqa: E402
from simulator.types import (  # noqa: E402
    CandidateCard,
    MatchingContext,
    TaskSpec,
    UserProfile,
)


# ---------- Experiment configuration ----------
ENC = SimpleEncoder(dim=64)
CFG = MatchConfig(embedding_dim=64)

K_CONVERGE = 3            # same candidate selected K times in a row → converged
N_MAX_ROUNDS = 60         # hard cap per run
RANDOM_SEEDS = 5          # seeds averaged for the random baseline
SIGMA_INIT_FALLBACK = 0.25  # fallback sigma if a skill has no entry in capability_priors
TRUE_SIGMA = 1e-3         # near-zero sigma for the "ground truth" M_optimum
# Candidate-level UCB exploration constant. sqrt(2) is UCB1's default but is
# too aggressive for 20-candidate pools with small M gaps; tuning down helps
# the policy actually settle within the round budget.
CANDIDATE_UCB_C = 0.5

# Prior-noise configuration. Set via CLI (--noise-mode / --noise-sigma).
# Default is "none" so existing behaviour is preserved when called with no
# arguments. The two modes are:
#   gaussian : mu_init = clip(mu_true + N(0, noise_sigma), 0, 1)
#   uniform  : mu_init = Uniform(0, 1) (completely random priors)
DEFAULT_NOISE_MODE = "none"
DEFAULT_NOISE_SIGMA = 0.0
NOISE_SEED = 12345

# Default analytical theta — used for *evaluation* (M, M_optimum) in every condition.
EVAL_THETA_C = 0.4
EVAL_THETA_N = -0.1

# Theta used by the S_cap-only baseline at training time (w_n ~= 0).
SCAP_THETA_C = 10.0
SCAP_THETA_N = -10.0


# ---------- Test set → datatypes ----------

def _capabilities_from_dict(
    cap_dict: Dict[str, float],
    sigma: float,
    source: str = "explicit",
) -> List[CapabilityEntry]:
    return [
        CapabilityEntry(ENC(skill), mu=float(level), sigma=sigma, source=source, description=skill)
        for skill, level in cap_dict.items()
    ]


def _needs_from_dict(need_dict: Dict[str, float]) -> List[NeedEntry]:
    return [
        NeedEntry(ENC(skill), intensity=float(intensity), description=skill)
        for skill, intensity in need_dict.items()
    ]


def _offers_from_dict(offer_dict: Dict[str, float]) -> List[TaskOffer]:
    return [
        TaskOffer(ENC(offer), strength=float(strength), source="explicit", description=offer)
        for offer, strength in offer_dict.items()
    ]


def build_user_state(profile: dict, sigma: float, source: str = "explicit") -> UserState:
    return UserState(
        user_id=profile["user_id"],
        capabilities=_capabilities_from_dict(profile.get("capabilities", {}), sigma, source),
        needs=_needs_from_dict(profile.get("needs", {})),
        clearance_level=0,
    )


def build_learning_state_from_priors(
    candidate_entry: dict,
    fallback_sigma: float = SIGMA_INIT_FALLBACK,
) -> UserState:
    """Build a learning UserState from a tier-augmented candidate entry.

    Uses ``capability_priors[skill] = {mu_init, sigma_init}`` as the agent's
    initial belief about each capability. Falls back to the truth μ and
    ``fallback_sigma`` for skills not present in priors (shouldn't happen
    with the standard augment script, but defensive).
    """
    profile = candidate_entry["candidate_profile"]
    priors = candidate_entry.get("capability_priors", {}) or {}
    capabilities = profile.get("capabilities", {}) or {}

    cap_entries: List[CapabilityEntry] = []
    for skill, mu_true in capabilities.items():
        p = priors.get(skill)
        if p is not None:
            mu_init = float(p["mu_init"])
            sigma_init = float(p["sigma_init"])
        else:
            mu_init = float(mu_true)
            sigma_init = fallback_sigma
        cap_entries.append(
            CapabilityEntry(
                ENC(skill),
                mu=mu_init,
                sigma=sigma_init,
                source="explicit",
                description=skill,
            )
        )

    return UserState(
        user_id=profile["user_id"],
        capabilities=cap_entries,
        needs=_needs_from_dict(profile.get("needs", {})),
        clearance_level=0,
    )


def perturb_userstate_mu(
    state: UserState,
    mode: str,
    sigma: float,
    rng: np.random.Generator,
) -> UserState:
    """Return a deep copy of `state` with capability μ perturbed in place.

    mode == "none":     no change.
    mode == "gaussian": mu_init = clip(mu_true + N(0, sigma), 0, 1)
    mode == "uniform":  mu_init = Uniform(0, 1)  (sigma is ignored)
    """
    if mode == "none":
        return state
    perturbed = copy.deepcopy(state)
    for cap in perturbed.capabilities:
        if mode == "gaussian":
            cap.mu = float(np.clip(cap.mu + rng.normal(0.0, sigma), 0.0, 1.0))
        elif mode == "uniform":
            cap.mu = float(rng.uniform(0.0, 1.0))
        else:
            raise ValueError(f"Unknown noise mode '{mode}'")
    return perturbed


class GroundTruthFeedback(FeedbackProvider):
    """Feedback provider that grounds the reward in the candidate's TRUE
    capabilities rather than the system's current (possibly-noisy) belief.

    Without this, when the initial μ are wrong the system would get rewards
    that confirm its own mistaken belief and never recover. A bandit-style
    setup needs a fixed environment that can correct the agent's posterior;
    this class plays that role by computing M against the true UserStates
    under a fixed evaluation WorldModel.
    """

    def __init__(self, true_states_by_id: Dict[str, UserState]):
        self.true_states = true_states_by_id
        self._eval_wm = WorldModel(config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N)

    def collect(self, requester: UserState, candidate: UserState,
                task: Task, match) -> dict:
        true_v = self.true_states.get(candidate.user_id, candidate)
        true_match = self._eval_wm.compute_match(
            requester, true_v, task, use_ucb=False, round_t=1,
        )
        return dummy_user_feedback(true_match)


def build_requester_for_match(profile: dict) -> UserState:
    """Build a requester used for matching: empty capabilities so the task is
    truly delegated. Otherwise the requester's own skills collapse the gap
    inside S_cap and every candidate saturates at 1.0. Needs are preserved
    for completeness but only matter if the task has offers.
    """
    return UserState(
        user_id=profile["user_id"],
        capabilities=[],
        needs=_needs_from_dict(profile.get("needs", {})),
        clearance_level=0,
    )


def build_task(task_dict: dict) -> Task:
    reqs = [
        TaskRequirement(ENC(skill), level=float(level), constraint_type="soft", description=skill)
        for skill, level in task_dict.get("required_skills", {}).items()
    ]
    return Task(
        task_id=task_dict["task_id"],
        goal=task_dict.get("description", task_dict.get("title", task_dict["task_id"])),
        requirements=reqs,
        offers=_offers_from_dict(task_dict.get("offers", {}) or {}),
        data_clearance=0,
    )


# ---------- M_optimum (ground truth) ----------

def compute_ground_truth_M(
    requester: UserState,
    candidate_profiles: List[dict],
    task: Task,
) -> Tuple[Dict[str, float], str, float]:
    """Return {candidate_id: M_true}, optimum_id, M_optimum.

    Uses the candidates' raw capabilities with TRUE_SIGMA (≈0) and the
    neutral evaluation theta. No UCB, no learning state involved.
    """
    eval_wm = WorldModel(config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N)
    per_candidate: Dict[str, float] = {}
    for c in candidate_profiles:
        cand_state = build_user_state(c["candidate_profile"], sigma=TRUE_SIGMA)
        m = eval_wm.compute_match(requester, cand_state, task, use_ucb=False, round_t=1)
        per_candidate[cand_state.user_id] = float(m.match_score)
    optimum_id = max(per_candidate, key=per_candidate.get)
    return per_candidate, optimum_id, per_candidate[optimum_id]


# ---------- Convergence detector ----------

def _converged_at(selections: List[str], k: int) -> Optional[int]:
    """Return the 1-based round index at which the last K selections agree."""
    if len(selections) < k:
        return None
    last = selections[-1]
    for i in range(len(selections) - 1, len(selections) - k, -1):
        if selections[i] != last:
            return None
    return len(selections)  # 1-based: index of the round that closed the K-streak


def _first_optimal_round(selections: List[str], optimum_id: str) -> Optional[int]:
    """Return the 1-based round at which the optimum was first selected."""
    for i, s in enumerate(selections, start=1):
        if s == optimum_id:
            return i
    return None


def _mode_selection(selections: List[str]) -> str:
    """Return the most-frequently selected candidate id (ties broken arbitrarily)."""
    if not selections:
        return ""
    counts: Dict[str, int] = {}
    for s in selections:
        counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get)


# ---------- Condition runners ----------

def run_online_ucb(
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
    n_max: int,
    k_converge: int,
    feedback_provider: Optional[FeedbackProvider] = None,
) -> Tuple[int, str, List[str]]:
    world_model = WorldModel(config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N)
    engine = OnlineLearning(
        world_model,
        feedback_provider=feedback_provider,
        enable_dreaming=False,            # dreaming is orthogonal; off for speed
        enable_candidate_ucb=True,
        enable_weight_update=True,
        candidate_ucb_c=CANDIDATE_UCB_C,
        top_k=min(20, len(candidate_pool)),
        top_n=3,
    )
    selections: List[str] = []
    for _ in range(n_max):
        report = engine.run_one_round(
            requester=requester,
            candidate=None,
            task=task,
            candidate_pool=candidate_pool,
        )
        selections.append(report["selected_candidate_id"])
        if _converged_at(selections, k_converge) is not None:
            break
    n_rounds = _converged_at(selections, k_converge) or n_max
    return n_rounds, selections[-1], selections


def run_scap_greedy(
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
    n_max: int,
    k_converge: int,
    feedback_provider: Optional[FeedbackProvider] = None,
) -> Tuple[int, str, List[str]]:
    world_model = WorldModel(config=CFG, theta_c=SCAP_THETA_C, theta_n=SCAP_THETA_N)
    engine = OnlineLearning(
        world_model,
        feedback_provider=feedback_provider,
        enable_dreaming=False,
        enable_candidate_ucb=False,        # pure greedy on S_cap
        enable_weight_update=False,        # freeze w_n ≈ 0
        use_capability_ucb=False,          # no μ+β·σ inflation: rank on raw mu
        top_k=min(20, len(candidate_pool)),
        top_n=3,
    )
    selections: List[str] = []
    for _ in range(n_max):
        report = engine.run_one_round(
            requester=requester,
            candidate=None,
            task=task,
            candidate_pool=candidate_pool,
        )
        selections.append(report["selected_candidate_id"])
        if _converged_at(selections, k_converge) is not None:
            break
    n_rounds = _converged_at(selections, k_converge) or n_max
    return n_rounds, selections[-1], selections


def run_random(
    candidate_pool: List[UserState],
    n_max: int,
    k_converge: int,
    seed: int,
) -> Tuple[int, str, List[str]]:
    rng = random.Random(seed)
    ids = [c.user_id for c in candidate_pool]
    selections: List[str] = []
    for _ in range(n_max):
        selections.append(rng.choice(ids))
        if _converged_at(selections, k_converge) is not None:
            break
    n_rounds = _converged_at(selections, k_converge) or n_max
    return n_rounds, selections[-1], selections


def lappas_style_skill_coverage_score(candidate: UserState, task: Task) -> float:
    """Discrete weighted skill coverage for one candidate.

    This Lappas-style Skill Coverage baseline only asks whether a candidate's
    observed capability level meets each required skill level. It intentionally
    does not use task offers, candidate needs, MapScore, UCB, feedback updates,
    Dreaming, or Lappas et al.'s social-network team-formation components.
    """
    requirements = task.requirements
    if not requirements:
        return 1.0

    total_weight = sum(max(0.0, float(req.level)) for req in requirements)
    if total_weight <= 0.0:
        return 0.0

    score = 0.0
    caps_by_description = {
        cap.description: float(cap.mu)
        for cap in candidate.capabilities
        if cap.description
    }
    for req in requirements:
        weight = max(0.0, float(req.level))
        if caps_by_description.get(req.description, -1.0) >= float(req.level):
            score += weight
    return score / total_weight


def run_lappas_coverage(
    candidate_pool: List[UserState],
    task: Task,
) -> Tuple[int, str, List[str], Dict[str, float]]:
    """Single-round static top-1 selection.

    Method name in outputs: "Lappas-style Skill Coverage". This is a narrow
    external baseline for weighted discrete skill coverage, not a claim of full
    Lappas social-network team formation reproduction.
    """
    scores = {
        candidate.user_id: lappas_style_skill_coverage_score(candidate, task)
        for candidate in candidate_pool
    }
    selected_id = max(scores, key=lambda cid: (scores[cid], cid))
    return 1, selected_id, [selected_id], scores


def score_full_mapscore_static(
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
) -> Dict[str, object]:
    """Compute full static MapScore for every candidate without UCB or learning."""
    world_model = WorldModel(config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N)
    matches = {
        candidate.user_id: world_model.compute_match(
            requester, candidate, task, use_ucb=False, round_t=1,
        )
        for candidate in candidate_pool
    }
    selected_id = max(matches, key=lambda cid: (matches[cid].match_score, cid))
    return {"selected_id": selected_id, "matches": matches}


def run_mapscore_greedy(
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
) -> Tuple[int, str, List[str], Dict[str, object]]:
    """Static full MapScore top-1; no UCB, feedback update, or dreaming."""
    scored = score_full_mapscore_static(requester, candidate_pool, task)
    selected_id = str(scored["selected_id"])
    return 1, selected_id, [selected_id], scored


def _match_metrics(match) -> dict:
    return {
        "S_cap": round(float(match.s_cap), 4),
        "S_need": round(float(match.s_need), 4),
        "MapScore": round(float(match.match_score), 4),
        "sigma_gate": int(match.sigma_gate),
    }


def _distribution_stats(values: List[float]) -> dict:
    if not values:
        return {"std": 0.0, "range": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "std": round(float(np.std(arr)), 4),
        "range": round(float(np.max(arr) - np.min(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
    }


def _user_profile_from_dict(data: dict) -> UserProfile:
    return UserProfile(
        user_id=data["user_id"],
        role=data.get("role", ""),
        capabilities=data.get("capabilities", {}) or {},
        needs=data.get("needs", {}) or {},
        preferences=data.get("preferences", {}) or {},
        constraints=data.get("constraints", {}) or {},
        history_summary=data.get("history_summary"),
    )


def _task_spec_from_dict(data: dict) -> TaskSpec:
    return TaskSpec(
        task_id=data["task_id"],
        title=data.get("title", data["task_id"]),
        description=data.get("description", data.get("title", data["task_id"])),
        required_skills=data.get("required_skills", {}) or {},
        offers=data.get("offers", {}) or {},
        metadata=data.get("metadata", {}) or {},
    )


def _candidate_card_from_dict(data: dict) -> CandidateCard:
    return CandidateCard(
        candidate_id=data["candidate_id"],
        summary=data.get("summary", ""),
        highlighted_strengths=data.get("highlighted_strengths", []) or [],
        highlighted_risks=data.get("highlighted_risks", []) or [],
        explanation=data.get("explanation", ""),
    )


def build_matching_context(
    task_entry: dict,
    candidate_id: str,
) -> MatchingContext:
    task = _task_spec_from_dict(task_entry["task"])
    requester = _user_profile_from_dict(task_entry["proposer_profile"])
    candidate_entry = next(
        c for c in task_entry["candidates"]
        if c["candidate_profile"]["user_id"] == candidate_id
    )
    latents = candidate_entry.get("context_latents", {}) or {}
    return MatchingContext(
        requester=requester,
        candidate=_user_profile_from_dict(candidate_entry["candidate_profile"]),
        task=task,
        card=_candidate_card_from_dict(candidate_entry["candidate_card"]),
        history=latents.get("history", {}) or {},
        latent_requester_preferences=latents.get("latent_requester_preferences", {}) or {},
        latent_candidate_preferences=latents.get("latent_candidate_preferences", {}) or {},
        latent_interpersonal_affinity=latents.get("latent_interpersonal_affinity"),
        latent_risk_tolerance=latents.get("latent_risk_tolerance"),
        latent_opportunity_bias=latents.get("latent_opportunity_bias"),
    )


def compute_bilateral_validity(task_entry: dict, candidate_id: str) -> dict:
    """Run the existing simulator outcome/reward path for one selected pair."""
    cfg = SimulatorConfig(
        backend_type="mock",
        random_seed=42,
        persona_selection_seed=42,
        decision_mode="threshold",
        trace_verbose=False,
    )
    cfg.apply_seed()
    context = build_matching_context(task_entry, candidate_id)
    backend = RuleBasedBackend(cfg)
    bilateral = BilateralSimulator(backend, cfg).run(context)
    outcome = OutcomeSimulator(cfg).simulate(context, bilateral)
    reward = compute_reward(bilateral, outcome, cfg)
    return {
        "candidate_id": candidate_id,
        "mutual_accept_probability": round(float(bilateral.joint_accept_prob), 4),
        "completion_probability": round(float(outcome.completion_probability), 4),
        "requester_satisfaction": round(float(outcome.requester_satisfaction), 4),
        "candidate_satisfaction": round(float(outcome.candidate_satisfaction), 4),
        "total_reward": round(float(reward.total_reward), 4),
        "joint_reward": round(float(reward.feedback_reward), 4),
        "joint_action": bilateral.joint_action.value,
    }


# ---------- Per-task evaluation ----------

def evaluate_task(
    task_entry: dict,
    noise_mode: str = DEFAULT_NOISE_MODE,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    task_dict = task_entry["task"]
    proposer = task_entry["proposer_profile"]
    candidate_entries = task_entry["candidates"]

    task = build_task(task_dict)
    requester = build_requester_for_match(proposer)
    rng = rng or np.random.default_rng(NOISE_SEED)

    # Ground-truth M per candidate using true capabilities
    M_true_per_cand, optimum_id, M_optimum = compute_ground_truth_M(
        requester, candidate_entries, task
    )

    # Ground-truth UserStates (sigma≈0, true mu) — used by the feedback
    # provider so the system gets a signal that points toward the truth even
    # when its own priors are noisy.
    true_states_by_id: Dict[str, UserState] = {
        c["candidate_profile"]["user_id"]:
            build_user_state(c["candidate_profile"], sigma=TRUE_SIGMA)
        for c in candidate_entries
    }
    feedback_provider = GroundTruthFeedback(true_states_by_id)

    def rho_for(selected_id: str) -> float:
        m_sel = M_true_per_cand.get(selected_id, 0.0)
        if M_optimum <= 0.0:
            return 0.0
        return m_sel / M_optimum

    # Build fresh learning candidate pools per condition. The initial μ and
    # σ for each candidate come from ``capability_priors`` (tier-based:
    # veteran/mid/new_explicit/new_implicit), so prior noise is already
    # baked in. Any additional perturbation from --noise-mode is layered
    # on top of those priors.
    def fresh_pool() -> List[UserState]:
        pool = [build_learning_state_from_priors(c) for c in candidate_entries]
        if noise_mode == "none":
            return pool
        return [perturb_userstate_mu(c, noise_mode, noise_sigma, rng) for c in pool]

    def first_opt_or_max(sels: List[str]) -> int:
        idx = _first_optimal_round(sels, optimum_id)
        return idx if idx is not None else N_MAX_ROUNDS

    # 1) Online UCB
    n_ol, sel_ol, sels_ol = run_online_ucb(
        requester, fresh_pool(), task, N_MAX_ROUNDS, K_CONVERGE,
        feedback_provider=feedback_provider,
    )
    rho_ol_last = rho_for(sel_ol)
    rho_ol_mode = rho_for(_mode_selection(sels_ol))

    # 2) S_cap greedy
    n_sc, sel_sc, sels_sc = run_scap_greedy(
        requester, fresh_pool(), task, N_MAX_ROUNDS, K_CONVERGE,
        feedback_provider=feedback_provider,
    )
    rho_sc_last = rho_for(sel_sc)
    rho_sc_mode = rho_for(_mode_selection(sels_sc))

    # 3) Random (5 seeds)
    random_n: List[int] = []
    random_rho_last: List[float] = []
    random_rho_mode: List[float] = []
    random_first_opt: List[int] = []
    random_selections: List[str] = []
    for seed in range(RANDOM_SEEDS):
        n_r, sel_r, sels_r = run_random(fresh_pool(), N_MAX_ROUNDS, K_CONVERGE, seed=seed)
        random_n.append(n_r)
        random_rho_last.append(rho_for(sel_r))
        random_rho_mode.append(rho_for(_mode_selection(sels_r)))
        random_first_opt.append(first_opt_or_max(sels_r))
        random_selections.append(sel_r)

    # 4-5) Static baselines share the same observed pool so the diagnostics
    # compare Lappas-style coverage and full MapScore on identical inputs.
    static_pool = fresh_pool()

    # 4) Lappas-style Skill Coverage: static top-1 on observed skill priors.
    n_la, sel_la, sels_la, scores_la = run_lappas_coverage(static_pool, task)
    rho_la_last = rho_for(sel_la)
    rho_la_mode = rho_for(_mode_selection(sels_la))

    # 5) Static full MapScore greedy: gate + S_cap + S_need, no UCB/update/dreaming.
    n_mg, sel_mg, sels_mg, scored_mg = run_mapscore_greedy(requester, static_pool, task)
    rho_mg_last = rho_for(sel_mg)
    rho_mg_mode = rho_for(_mode_selection(sels_mg))
    mapscore_matches = scored_mg["matches"]
    s_need_values = [float(m.s_need) for m in mapscore_matches.values()]
    selected_pair_metrics = {
        "mapscore_greedy": _match_metrics(mapscore_matches[sel_mg]),
        "lappas_coverage": _match_metrics(mapscore_matches[sel_la]),
    }
    bilateral_validity = {
        "online_ucb": compute_bilateral_validity(task_entry, sel_ol),
        "scap_greedy": compute_bilateral_validity(task_entry, sel_sc),
        "lappas_coverage": compute_bilateral_validity(task_entry, sel_la),
        "mapscore_greedy": compute_bilateral_validity(task_entry, sel_mg),
    }

    return {
        "task_id": task_dict["task_id"],
        "title": task_dict.get("title", ""),
        "num_candidates": len(candidate_entries),
        "M_optimum": round(M_optimum, 4),
        "optimum_candidate_id": optimum_id,
        "conditions": {
            "online_ucb": {
                "n_rounds_consensus": n_ol,
                "n_rounds_first_optimal": first_opt_or_max(sels_ol),
                "selected_candidate_id": sel_ol,
                "mode_candidate_id": _mode_selection(sels_ol),
                "rho_last": round(rho_ol_last, 4),
                "rho_mode": round(rho_ol_mode, 4),
                "outcome_reward": bilateral_validity["online_ucb"],
            },
            "scap_greedy": {
                "n_rounds_consensus": n_sc,
                "n_rounds_first_optimal": first_opt_or_max(sels_sc),
                "selected_candidate_id": sel_sc,
                "mode_candidate_id": _mode_selection(sels_sc),
                "rho_last": round(rho_sc_last, 4),
                "rho_mode": round(rho_sc_mode, 4),
                "outcome_reward": bilateral_validity["scap_greedy"],
            },
            "random": {
                "n_rounds_consensus_mean": round(float(np.mean(random_n)), 4),
                "n_rounds_consensus_std": round(float(np.std(random_n)), 4),
                "n_rounds_first_optimal_mean": round(float(np.mean(random_first_opt)), 4),
                "n_rounds_first_optimal_std": round(float(np.std(random_first_opt)), 4),
                "n_rounds_per_seed": random_n,
                "rho_last_mean": round(float(np.mean(random_rho_last)), 4),
                "rho_last_std": round(float(np.std(random_rho_last)), 4),
                "rho_mode_mean": round(float(np.mean(random_rho_mode)), 4),
                "rho_mode_std": round(float(np.std(random_rho_mode)), 4),
                "rho_last_per_seed": [round(x, 4) for x in random_rho_last],
                "rho_mode_per_seed": [round(x, 4) for x in random_rho_mode],
                "selected_per_seed": random_selections,
            },
            "lappas_coverage": {
                "method_name": "Lappas-style Skill Coverage",
                "n_rounds_consensus": n_la,
                "n_rounds_first_optimal": first_opt_or_max(sels_la),
                "selected_candidate_id": sel_la,
                "mode_candidate_id": _mode_selection(sels_la),
                "rho_last": round(rho_la_last, 4),
                "rho_mode": round(rho_la_mode, 4),
                "coverage_score_selected": round(scores_la[sel_la], 4),
                "outcome_reward": bilateral_validity["lappas_coverage"],
            },
            "mapscore_greedy": {
                "method_name": "MapScore Greedy",
                "n_rounds_consensus": n_mg,
                "n_rounds_first_optimal": first_opt_or_max(sels_mg),
                "selected_candidate_id": sel_mg,
                "mode_candidate_id": _mode_selection(sels_mg),
                "rho_last": round(rho_mg_last, 4),
                "rho_mode": round(rho_mg_mode, 4),
                "outcome_reward": bilateral_validity["mapscore_greedy"],
            },
        },
        "diagnostics": {
            "s_need_pool": _distribution_stats(s_need_values),
            "mapscore_vs_lappas_top1_different": sel_mg != sel_la,
            "selected_pair_metrics": selected_pair_metrics,
            "bilateral_validity": bilateral_validity,
        },
    }


# ---------- Main ----------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise-mode", choices=("none", "gaussian", "uniform"),
        default=DEFAULT_NOISE_MODE,
        help="How to perturb the initial μ of candidate capabilities.",
    )
    parser.add_argument(
        "--noise-sigma", type=float, default=DEFAULT_NOISE_SIGMA,
        help="Std dev for gaussian noise on μ (ignored for other modes).",
    )
    parser.add_argument(
        "--noise-seed", type=int, default=NOISE_SEED,
        help="Seed for the μ-perturbation RNG so runs are reproducible.",
    )
    global N_MAX_ROUNDS, K_CONVERGE  # set before any reference below
    parser.add_argument(
        "--n-max-rounds", type=int, default=N_MAX_ROUNDS,
        help="Hard cap on the number of rounds per run.",
    )
    parser.add_argument(
        "--k-converge", type=int, default=K_CONVERGE,
        help="Consecutive identical selections required to declare consensus.",
    )
    parser.add_argument(
        "--pool-size", type=int, default=0,
        help="If > 0, subsample this many candidates per task (deterministic).",
    )
    parser.add_argument(
        "--testset",
        type=Path,
        default=TESTSET_PATH,
        help="Path to a 20-task benchmark JSON. Old v1 files remain supported; v2 files may include task.offers.",
    )
    args = parser.parse_args(argv)

    # The runners read these from module-level globals; override them now so
    # all three conditions (online_ucb, scap_greedy, random) use the same
    # budget within this invocation.
    N_MAX_ROUNDS = args.n_max_rounds
    K_CONVERGE = args.k_converge

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    testset_path = args.testset
    data = json.loads(testset_path.read_text())
    tasks = data["tasks"]
    rng = np.random.default_rng(args.noise_seed)

    noise_tag = args.noise_mode
    if args.noise_mode == "gaussian":
        noise_tag += f"{args.noise_sigma}"
    noise_tag += f"_n{N_MAX_ROUNDS}"
    if args.pool_size and args.pool_size > 0:
        noise_tag += f"_p{args.pool_size}"
    print(
        f"Loaded {len(tasks)} tasks from {testset_path}  |  "
        f"noise={args.noise_mode} sigma={args.noise_sigma} seed={args.noise_seed}  |  "
        f"n_max={N_MAX_ROUNDS} k_converge={K_CONVERGE}"
    )

    results = []
    for i, task_entry in enumerate(tasks, start=1):
        if args.pool_size and args.pool_size > 0:
            task_entry = dict(task_entry)
            task_entry["candidates"] = task_entry["candidates"][: args.pool_size]
        result = evaluate_task(
            task_entry,
            noise_mode=args.noise_mode,
            noise_sigma=args.noise_sigma,
            rng=rng,
        )
        cond = result["conditions"]
        print(
            f"[{i:>2}/{len(tasks)}] {result['task_id']}: "
            f"online_ucb(n_c={cond['online_ucb']['n_rounds_consensus']:>2}, "
            f"n_opt={cond['online_ucb']['n_rounds_first_optimal']:>2}, "
            f"rho_mode={cond['online_ucb']['rho_mode']:.3f})  "
            f"scap_greedy(n_c={cond['scap_greedy']['n_rounds_consensus']:>2}, "
            f"n_opt={cond['scap_greedy']['n_rounds_first_optimal']:>2}, "
            f"rho_mode={cond['scap_greedy']['rho_mode']:.3f})  "
            f"random(n_c={cond['random']['n_rounds_consensus_mean']:.1f}, "
            f"n_opt={cond['random']['n_rounds_first_optimal_mean']:.1f}, "
            f"rho_mode={cond['random']['rho_mode_mean']:.3f})  "
            f"lappas_coverage(n_c={cond['lappas_coverage']['n_rounds_consensus']:>2}, "
            f"n_opt={cond['lappas_coverage']['n_rounds_first_optimal']:>2}, "
            f"rho_mode={cond['lappas_coverage']['rho_mode']:.3f})  "
            f"mapscore_greedy(n_c={cond['mapscore_greedy']['n_rounds_consensus']:>2}, "
            f"n_opt={cond['mapscore_greedy']['n_rounds_first_optimal']:>2}, "
            f"rho_mode={cond['mapscore_greedy']['rho_mode']:.3f})"
        )
        results.append(result)

    def aggregate(key_path: List[str]) -> dict:
        vals = []
        for r in results:
            v = r["conditions"]
            for k in key_path:
                v = v[k]
            vals.append(v)
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    summary = {
        "online_ucb": {
            "n_rounds_consensus": aggregate(["online_ucb", "n_rounds_consensus"]),
            "n_rounds_first_optimal": aggregate(["online_ucb", "n_rounds_first_optimal"]),
            "rho_last": aggregate(["online_ucb", "rho_last"]),
            "rho_mode": aggregate(["online_ucb", "rho_mode"]),
        },
        "scap_greedy": {
            "n_rounds_consensus": aggregate(["scap_greedy", "n_rounds_consensus"]),
            "n_rounds_first_optimal": aggregate(["scap_greedy", "n_rounds_first_optimal"]),
            "rho_last": aggregate(["scap_greedy", "rho_last"]),
            "rho_mode": aggregate(["scap_greedy", "rho_mode"]),
        },
        "random": {
            "n_rounds_consensus": aggregate(["random", "n_rounds_consensus_mean"]),
            "n_rounds_first_optimal": aggregate(["random", "n_rounds_first_optimal_mean"]),
            "rho_last": aggregate(["random", "rho_last_mean"]),
            "rho_mode": aggregate(["random", "rho_mode_mean"]),
        },
        "lappas_coverage": {
            "method_name": "Lappas-style Skill Coverage",
            "n_rounds_consensus": aggregate(["lappas_coverage", "n_rounds_consensus"]),
            "n_rounds_first_optimal": aggregate(["lappas_coverage", "n_rounds_first_optimal"]),
            "rho_last": aggregate(["lappas_coverage", "rho_last"]),
            "rho_mode": aggregate(["lappas_coverage", "rho_mode"]),
        },
        "mapscore_greedy": {
            "method_name": "MapScore Greedy",
            "n_rounds_consensus": aggregate(["mapscore_greedy", "n_rounds_consensus"]),
            "n_rounds_first_optimal": aggregate(["mapscore_greedy", "n_rounds_first_optimal"]),
            "rho_last": aggregate(["mapscore_greedy", "rho_last"]),
            "rho_mode": aggregate(["mapscore_greedy", "rho_mode"]),
        },
    }

    s_need_stds = [r["diagnostics"]["s_need_pool"]["std"] for r in results]
    s_need_ranges = [r["diagnostics"]["s_need_pool"]["range"] for r in results]
    s_need_means = [r["diagnostics"]["s_need_pool"]["mean"] for r in results]
    s_need_maxes = [r["diagnostics"]["s_need_pool"]["max"] for r in results]
    top1_diff_flags = [
        bool(r["diagnostics"]["mapscore_vs_lappas_top1_different"])
        for r in results
    ]
    top1_diff_rate = float(np.mean(top1_diff_flags)) if top1_diff_flags else 0.0
    diagnostics_summary = {
        "mapscore_vs_lappas_top1_difference_rate": round(top1_diff_rate, 4),
        "s_need_pool_std": {
            "mean": round(float(np.mean(s_need_stds)), 4),
            "median": round(float(np.median(s_need_stds)), 4),
            "min": round(float(np.min(s_need_stds)), 4),
            "max": round(float(np.max(s_need_stds)), 4),
        },
        "s_need_pool_range": {
            "mean": round(float(np.mean(s_need_ranges)), 4),
            "median": round(float(np.median(s_need_ranges)), 4),
            "min": round(float(np.min(s_need_ranges)), 4),
            "max": round(float(np.max(s_need_ranges)), 4),
        },
        "s_need_pool_mean": {
            "mean": round(float(np.mean(s_need_means)), 4),
            "median": round(float(np.median(s_need_means)), 4),
            "min": round(float(np.min(s_need_means)), 4),
            "max": round(float(np.max(s_need_means)), 4),
        },
        "s_need_pool_max": {
            "mean": round(float(np.mean(s_need_maxes)), 4),
            "median": round(float(np.median(s_need_maxes)), 4),
            "min": round(float(np.min(s_need_maxes)), 4),
            "max": round(float(np.max(s_need_maxes)), 4),
        },
        "low_difference_rate_note": (
            "Difference rate is below 25%; inspect offer/need matching strength "
            "and within-pool S_need variance before changing model parameters."
            if top1_diff_rate < 0.25 else ""
        ),
    }

    outcome_fields = [
        "mutual_accept_probability",
        "completion_probability",
        "requester_satisfaction",
        "candidate_satisfaction",
        "joint_reward",
        "total_reward",
    ]
    outcome_summary = {}
    for method in ("online_ucb", "scap_greedy", "lappas_coverage", "mapscore_greedy"):
        outcome_summary[method] = {}
        for field in outcome_fields:
            vals = [
                r["diagnostics"]["bilateral_validity"][method][field]
                for r in results
            ]
            outcome_summary[method][field] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = LOGS_DIR / f"online_learning_20tasks_eval_{noise_tag}_{timestamp}.json"
    out_path.write_text(json.dumps({
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "testset": str(testset_path),
            "k_converge": K_CONVERGE,
            "n_max_rounds": N_MAX_ROUNDS,
            "random_seeds": RANDOM_SEEDS,
            "priors_source": "capability_priors (tier-based)",
            "sigma_init_fallback": SIGMA_INIT_FALLBACK,
            "eval_theta": [EVAL_THETA_C, EVAL_THETA_N],
            "noise_mode": args.noise_mode,
            "noise_sigma": args.noise_sigma,
            "noise_seed": args.noise_seed,
            "feedback_provider": "GroundTruthFeedback",
            "external_baselines": {
                "lappas_coverage": (
                    "Lappas-style Skill Coverage; discrete weighted required-skill "
                    "coverage only, not full social-network team formation."
                ),
            },
        },
        "per_task": results,
        "summary": summary,
        "diagnostics_summary": diagnostics_summary,
        "outcome_summary": outcome_summary,
    }, indent=2))

    print()
    print("=" * 80)
    print(
        f"Summary (across {len(results)} tasks)  |  "
        f"noise={args.noise_mode} sigma={args.noise_sigma}:"
    )
    for cond, stats in summary.items():
        print(
            f"  {cond:<12}  "
            f"n_consensus={stats['n_rounds_consensus']['mean']:>5.2f}  "
            f"n_first_opt={stats['n_rounds_first_optimal']['mean']:>5.2f}  "
            f"rho_mode={stats['rho_mode']['mean']:.3f}  "
            f"rho_last={stats['rho_last']['mean']:.3f}"
        )
    print(
        "Diagnostics: "
        f"mapscore_vs_lappas_diff_rate={top1_diff_rate:.3f}, "
        f"S_need_std_mean={diagnostics_summary['s_need_pool_std']['mean']:.4f}, "
        f"S_need_range_mean={diagnostics_summary['s_need_pool_range']['mean']:.4f}, "
        f"S_need_max_mean={diagnostics_summary['s_need_pool_max']['mean']:.4f}"
    )
    if top1_diff_rate < 0.25:
        print(
            "Difference rate is below 25%; not changing model parameters. "
            "Check whether benchmark offers/needs are too uniform or weak using "
            "the S_need stats above and per-task diagnostics in the JSON."
        )
    print("Outcome/reward means:")
    for method, stats in outcome_summary.items():
        print(
            f"  {method:<16} "
            f"mutual={stats['mutual_accept_probability']['mean']:.3f}  "
            f"completion={stats['completion_probability']['mean']:.3f}  "
            f"req_sat={stats['requester_satisfaction']['mean']:.3f}  "
            f"cand_sat={stats['candidate_satisfaction']['mean']:.3f}  "
            f"joint_reward={stats['joint_reward']['mean']:.3f}  "
            f"total_reward={stats['total_reward']['mean']:.3f}"
        )
    print(f"Wrote evaluation results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
