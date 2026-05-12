"""Evaluate OnlineLearning against two baselines on simulator/20_Tasks_Testset.json.

Three conditions are run per task:
  1. online_ucb   — full pipeline with candidate-level UCB (no dreaming for speed)
  2. random       — pick a candidate uniformly at random each round (5 seeds)
  3. scap_greedy  — S_cap only (w_n forced to 0, no candidate UCB, no weight SGD)

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
    TaskRequirement,
    UserState,
)
from encoder import SimpleEncoder  # noqa: E402
from feedback_provider import FeedbackProvider  # noqa: E402
from ol_utils import dummy_user_feedback  # noqa: E402


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
        offers=[],
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
            },
            "scap_greedy": {
                "n_rounds_consensus": n_sc,
                "n_rounds_first_optimal": first_opt_or_max(sels_sc),
                "selected_candidate_id": sel_sc,
                "mode_candidate_id": _mode_selection(sels_sc),
                "rho_last": round(rho_sc_last, 4),
                "rho_mode": round(rho_sc_mode, 4),
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
    args = parser.parse_args(argv)

    # The runners read these from module-level globals; override them now so
    # all three conditions (online_ucb, scap_greedy, random) use the same
    # budget within this invocation.
    N_MAX_ROUNDS = args.n_max_rounds
    K_CONVERGE = args.k_converge

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(TESTSET_PATH.read_text())
    tasks = data["tasks"]
    rng = np.random.default_rng(args.noise_seed)

    noise_tag = args.noise_mode
    if args.noise_mode == "gaussian":
        noise_tag += f"{args.noise_sigma}"
    noise_tag += f"_n{N_MAX_ROUNDS}"
    if args.pool_size and args.pool_size > 0:
        noise_tag += f"_p{args.pool_size}"
    print(
        f"Loaded {len(tasks)} tasks from {TESTSET_PATH}  |  "
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
            f"rho_mode={cond['random']['rho_mode_mean']:.3f})"
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
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = LOGS_DIR / f"online_learning_20tasks_eval_{noise_tag}_{timestamp}.json"
    out_path.write_text(json.dumps({
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "testset": str(TESTSET_PATH),
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
        },
        "per_task": results,
        "summary": summary,
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
    print(f"Wrote evaluation results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
