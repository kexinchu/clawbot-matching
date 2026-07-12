"""Evaluation metrics for batch stable matching."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_OL = _ROOT / "Online_learning"
_MA = _ROOT / "mapping-algo"
_TESTS = _OL / "tests"
for _p in (str(_ROOT), str(_OL), str(_MA), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulator.config import SimulatorConfig
from simulator.bilateral_simulator import BilateralSimulator
from simulator.mock_backend import RuleBasedBackend
from simulator.outcome_simulator import OutcomeSimulator
from simulator.reward import compute_reward

from Experiments.stable_matching.batch_generator_utils import (
    as_individual_task_entry,
)
from Experiments.stable_matching.deferred_acceptance import (
    count_blocking_pairs,
    is_stable_matching,
)
from Experiments.stable_matching.preferences import (
    METHOD_CW,
    METHOD_GS,
    PreferenceBundle,
    discrete_offer_need_fit,
    discrete_weighted_skill_coverage,
)
from run_20_tasks_evaluation import (  # noqa: E402
    build_matching_context,
)


def kendall_tau(rank_a: list[str], rank_b: list[str]) -> float:
    """Kendall τ over the intersection of ranked ids (pairwise concordant)."""
    common = [x for x in rank_a if x in set(rank_b)]
    if len(common) < 2:
        return 1.0
    pos_a = {x: i for i, x in enumerate(rank_a)}
    pos_b = {x: i for i, x in enumerate(rank_b)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            u, v = common[i], common[j]
            sign_a = pos_a[u] - pos_a[v]
            sign_b = pos_b[u] - pos_b[v]
            if sign_a == 0 or sign_b == 0:
                continue
            if (sign_a > 0) == (sign_b > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def mean_assigned_rank(
    matching: dict[str, str | None],
    prefs: dict[str, list[str]],
) -> float:
    ranks = []
    for tid, cid in matching.items():
        if cid is None:
            continue
        plist = prefs.get(tid, [])
        if cid in plist:
            ranks.append(plist.index(cid) + 1)  # 1-based
    return float(np.mean(ranks)) if ranks else float("nan")


def mean_candidate_rank_of_task(
    matching: dict[str, str | None],
    candidate_prefs: dict[str, list[str]],
) -> float:
    ranks = []
    for tid, cid in matching.items():
        if cid is None:
            continue
        plist = candidate_prefs.get(cid, [])
        if tid in plist:
            ranks.append(plist.index(tid) + 1)
    return float(np.mean(ranks)) if ranks else float("nan")


def simulate_pair_outcome(
    batch: dict,
    task_id: str,
    candidate_id: str,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Hidden oracle outcome via BilateralSimulator → OutcomeSimulator → reward."""
    task_entry = as_individual_task_entry(batch, task_id, candidate_id)
    cfg = SimulatorConfig(
        backend_type="mock",
        random_seed=seed,
        persona_selection_seed=seed,
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
        "task_id": task_id,
        "candidate_id": candidate_id,
        "mutual_accept_probability": float(bilateral.joint_accept_prob),
        "completion_probability": float(outcome.completion_probability),
        "requester_satisfaction": float(outcome.requester_satisfaction),
        "candidate_satisfaction": float(outcome.candidate_satisfaction),
        "joint_reward": float(reward.feedback_reward),
        "total_reward": float(reward.total_reward),
        "joint_action": bilateral.joint_action.value,
    }


def build_oracle_utilities(batch: dict, *, seed: int = 42) -> dict[tuple[str, str], float]:
    """U*(i,j) = hidden total_reward for every public-feasible pair."""
    utilities: dict[tuple[str, str], float] = {}
    for task_entry in batch["tasks"]:
        tid = task_entry["task"]["task_id"]
        for pair in task_entry["pair_entries"]:
            if not pair.get("public_feasible", True):
                continue
            cid = pair["candidate_id"]
            out = simulate_pair_outcome(batch, tid, cid, seed=seed)
            utilities[(tid, cid)] = float(out["total_reward"])
    return utilities


def oracle_max_weight_assignment(
    task_ids: list[str],
    candidate_ids: list[str],
    utilities: dict[tuple[str, str], float],
    *,
    candidate_capacity: int = 1,
) -> tuple[dict[str, str | None], float]:
    """Exact max-weight many-to-one assignment via exhaustive search (small n).

    Explicitly an oracle upper bound — must not feed GS / CoWeaver selection.
    """
    # Expand candidate slots for capacity > 1.
    slots: list[str] = []
    for cid in candidate_ids:
        for k in range(candidate_capacity):
            slots.append(f"{cid}#slot{k}")

    best_matching: dict[str, str | None] = {tid: None for tid in task_ids}
    best_value = float("-inf")

    def rec(
        i: int,
        used_slots: set[str],
        current: dict[str, str | None],
        value: float,
    ) -> None:
        nonlocal best_matching, best_value
        if i == len(task_ids):
            if value > best_value:
                best_value = value
                best_matching = dict(current)
            return
        tid = task_ids[i]
        # Option: leave unmatched.
        current[tid] = None
        rec(i + 1, used_slots, current, value)
        for slot in slots:
            if slot in used_slots:
                continue
            cid = slot.split("#slot")[0]
            u = utilities.get((tid, cid))
            if u is None:
                continue
            used_slots.add(slot)
            current[tid] = cid
            rec(i + 1, used_slots, current, value + u)
            used_slots.remove(slot)
            current[tid] = None

    rec(0, set(), {tid: None for tid in task_ids}, 0.0)
    if best_value == float("-inf"):
        best_value = 0.0
    return best_matching, float(best_value)


def outcome_derived_prefs(
    utilities: dict[tuple[str, str], float],
    task_ids: list[str],
    candidate_ids: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Hidden preference lists induced by oracle total_reward (for blocking-pair rate)."""
    task_prefs: dict[str, list[str]] = {}
    for tid in task_ids:
        scored = {
            cid: utilities[(tid, cid)]
            for cid in candidate_ids
            if (tid, cid) in utilities
        }
        task_prefs[tid] = sorted(scored, key=lambda c: (scored[c], c), reverse=True)

    cand_prefs: dict[str, list[str]] = {}
    for cid in candidate_ids:
        scored = {
            tid: utilities[(tid, cid)]
            for tid in task_ids
            if (tid, cid) in utilities
        }
        cand_prefs[cid] = sorted(scored, key=lambda t: (scored[t], t), reverse=True)
    return task_prefs, cand_prefs


def evaluate_matching_outcomes(
    batch: dict,
    matching: dict[str, str | None],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    paired = [(tid, cid) for tid, cid in matching.items() if cid is not None]
    outcomes = [simulate_pair_outcome(batch, tid, cid, seed=seed) for tid, cid in paired]
    if not outcomes:
        return {
            "n_matched": 0,
            "mean_mutual_accept_probability": 0.0,
            "mean_completion_probability": 0.0,
            "mean_requester_satisfaction": 0.0,
            "mean_candidate_satisfaction": 0.0,
            "mean_joint_reward": 0.0,
            "mean_total_reward": 0.0,
            "total_batch_reward": 0.0,
            "min_matched_pair_total_reward": 0.0,
            "pair_outcomes": [],
        }

    totals = [o["total_reward"] for o in outcomes]
    return {
        "n_matched": len(outcomes),
        "mean_mutual_accept_probability": float(np.mean([o["mutual_accept_probability"] for o in outcomes])),
        "mean_completion_probability": float(np.mean([o["completion_probability"] for o in outcomes])),
        "mean_requester_satisfaction": float(np.mean([o["requester_satisfaction"] for o in outcomes])),
        "mean_candidate_satisfaction": float(np.mean([o["candidate_satisfaction"] for o in outcomes])),
        "mean_joint_reward": float(np.mean([o["joint_reward"] for o in outcomes])),
        "mean_total_reward": float(np.mean(totals)),
        "total_batch_reward": float(np.sum(totals)),
        "min_matched_pair_total_reward": float(np.min(totals)),
        "pair_outcomes": outcomes,
    }


def allocation_metrics(
    batch: dict,
    matching: dict[str, str | None],
    prefs: PreferenceBundle,
    *,
    candidate_capacity: int,
    oracle_task_prefs: dict[str, list[str]] | None = None,
    oracle_cand_prefs: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    n_tasks = len(batch["tasks"])
    n_cands = len(batch["shared_candidates"])
    matched = {tid: cid for tid, cid in matching.items() if cid is not None}
    n_matched = len(matched)

    # Feasible match rate among matched edges.
    gate_violations = 0
    feasible_matched = 0
    for tid, cid in matched.items():
        if (tid, cid) in prefs.feasible_edges:
            feasible_matched += 1
        else:
            gate_violations += 1

    utilized = len({cid for cid in matched.values()})
    stable = is_stable_matching(
        matching,
        prefs.task_prefs,
        prefs.candidate_prefs,
        candidate_capacity=candidate_capacity,
    )
    own_blocking = count_blocking_pairs(
        matching,
        prefs.task_prefs,
        prefs.candidate_prefs,
        candidate_capacity=candidate_capacity,
    )

    true_blocking_rate = float("nan")
    if oracle_task_prefs is not None and oracle_cand_prefs is not None:
        n_block = count_blocking_pairs(
            matching,
            oracle_task_prefs,
            oracle_cand_prefs,
            candidate_capacity=candidate_capacity,
        )
        # Normalize by number of possible task-candidate pairs with oracle prefs.
        n_possible = sum(len(v) for v in oracle_task_prefs.values())
        true_blocking_rate = (n_block / n_possible) if n_possible else 0.0

    return {
        "matched_task_rate": n_matched / n_tasks if n_tasks else 0.0,
        "unmatched_task_rate": (n_tasks - n_matched) / n_tasks if n_tasks else 0.0,
        "candidate_utilization": utilized / n_cands if n_cands else 0.0,
        "feasible_match_rate": feasible_matched / n_matched if n_matched else 1.0,
        "gate_violation_count": gate_violations,
        "mean_requester_rank": mean_assigned_rank(matching, prefs.task_prefs),
        "mean_candidate_rank": mean_candidate_rank_of_task(matching, prefs.candidate_prefs),
        "assignment_stable_under_own_prefs": bool(stable),
        "own_blocking_pair_count": own_blocking,
        "true_blocking_pair_rate": true_blocking_rate,
    }


def preference_diagnostics(
    gs: PreferenceBundle,
    cw: PreferenceBundle,
    gs_matching: dict[str, str | None],
    cw_matching: dict[str, str | None],
    batch: dict,
) -> dict[str, Any]:
    taus = []
    for tid in gs.task_ids:
        taus.append(kendall_tau(gs.task_prefs.get(tid, []), cw.task_prefs.get(tid, [])))

    n_diff = sum(
        1 for tid in gs.task_ids if gs_matching.get(tid) != cw_matching.get(tid)
    )

    def _matched_mean(matching, score_fn):
        vals = []
        for tid, cid in matching.items():
            if cid is None:
                continue
            te = next(t for t in batch["tasks"] if t["task"]["task_id"] == tid)
            pe = next(p for p in te["pair_entries"] if p["candidate_id"] == cid)
            vals.append(score_fn(te, pe))
        return float(np.mean(vals)) if vals else float("nan")

    def coverage(te, pe):
        return discrete_weighted_skill_coverage(
            te["task"].get("required_skills", {}) or {},
            pe["candidate_profile"].get("capabilities", {}) or {},
        )

    def need_fit(te, pe):
        return discrete_offer_need_fit(
            te["task"].get("offers", {}) or {},
            pe["candidate_profile"].get("needs", {}) or {},
        )

    def sneed_from_cw(te, pe):
        tid, cid = te["task"]["task_id"], pe["candidate_id"]
        return float(cw.candidate_scores.get(cid, {}).get(tid, float("nan")))

    return {
        "mean_requester_ranking_kendall_tau": float(np.mean(taus)) if taus else 1.0,
        "n_tasks_assignment_differ": n_diff,
        "gs_mean_matched_coverage": _matched_mean(gs_matching, coverage),
        "cw_mean_matched_coverage": _matched_mean(cw_matching, coverage),
        "gs_mean_matched_offer_need_fit": _matched_mean(gs_matching, need_fit),
        "cw_mean_matched_offer_need_fit": _matched_mean(cw_matching, need_fit),
        "gs_mean_matched_s_need": _matched_mean(gs_matching, sneed_from_cw),
        "cw_mean_matched_s_need": _matched_mean(cw_matching, sneed_from_cw),
        "cw_prefers_higher_sneed_than_gs": (
            _matched_mean(cw_matching, sneed_from_cw) >= _matched_mean(gs_matching, sneed_from_cw)
            if math.isfinite(_matched_mean(cw_matching, sneed_from_cw))
            and math.isfinite(_matched_mean(gs_matching, sneed_from_cw))
            else None
        ),
        "gs_prefers_higher_coverage_than_cw": (
            _matched_mean(gs_matching, coverage) >= _matched_mean(cw_matching, coverage)
        ),
    }


def mean_std(values: Iterable[float]) -> dict[str, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}
