"""Preference construction for batch stable matching.

Two methods share the same feasible edge set and the same DA allocator:

* ``GaleShapley-SkillCoverage`` — discrete weighted skill coverage / offer-need fit
* ``CoWeaver-DA`` — feasibility-gated analytical MapScore / S_need

Neither method may read hidden simulator latents, oracle outcomes, or rewards
when building preferences.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure mapping-algo / Online_learning imports resolve when loaded as a package.
_ROOT = Path(__file__).resolve().parents[2]
_OL = _ROOT / "Online_learning"
_MA = _ROOT / "mapping-algo"
_TESTS = _OL / "tests"
for _p in (str(_ROOT), str(_OL), str(_MA), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from WorldModel import WorldModel  # noqa: E402
from config import MatchConfig  # noqa: E402
from encoder import SimpleEncoder  # noqa: E402
from run_20_tasks_evaluation import (  # noqa: E402
    EVAL_THETA_C,
    EVAL_THETA_N,
    TRUE_SIGMA,
    build_requester_for_match,
    build_task,
    build_user_state,
    lappas_style_skill_coverage_score,
)
from simulator.utils import offer_need_fit  # noqa: E402

ENC = SimpleEncoder(dim=64)
CFG = MatchConfig(embedding_dim=64)

METHOD_GS = "GaleShapley-SkillCoverage"
METHOD_CW = "CoWeaver-DA"

# Keys that preference construction must never read from pair / batch payloads.
FORBIDDEN_PREF_KEYS = frozenset({
    "context_latents",
    "latent_requester_preferences",
    "latent_candidate_preferences",
    "latent_interpersonal_affinity",
    "latent_risk_tolerance",
    "latent_opportunity_bias",
    "MapScore",
    "S_cap",
    "S_need",
    "outcome",
    "reward",
    "total_reward",
    "mutual_accept_probability",
    "completion_probability",
    "oracle",
})


@dataclass
class PreferenceBundle:
    method: str
    task_ids: list[str]
    candidate_ids: list[str]
    task_prefs: dict[str, list[str]]
    candidate_prefs: dict[str, list[str]]
    task_scores: dict[str, dict[str, float]]  # task -> cand -> score
    candidate_scores: dict[str, dict[str, float]]  # cand -> task -> score
    feasible_edges: set[tuple[str, str]]
    diagnostics: dict[str, Any]


def _stable_id_seed(label: str) -> int:
    """Process-stable seed contribution (avoid PYTHONHASHSEED-dependent hash())."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(label)) % 1_000_003


def _tie_break_sort(
    scored: dict[str, float],
    rng: random.Random,
) -> list[str]:
    """Sort by score descending; break ties with a seeded shuffle of equals."""
    by_score: dict[float, list[str]] = {}
    for oid, score in scored.items():
        by_score.setdefault(score, []).append(oid)
    ordered_scores = sorted(by_score.keys(), reverse=True)
    result: list[str] = []
    for score in ordered_scores:
        group = by_score[score]
        rng.shuffle(group)
        result.extend(group)
    return result


def discrete_weighted_skill_coverage(
    required_skills: dict[str, float],
    capabilities: dict[str, float],
) -> float:
    """Public GS requester preference: weighted discrete skill coverage."""
    if not required_skills:
        return 1.0
    total = sum(max(0.0, float(v)) for v in required_skills.values())
    if total <= 0.0:
        return 0.0
    covered = 0.0
    for skill, level in required_skills.items():
        if float(capabilities.get(skill, -1.0)) >= float(level):
            covered += max(0.0, float(level))
    return covered / total


def discrete_offer_need_fit(
    offers: dict[str, float],
    needs: dict[str, float],
) -> float:
    """Public GS candidate preference: simulator offer_need_fit (public fields)."""
    return float(offer_need_fit(offers, needs))


def collect_feasible_edges(batch: dict) -> set[tuple[str, str]]:
    """Shared feasible edge set from public_feasible flags (and coverage>0)."""
    edges: set[tuple[str, str]] = set()
    for task_entry in batch["tasks"]:
        tid = task_entry["task"]["task_id"]
        for pair in task_entry["pair_entries"]:
            cid = pair["candidate_id"]
            if pair.get("public_feasible", True):
                # Extra safety: coverage must be positive.
                cov = discrete_weighted_skill_coverage(
                    task_entry["task"].get("required_skills", {}) or {},
                    pair["candidate_profile"].get("capabilities", {}) or {},
                )
                if cov > 0.0:
                    edges.add((tid, cid))
    return edges


def _assert_no_hidden_keys(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_PREF_KEYS:
                raise AssertionError(f"Preference input contains forbidden key {path}.{k}")
            _assert_no_hidden_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_hidden_keys(v, f"{path}[{i}]")


def _public_pair_view(pair: dict) -> dict:
    """Strip hidden fields before preference scoring."""
    return {
        "candidate_id": pair["candidate_id"],
        "candidate_profile": {
            "user_id": pair["candidate_profile"]["user_id"],
            "role": pair["candidate_profile"].get("role", ""),
            "capabilities": pair["candidate_profile"].get("capabilities", {}) or {},
            "needs": pair["candidate_profile"].get("needs", {}) or {},
            "preferences": pair["candidate_profile"].get("preferences", {}) or {},
            "constraints": pair["candidate_profile"].get("constraints", {}) or {},
            "history_summary": pair["candidate_profile"].get("history_summary"),
        },
        "candidate_card": pair.get("candidate_card", {}),
        "public_feasible": pair.get("public_feasible", True),
    }


def build_gs_preferences(
    batch: dict,
    feasible_edges: set[tuple[str, str]],
    *,
    tie_break_seed: int = 0,
) -> PreferenceBundle:
    """GaleShapley-SkillCoverage preferences (no MapScore)."""
    task_ids = [t["task"]["task_id"] for t in batch["tasks"]]
    candidate_ids = [
        c["candidate_profile"]["user_id"] for c in batch["shared_candidates"]
    ]

    task_scores: dict[str, dict[str, float]] = {tid: {} for tid in task_ids}
    cand_scores: dict[str, dict[str, float]] = {cid: {} for cid in candidate_ids}

    for task_entry in batch["tasks"]:
        task = task_entry["task"]
        tid = task["task_id"]
        # Public-only view for preference construction.
        public_task = {
            "task_id": task["task_id"],
            "required_skills": task.get("required_skills", {}) or {},
            "offers": task.get("offers", {}) or {},
            "metadata": task.get("metadata", {}) or {},
        }
        _assert_no_hidden_keys(public_task)
        for pair in task_entry["pair_entries"]:
            pub = _public_pair_view(pair)
            _assert_no_hidden_keys(pub)
            cid = pub["candidate_id"]
            if (tid, cid) not in feasible_edges:
                continue
            req_score = discrete_weighted_skill_coverage(
                public_task["required_skills"],
                pub["candidate_profile"]["capabilities"],
            )
            cand_score = discrete_offer_need_fit(
                public_task["offers"],
                pub["candidate_profile"]["needs"],
            )
            task_scores[tid][cid] = float(req_score)
            cand_scores[cid][tid] = float(cand_score)

    task_prefs = {
        tid: _tie_break_sort(scores, random.Random(tie_break_seed + _stable_id_seed(tid)))
        for tid, scores in task_scores.items()
    }
    candidate_prefs = {
        cid: _tie_break_sort(scores, random.Random(tie_break_seed + _stable_id_seed(cid)))
        for cid, scores in cand_scores.items()
    }

    return PreferenceBundle(
        method=METHOD_GS,
        task_ids=task_ids,
        candidate_ids=candidate_ids,
        task_prefs=task_prefs,
        candidate_prefs=candidate_prefs,
        task_scores=task_scores,
        candidate_scores=cand_scores,
        feasible_edges=set(feasible_edges),
        diagnostics={"uses_mapscore": False, "uses_hidden_latents": False},
    )


def build_coweaver_preferences(
    batch: dict,
    feasible_edges: set[tuple[str, str]],
    *,
    tie_break_seed: int = 0,
) -> PreferenceBundle:
    """CoWeaver-DA: MapScore (no UCB) for tasks; S_need for candidates."""
    task_ids = [t["task"]["task_id"] for t in batch["tasks"]]
    candidate_ids = [
        c["candidate_profile"]["user_id"] for c in batch["shared_candidates"]
    ]
    world_model = WorldModel(config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N)

    task_scores: dict[str, dict[str, float]] = {tid: {} for tid in task_ids}
    cand_scores: dict[str, dict[str, float]] = {cid: {} for cid in candidate_ids}

    for task_entry in batch["tasks"]:
        task_dict = {
            "task_id": task_entry["task"]["task_id"],
            "title": task_entry["task"].get("title", ""),
            "description": task_entry["task"].get("description", ""),
            "required_skills": task_entry["task"].get("required_skills", {}) or {},
            "offers": task_entry["task"].get("offers", {}) or {},
            "metadata": task_entry["task"].get("metadata", {}) or {},
        }
        proposer = {
            "user_id": task_entry["proposer_profile"]["user_id"],
            "needs": task_entry["proposer_profile"].get("needs", {}) or {},
            "capabilities": {},  # matching uses empty caps via build_requester_for_match
            "role": task_entry["proposer_profile"].get("role", "requester"),
        }
        _assert_no_hidden_keys(task_dict)
        _assert_no_hidden_keys(proposer)

        task = build_task(task_dict)
        requester = build_requester_for_match(proposer)
        tid = task_dict["task_id"]

        for pair in task_entry["pair_entries"]:
            pub = _public_pair_view(pair)
            _assert_no_hidden_keys(pub)
            cid = pub["candidate_id"]
            if (tid, cid) not in feasible_edges:
                continue
            cand_state = build_user_state(pub["candidate_profile"], sigma=TRUE_SIGMA)
            match = world_model.compute_match(
                requester, cand_state, task, use_ucb=False, round_t=1,
            )
            if int(match.sigma_gate) == 0 or float(match.match_score) <= 0.0:
                # Treat as infeasible for this method — but shared edge set already
                # filtered; soft-gate path usually passes. Skip zero/failed scores.
                continue
            task_scores[tid][cid] = float(match.match_score)
            cand_scores[cid][tid] = float(match.s_need)

    # Candidate rejects tasks with non-positive S_need.
    for cid in list(cand_scores.keys()):
        cand_scores[cid] = {
            tid: s for tid, s in cand_scores[cid].items() if s > 0.0
        }
    # Drop edges candidates refuse.
    for tid in list(task_scores.keys()):
        task_scores[tid] = {
            cid: s
            for cid, s in task_scores[tid].items()
            if tid in cand_scores.get(cid, {})
        }

    task_prefs = {
        tid: _tie_break_sort(scores, random.Random(tie_break_seed + _stable_id_seed(tid)))
        for tid, scores in task_scores.items()
    }
    candidate_prefs = {
        cid: _tie_break_sort(scores, random.Random(tie_break_seed + _stable_id_seed(cid)))
        for cid, scores in cand_scores.items()
    }

    return PreferenceBundle(
        method=METHOD_CW,
        task_ids=task_ids,
        candidate_ids=candidate_ids,
        task_prefs=task_prefs,
        candidate_prefs=candidate_prefs,
        task_scores=task_scores,
        candidate_scores=cand_scores,
        feasible_edges=set(feasible_edges),
        diagnostics={
            "uses_mapscore": True,
            "uses_ucb": False,
            "uses_dreaming": False,
            "uses_hidden_latents": False,
            "theta_c": EVAL_THETA_C,
            "theta_n": EVAL_THETA_N,
        },
    )


def build_preferences(
    method: str,
    batch: dict,
    *,
    feasible_edges: set[tuple[str, str]] | None = None,
    tie_break_seed: int = 0,
) -> PreferenceBundle:
    edges = feasible_edges if feasible_edges is not None else collect_feasible_edges(batch)
    if method == METHOD_GS:
        return build_gs_preferences(batch, edges, tie_break_seed=tie_break_seed)
    if method == METHOD_CW:
        return build_coweaver_preferences(batch, edges, tie_break_seed=tie_break_seed)
    raise ValueError(f"Unknown method: {method}")


# Re-export helper used by tests to confirm GS does not call MapScore via coverage path.
def gs_coverage_via_lappas_task(candidate_profile: dict, task_dict: dict) -> float:
    """Optional consistency check against the Lappas helper on UserState/Task."""
    cand = build_user_state(candidate_profile, sigma=TRUE_SIGMA)
    task = build_task(task_dict)
    return float(lappas_style_skill_coverage_score(cand, task))
