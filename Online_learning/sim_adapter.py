"""Adapter: new types (UserState/Task/MatchResult) → simulator types.

The simulator uses dict-based capabilities/needs and a flat skills schema.
We translate at the boundary so the simulator stays a self-contained black box;
its outputs (decisions, outcome) flow back as a plain dict that L5 already
understands (r_u, r_v, n_rounds, f_completion, stars_u, stars_v).

This is one-way: UserState → simulator.UserProfile.
The simulator never writes back into UserState — that is L5.2's job.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [
    os.path.join(_HERE, '..'),                  # repo root → simulator package
    os.path.join(_HERE, '..', 'mapping-algo'),  # datatypes
]:
    if _p not in sys.path:
        sys.path.append(_p)

from typing import List

from datatypes import (
    UserState, Task, MatchResult, GapDetail,
)
import simulator as sim


def to_sim_userprofile(state: UserState, role: str = "user") -> sim.UserProfile:
    """UserState → simulator.UserProfile.

    capabilities: List[CapabilityEntry] → {description: mu}
    needs:        List[NeedEntry]       → {description: intensity}
    """
    return sim.UserProfile(
        user_id=state.user_id,
        role=role,
        capabilities={c.description: float(c.mu) for c in state.capabilities},
        needs={n.description: float(n.intensity) for n in state.needs},
        preferences={},
        constraints={"clearance_level": int(state.clearance_level)},
        history_summary=None,
    )


def to_sim_taskspec(task: Task) -> sim.TaskSpec:
    """Task → simulator.TaskSpec.

    Only soft requirements feed required_skills. Hard requirements are
    enforced upstream by L2's gate, so the simulator never sees them.
    """
    soft = [r for r in task.requirements if r.constraint_type == "soft"]
    return sim.TaskSpec(
        task_id=task.task_id,
        title=task.goal[:60] if task.goal else task.task_id,
        description=task.goal,
        required_skills={r.description: float(r.level) for r in soft},
        metadata={
            "data_clearance": int(task.data_clearance),
            "n_offers": len(task.offers),
        },
    )


def build_candidate_card(
    candidate: UserState,
    task: Task,
    match: MatchResult,
) -> sim.CandidateCard:
    """Synthesize a CandidateCard from L2's MatchResult.

    Strengths: requirements where the candidate's coverage is high.
    Risks:    requirements where a meaningful gap remains.
    """
    strengths: List[str] = []
    risks: List[str] = []

    for g in match.gap_details:
        if g.gap < 1e-6:
            continue
        coverage_ratio = g.coverage / (g.gap + 1e-8)
        if coverage_ratio >= 0.7:
            strengths.append(
                f"Covers '{g.req_description}' at {g.coverage_pct} of the gap"
            )
        elif coverage_ratio < 0.4:
            risks.append(
                f"Limited coverage of '{g.req_description}' "
                f"(gap={g.gap:.2f}, covered={g.coverage:.2f})"
            )

    if not strengths:
        strengths.append("No standout strengths flagged by analytical match")
    if not risks and match.match_score < 0.6:
        risks.append("Overall match score is moderate")

    summary = (
        f"{candidate.user_id}: M={match.match_score:.2f} "
        f"(S_cap={match.s_cap:.2f}, S_need={match.s_need:.2f})"
    )

    explanation = (
        f"Analytical match score {match.match_score:.4f} from L2 with "
        f"weights w_c={match.w_c:.2f}, w_n={match.w_n:.2f}. "
        f"Gate σ={match.sigma_gate}."
    )

    return sim.CandidateCard(
        candidate_id=candidate.user_id,
        summary=summary,
        highlighted_strengths=strengths,
        highlighted_risks=risks,
        explanation=explanation,
    )


def build_matching_context(
    requester: UserState,
    candidate: UserState,
    task: Task,
    match: MatchResult,
    history: dict | None = None,
) -> sim.MatchingContext:
    """Pack L1+L2 outputs into a simulator MatchingContext.

    Latent signals are intentionally left None here. Inject them at the
    feedback-provider level if the experiment needs them (deterministic
    seeding by (requester_id, candidate_id) is recommended).
    """
    return sim.MatchingContext(
        requester=to_sim_userprofile(requester, role="requester"),
        candidate=to_sim_userprofile(candidate, role="candidate"),
        task=to_sim_taskspec(task),
        card=build_candidate_card(candidate, task, match),
        history=history,
    )
