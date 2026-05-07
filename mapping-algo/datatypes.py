"""Core data structures — corresponds to §4A.1 of clawbot-nips.md"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Node state s_v = (Cap_v, Need_v) ──────────────────────────────────

@dataclass
class CapabilityEntry:
    """Single capability — element of Cap_v (§2.1).

    (embedding, mu, sigma) triple.
    mu  : proficiency expectation (Bayesian posterior mean)
    sigma: uncertainty (Bayesian posterior std), drives UCB exploration
    """
    embedding: np.ndarray       # shape (d,)
    mu: float                   # p_i^v ∈ [0,1]
    sigma: float                # σ_i^v ≥ 0, init 1.0
    source: str = "explicit"    # "explicit" | "implicit" | "meta"
    description: str = ""

    def ucb(self, beta: float) -> float:
        """UCB optimistic estimate: μ̃ = min(μ + β·σ, 1.0)."""
        return min(self.mu + beta * self.sigma, 1.0)


@dataclass
class NeedEntry:
    """Single need — element of Need_v (§2.2)."""
    embedding: np.ndarray       # shape (d,)
    intensity: float            # n_l^v ∈ [0,1]
    description: str = ""


@dataclass
class UserState:
    """Node state s_v = (Cap_v, Need_v) — output of MBRL Layer 1."""
    user_id: str
    capabilities: List[CapabilityEntry] = field(default_factory=list)
    needs: List[NeedEntry] = field(default_factory=list)
    clearance_level: int = 0    # for gate check


# ── Task T = (G_T, Q_T, O_T) ─────────────────────────────────────────

@dataclass
class TaskRequirement:
    """Single task requirement — element of Q_T (§1.2)."""
    embedding: np.ndarray       # shape (d,)
    level: float                # q_j ∈ [0,1]
    constraint_type: str        # "hard" | "soft"
    description: str = ""


@dataclass
class TaskOffer:
    """Single task offer — element of O_T (§3.2)."""
    embedding: np.ndarray       # shape (d,)
    strength: float             # o_r ∈ [0,1]
    source: str = "explicit"    # "explicit" | "inferred"
    description: str = ""


@dataclass
class Task:
    """Task T = (G_T, Q_T, O_T)."""
    task_id: str
    goal: str
    requirements: List[TaskRequirement] = field(default_factory=list)
    offers: List[TaskOffer] = field(default_factory=list)
    data_clearance: int = 0


# ── Output structures ─────────────────────────────────────────────────

@dataclass
class GapDetail:
    """Per-soft-requirement gap coverage detail."""
    req_description: str
    q_j: float
    p_tilde_u: float
    gap: float
    p_tilde_v: float
    coverage: float
    coverage_pct: str

    def to_dict(self) -> dict:
        return {
            "gap": round(self.gap, 4),
            "covered": round(self.coverage, 4),
            "coverage": self.coverage_pct,
        }


@dataclass
class NeedDetail:
    """Per-need satisfaction detail."""
    need_description: str
    need_intensity: float
    offer_matched: float
    satisfied: float
    satisfaction_pct: str

    def to_dict(self) -> dict:
        return {
            "need": round(self.need_intensity, 4),
            "offer_matched": round(self.offer_matched, 4),
            "satisfied": self.satisfaction_pct,
        }


@dataclass
class MatchResult:
    """Layer 2 output — per-candidate match result."""
    candidate_id: str
    match_score: float
    s_cap: float
    s_need: float
    w_c: float
    w_n: float
    sigma_gate: int
    gate_fail_reason: str = ""
    gap_details: List[GapDetail] = field(default_factory=list)
    need_details: List[NeedDetail] = field(default_factory=list)
    ucb_bonus: float = 0.0

    def to_dict(self) -> dict:
        result: dict = {
            "candidate_id": self.candidate_id,
            "match_score": round(self.match_score, 4),
            "components": {
                "S_cap": round(self.s_cap, 4),
                "S_need": round(self.s_need, 4),
                "w_c": round(self.w_c, 4),
                "w_n": round(self.w_n, 4),
                "sigma": self.sigma_gate,
            },
            "gap_coverage_detail": {
                g.req_description: g.to_dict() for g in self.gap_details
            },
            "need_satisfaction_detail": {
                n.need_description: n.to_dict() for n in self.need_details
            },
        }
        if self.ucb_bonus > 0:
            result["ucb_bonus"] = round(self.ucb_bonus, 4)
        if self.gate_fail_reason:
            result["gate_fail_reason"] = self.gate_fail_reason
        return result


@dataclass
class TeamResult:
    """Layer 3.1 output — 1-N team building result."""
    team_members: List[str]
    collective_coverage: float
    residual_gaps: Dict[str, float]
    selection_order: List[Tuple[str, float]] = field(default_factory=list)
    per_member: List[MatchResult] = field(default_factory=list)
    termination_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "team": {
                "members": self.team_members,
                "selection_order": [
                    {"id": uid, "marginal_gain": round(g, 4)}
                    for uid, g in self.selection_order
                ],
                "collective_coverage": round(self.collective_coverage, 4),
                "residual_gaps": {
                    k: round(v, 4) for k, v in self.residual_gaps.items()
                },
                "team_size": len(self.team_members),
                "termination_reason": self.termination_reason,
            },
            "per_member": [m.to_dict() for m in self.per_member],
        }
