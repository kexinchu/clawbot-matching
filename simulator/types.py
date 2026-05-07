"""
simulator/types.py
==================
All dataclass definitions for the bilateral matching simulator.
"""
from __future__ import annotations

__all__ = [
    "UserProfile",
    "TaskSpec",
    "CandidateCard",
    "MatchingContext",
    "PersonaOpinion",
    "DeliberationTrace",
    "PersonaImportance",
    "DecisionResult",
    "BilateralDecisionResult",
    "OutcomeResult",
    "RewardResult",
    "PersonaSpec",
    "PersonaRole",
    "SideType",
    "Action",
]

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SideType(Enum):
    REQUESTER = "requester"
    CANDIDATE = "candidate"


class Action(Enum):
    ACCEPT = "accept"
    SKIP = "skip"
    REJECT = "reject"


class JointAction(Enum):
    MUTUAL_ACCEPT = "mutual_accept"
    REQUESTER_ACCEPT_CANDIDATE_SKIP = "requester_accept_candidate_skip"
    REQUESTER_SKIP_CANDIDATE_ACCEPT = "requester_skip_candidate_accept"
    MUTUAL_SKIP = "mutual_skip"
    ONE_REJECT = "one_reject"
    MUTUAL_REJECT = "mutual_reject"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    user_id: str
    role: str
    capabilities: dict[str, float] = field(default_factory=dict)  # skill -> level [0,1]
    needs: dict[str, float] = field(default_factory=dict)          # need -> intensity [0,1]
    preferences: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    history_summary: str | None = None


@dataclass
class TaskSpec:
    task_id: str
    title: str
    description: str
    required_skills: dict[str, float] = field(default_factory=dict)  # skill -> min_level
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateCard:
    candidate_id: str
    summary: str
    highlighted_strengths: list[str] = field(default_factory=list)
    highlighted_risks: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class MatchingContext:
    requester: UserProfile
    candidate: UserProfile
    task: TaskSpec
    card: CandidateCard
    history: dict[str, Any] | None = None
    # ── Layer-2-invisible latent signals ──────────────────────────────────
    # These signals influence soft-persona scoring (e.g. _collab_style, _trust_safety)
    # but are NOT visible to Layer 2 as structural features.  They reduce the
    # self-fulfilling loop risk by injecting variance that Layer 2 cannot directly
    # observe or reverse-engineer from the requester/candidate/task fields alone.
    latent_requester_preferences: dict[str, float] | None = None  # e.g. {"formal": 0.8, "fastpaced": 0.3}
    latent_candidate_preferences: dict[str, float] | None = None  # e.g. {"creative": 0.9, "structured": 0.4}
    latent_interpersonal_affinity: float | None = None             # [-1, 1] hidden chemistry signal
    latent_risk_tolerance: float | None = None                    # [0, 1] hidden risk appetite
    latent_opportunity_bias: float | None = None                   # [-1, 1] hidden opportunity cost bias


# ---------------------------------------------------------------------------
# Persona structures
# ---------------------------------------------------------------------------


@dataclass
class PersonaSpec:
    """Lightweight persona definition used during generation."""
    persona_id: str
    persona_name: str
    persona_role: str          # e.g. "skill_match_evaluator"
    focus_dimension: str       # e.g. "skill coverage of the task"
    short_instruction: str     # what this persona evaluates


# ---------------------------------------------------------------------------
# Persona opinion (output of judge)
# ---------------------------------------------------------------------------


@dataclass
class PersonaOpinion:
    persona_id: str
    persona_role: str
    utility_score: float       # continuous, positive ≈ accept, negative ≈ reject
    action_probs: dict[str, float] = field(default_factory=dict)  # {accept, skip, reject}
    confidence: float = 0.5    # [0, 1]
    rationale: str = ""
    extracted_concerns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deliberation
# ---------------------------------------------------------------------------


@dataclass
class RoundUpdate:
    round_idx: int
    speaker_persona_id: str
    summary_text: str
    updated_opinions: dict[str, PersonaOpinion] = field(default_factory=dict)


@dataclass
class DeliberationTrace:
    initial_opinions: dict[str, PersonaOpinion] = field(default_factory=dict)
    per_round_updates: list[RoundUpdate] = field(default_factory=list)
    final_opinions: dict[str, PersonaOpinion] = field(default_factory=dict)
    stop_reason: str = ""


# ---------------------------------------------------------------------------
# Importance ranking
# ---------------------------------------------------------------------------


@dataclass
class PersonaImportance:
    persona_id: str
    raw_score: float
    normalized_weight: float   # sum to 1.0 across personas


# ---------------------------------------------------------------------------
# Decision results
# ---------------------------------------------------------------------------


@dataclass
class DecisionResult:
    side: SideType
    utility: float
    action: Action
    action_probs: dict[str, float]
    confidence: float
    selected_personas: list[str]
    all_persona_opinions: dict[str, PersonaOpinion]
    importance_scores: dict[str, float]
    final_rationale: str


@dataclass
class BilateralDecisionResult:
    requester_decision: DecisionResult
    candidate_decision: DecisionResult
    joint_action: JointAction
    joint_accept_prob: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Outcome simulation
# ---------------------------------------------------------------------------


@dataclass
class OutcomeResult:
    agreement_probability: float
    expected_rounds: float
    completion_probability: float
    requester_satisfaction: float
    candidate_satisfaction: float
    outcome_rationale: str = ""


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


@dataclass
class RewardResult:
    feedback_reward: float
    efficiency_reward: float
    quality_reward: float
    total_reward: float
    breakdown: dict[str, float] = field(default_factory=dict)
