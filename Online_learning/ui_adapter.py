"""Bridge between the Showcase frontend and the OnlineLearning backend.

This module is the single source of truth for:
  - mapping a UI action (accept / skip / reject) + star rating → feedback dict
    that RewardFunction.compute can consume.
  - defining the 3 demo scenarios that the showcase HTML displays, including
    both backend representations (UserState / Task) and frontend metadata
    (name, role, avatar, initials, strengths, risks).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [_HERE, os.path.join(_HERE, "..", "mapping-algo"), os.path.join(_HERE, "..")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

ENC = SimpleEncoder(dim=64)
CFG = MatchConfig(embedding_dim=64)


# ── UI action → 6-field feedback dict ─────────────────────────────────

# Acceptance signals consistent with simulator.feedback_provider._ACTION_TO_R
_R_BY_ACTION: Dict[str, float] = {
    "accepted": 1.0,
    "skipped": 0.5,
    "rejected": 0.3,
}

# Default star rating to assume when the user just clicks Accept/Skip/Reject
# without leaving a star rating.
_DEFAULT_STARS_BY_ACTION: Dict[str, int] = {
    "accepted": 5,
    "skipped": 3,
    "rejected": 2,
}


def action_to_feedback(
    action: str,
    stars: Optional[int] = None,
    *,
    candidate_stars: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert a UI action + (optional) star rating to the 6-field dict that
    Reward_function.RewardFunction.compute expects.

    Parameters
    ----------
    action : "accepted" | "skipped" | "rejected"
    stars  : 1..5 star rating from the requester (the user clicking the UI).
             If None, a default is chosen based on the action.
    candidate_stars : optional candidate-side rating; if not provided the
             requester's rating is mirrored (the showcase only collects one
             rating from the human in the loop).
    """
    if action not in _R_BY_ACTION:
        raise ValueError(f"Unknown action '{action}'. Expected accepted/skipped/rejected.")

    r_u = _R_BY_ACTION[action]
    r_v = r_u

    if stars is None:
        stars = _DEFAULT_STARS_BY_ACTION[action]
    stars = max(1, min(5, int(stars)))

    if candidate_stars is None:
        candidate_stars = stars
    candidate_stars = max(1, min(5, int(candidate_stars)))

    # Action-conditioned outcome signals (n_rounds, f_completion) that line up
    # with the qualitative meaning of accept/skip/reject. These are not
    # directly visible to the user; they're a deterministic derivation so the
    # reward formula stays a pure function of the UI inputs.
    if action == "accepted":
        f_completion = 1.0 if stars >= 4 else 0.5
        n_rounds = max(3, 9 - stars)             # higher rating → fewer rounds
    elif action == "skipped":
        f_completion = 0.5
        n_rounds = 15
    else:  # rejected
        f_completion = 0.0
        n_rounds = 22

    return {
        "r_u": r_u,
        "r_v": r_v,
        "n_rounds": int(n_rounds),
        "f_completion": float(f_completion),
        "stars_u": int(stars),
        "stars_v": int(candidate_stars),
    }


# ── Demo scenarios shown on the showcase HTML ──────────────────────────

@dataclass
class CandidateSpec:
    """Backend candidate + frontend display metadata bundled together."""
    user_id: str
    name: str
    initials: str
    role: str
    avatar: str
    capabilities: Dict[str, float]   # skill → mu
    sigma_init: float = 0.15
    needs: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioSpec:
    scenario_id: str
    label: str
    requester_id: str
    requester_name: str
    requester_initials: str
    requester_role: str
    requester_task: str
    accent: str
    task_id: str
    task_goal: str
    requirements: Dict[str, float]   # skill → required level
    offers: Dict[str, float]          # skill → offer strength (for S_need)
    candidates: List[CandidateSpec]


def _user_state(user_id: str, capabilities: Dict[str, float],
                needs: Dict[str, float], sigma: float = 0.15) -> UserState:
    return UserState(
        user_id=user_id,
        capabilities=[
            CapabilityEntry(ENC(k), mu=float(v), sigma=sigma,
                            source="explicit", description=k)
            for k, v in capabilities.items()
        ],
        needs=[
            NeedEntry(ENC(k), intensity=float(v), description=k)
            for k, v in needs.items()
        ],
        clearance_level=0,
    )


def scenario_to_backend(spec: ScenarioSpec) -> Dict[str, Any]:
    """Materialise the backend representation (requester, task, candidate pool)."""
    requester = _user_state(
        spec.requester_id,
        capabilities={},                  # leave empty so each req has real gap
        needs={"task completion": 0.5},
    )
    task = Task(
        task_id=spec.task_id,
        goal=spec.task_goal,
        requirements=[
            TaskRequirement(ENC(k), level=float(v),
                            constraint_type="soft", description=k)
            for k, v in spec.requirements.items()
        ],
        offers=[
            TaskOffer(ENC(k), strength=float(v),
                      source="explicit", description=k)
            for k, v in spec.offers.items()
        ],
        data_clearance=0,
    )
    candidate_states = [
        _user_state(c.user_id, c.capabilities, c.needs, c.sigma_init)
        for c in spec.candidates
    ]
    return {
        "requester": requester,
        "task": task,
        "candidate_states": candidate_states,
    }


def build_demo_scenarios() -> List[ScenarioSpec]:
    """The three showcase scenarios. Each has 4 candidates so the dreaming
    layer can meaningfully shortlist a top-3.
    """
    return [
        ScenarioSpec(
            scenario_id="research",
            label="Research collab",
            requester_id="alice_cto",
            requester_name="Alice Chen",
            requester_initials="AC",
            requester_role="AI Startup CTO",
            requester_task="Bayesian churn prediction model → NeurIPS",
            accent="#6C5CE7",
            task_id="task_research",
            task_goal="Build a Bayesian churn-prediction model and co-author a NeurIPS submission",
            requirements={
                "bayesian modeling": 0.90,
                "python programming": 0.80,
                "paper writing": 0.85,
            },
            offers={
                "research collaboration": 0.90,
                "publication credit": 0.85,
            },
            candidates=[
                CandidateSpec(
                    user_id="bob_zhang", name="Bob Zhang", initials="BZ",
                    role="Stats PhD · Stanford", avatar="#6C5CE7",
                    capabilities={"bayesian modeling": 0.9, "python programming": 0.8,
                                  "paper writing": 0.82},
                    needs={"research collaboration": 0.9, "publication credit": 0.95},
                    strengths=["Bayesian expert (μ=0.9)", "8 publications", "Same timezone (EST)"],
                    risks=["20h/week limit", "PhD coursework competing"],
                    stats={"collabs": 12, "successRate": "92%", "avgRating": 4.7},
                ),
                CandidateSpec(
                    user_id="carol_liu", name="Carol Liu", initials="CL",
                    role="ML Researcher · Freelance", avatar="#E17055",
                    capabilities={"bayesian modeling": 0.7, "python programming": 0.85,
                                  "paper writing": 0.6},
                    needs={"research collaboration": 0.7, "publication credit": 0.6},
                    strengths=["Creative novel approaches", "Strong Python (μ=0.85)", "Wants to publish"],
                    risks=["15h/week + freelancing", "6h timezone gap (CET)", "Inconsistent follow-through"],
                    stats={"collabs": 3, "successRate": "67%", "avgRating": 3.8},
                ),
                CandidateSpec(
                    user_id="david_park", name="David Park", initials="DP",
                    role="Stats Postdoc · MIT", avatar="#00B894",
                    capabilities={"bayesian modeling": 0.95, "python programming": 0.7,
                                  "paper writing": 0.9},
                    needs={"research collaboration": 0.95, "publication credit": 0.9},
                    strengths=["Top-tier Bayesian (μ=0.95)", "Strong writer (μ=0.9)", "5 NeurIPS papers"],
                    risks=["Limited availability", "Already on 2 projects"],
                    stats={"collabs": 18, "successRate": "94%", "avgRating": 4.8},
                ),
                CandidateSpec(
                    user_id="emily_wong", name="Emily Wong", initials="EW",
                    role="ML Engineer · Industry", avatar="#0984E3",
                    capabilities={"bayesian modeling": 0.5, "python programming": 0.95,
                                  "paper writing": 0.4},
                    needs={"research collaboration": 0.5},
                    strengths=["Production ML systems", "Python wizard (μ=0.95)"],
                    risks=["No publication track record", "Limited theoretical background"],
                    stats={"collabs": 6, "successRate": "83%", "avgRating": 4.2},
                ),
            ],
        ),
        ScenarioSpec(
            scenario_id="hiring",
            label="Talent search",
            requester_id="ben_park",
            requester_name="Ben Park",
            requester_initials="BP",
            requester_role="VP Data · FinTech Series B",
            requester_task="Hire Data Scientist — data mining & feature engineering",
            accent="#00B894",
            task_id="task_hiring",
            task_goal="Hire a data scientist with strong data mining and feature engineering",
            requirements={
                "data mining": 0.85,
                "feature engineering": 0.85,
                "python programming": 0.80,
                "production ml": 0.65,
            },
            offers={
                "stable employment": 0.85,
                "career growth": 0.75,
                "competitive salary": 0.70,
            },
            candidates=[
                CandidateSpec(
                    user_id="jenny_wang", name="Jenny Wang", initials="JW",
                    role="New grad · 3 internships", avatar="#00B894",
                    capabilities={"data mining": 0.7, "feature engineering": 0.65,
                                  "python programming": 0.85, "production ml": 0.4},
                    needs={"career growth": 0.9, "stable employment": 0.8},
                    strengths=["3 internships (Google, Stripe, startup)",
                               "Eager & available full-time", "Salary within budget"],
                    risks=["No production ML experience", "May need mentoring first 3 months"],
                    stats={"collabs": 3, "successRate": "100%", "avgRating": 4.9},
                ),
                CandidateSpec(
                    user_id="john_miller", name="John Miller", initials="JM",
                    role="Senior DS · 8 years exp", avatar="#2D3436",
                    capabilities={"data mining": 0.95, "feature engineering": 0.92,
                                  "python programming": 0.85, "production ml": 0.9},
                    needs={"competitive salary": 0.95, "career growth": 0.5},
                    strengths=["Expert data mining (μ=0.95)", "Production system experience",
                               "Can lead projects day 1"],
                    risks=["Salary 40% above budget", "Wants management track",
                           "May leave in ~1 year"],
                    stats={"collabs": 34, "successRate": "88%", "avgRating": 4.3},
                ),
                CandidateSpec(
                    user_id="raj_patel", name="Raj Patel", initials="RP",
                    role="Mid-level DS · 4 years", avatar="#FD79A8",
                    capabilities={"data mining": 0.82, "feature engineering": 0.85,
                                  "python programming": 0.82, "production ml": 0.72},
                    needs={"career growth": 0.85, "stable employment": 0.75, "competitive salary": 0.7},
                    strengths=["Solid all-around DS", "On budget", "Recent ML platform launch"],
                    risks=["Specializes in NLP, not data mining", "Wants remote-first"],
                    stats={"collabs": 14, "successRate": "85%", "avgRating": 4.4},
                ),
                CandidateSpec(
                    user_id="lisa_martinez", name="Lisa Martinez", initials="LM",
                    role="ML PhD · Recent graduate", avatar="#FDCB6E",
                    capabilities={"data mining": 0.78, "feature engineering": 0.6,
                                  "python programming": 0.78, "production ml": 0.35},
                    needs={"career growth": 0.95, "competitive salary": 0.6},
                    strengths=["PhD thesis on graph mining", "Strong academic credentials"],
                    risks=["Zero industry experience", "May prefer research role"],
                    stats={"collabs": 5, "successRate": "80%", "avgRating": 4.6},
                ),
            ],
        ),
        ScenarioSpec(
            scenario_id="marketplace",
            label="Marketplace",
            requester_id="simon_davis",
            requester_name="Simon Davis",
            requester_initials="SD",
            requester_role="Individual seller",
            requester_task="Sell 2021 Tesla Model 3 — 45k miles, good condition",
            accent="#D63031",
            task_id="task_marketplace",
            task_goal="Sell a 2021 Tesla Model 3 (45k miles) for a fair price with low hassle",
            requirements={
                "buyer reliability": 0.80,
                "financing readiness": 0.75,
                "fair price offer": 0.80,
            },
            offers={
                "quick close": 0.85,
                "fair sale": 0.80,
            },
            candidates=[
                CandidateSpec(
                    user_id="kevin_zhao", name="Kevin Zhao", initials="KZ",
                    role="Buyer · Pre-approved financing", avatar="#D63031",
                    capabilities={"buyer reliability": 0.9, "financing readiness": 0.95,
                                  "fair price offer": 0.85},
                    needs={"quick close": 0.85, "fair sale": 0.7},
                    strengths=["Offer: $28,500 (above market avg)",
                               "Pre-approved loan, ready to close",
                               "Flexible on pickup timing"],
                    risks=["Wants pre-purchase inspection (+3-5 days)"],
                    stats={"collabs": 7, "successRate": "86%", "avgRating": 4.6},
                ),
                CandidateSpec(
                    user_id="benjamin_foster", name="Benjamin Foster", initials="BF",
                    role="Buyer · Cash offer", avatar="#636E72",
                    capabilities={"buyer reliability": 0.55, "financing readiness": 0.95,
                                  "fair price offer": 0.55},
                    needs={"quick close": 0.95},
                    strengths=["Cash buyer, no financing delay", "Can pick up today"],
                    risks=["Offer: $25,200 (below market)",
                           "History of last-minute renegotiation",
                           "Low platform reputation"],
                    stats={"collabs": 2, "successRate": "50%", "avgRating": 3.5},
                ),
                CandidateSpec(
                    user_id="anna_kim", name="Anna Kim", initials="AK",
                    role="Buyer · Loan pending", avatar="#00CEC9",
                    capabilities={"buyer reliability": 0.85, "financing readiness": 0.7,
                                  "fair price offer": 0.82},
                    needs={"quick close": 0.6, "fair sale": 0.85},
                    strengths=["Offer: $27,800 (market price)", "Pre-qualified, awaiting loan approval"],
                    risks=["Loan approval pending (~5 days)"],
                    stats={"collabs": 4, "successRate": "75%", "avgRating": 4.3},
                ),
                CandidateSpec(
                    user_id="mike_brown", name="Mike Brown", initials="MB",
                    role="Buyer · Lease takeover", avatar="#A29BFE",
                    capabilities={"buyer reliability": 0.7, "financing readiness": 0.6,
                                  "fair price offer": 0.65},
                    needs={"quick close": 0.8},
                    strengths=["Repeat platform user", "Same-day decision"],
                    risks=["Offer: $26,000 (below market)", "Wants extended test drive"],
                    stats={"collabs": 11, "successRate": "78%", "avgRating": 4.1},
                ),
            ],
        ),
    ]
