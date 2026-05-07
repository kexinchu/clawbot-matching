"""Dummy data factories for testing the learning loop.

In production:
  - dummy_create_profile() → replaced by L1 LLM extraction (UserState from dialogue)
  - dummy_create_task()    → replaced by structured task intake form
  - dummy_user_feedback()  → replaced by L4 real feedback collection
"""

import sys
import os
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mapping-algo"); sys.path.append(_p) if _p not in sys.path else None  # os.path.join(os.path.dirname(__file__), '..', 'mapping-algo'))

from datatypes import (
    UserState, CapabilityEntry, NeedEntry,
    Task, TaskRequirement, TaskOffer, MatchResult,
)
from encoder import SimpleEncoder
import numpy as np
from typing import Optional

_enc = SimpleEncoder(dim=64)


def find_cap(user: UserState, description: str) -> Optional[CapabilityEntry]:
    """Look up a CapabilityEntry by description (exact match). Returns None if not found."""
    for cap in user.capabilities:
        if cap.description == description:
            return cap
    return None


def dummy_create_profile(user_id: str, role: str = "") -> UserState:
    """
    In production: LLM extracts (Cap, Need) from dialogue → UserState.
    Here: generate synthetic profiles with SimpleEncoder embeddings for testing.
    """
    profiles = {
        "alice": UserState(
            user_id="alice",
            capabilities=[
                CapabilityEntry(_enc("bayesian statistics"),  mu=0.3, sigma=0.2, source="explicit",  description="bayesian"),
                CapabilityEntry(_enc("python programming"),   mu=0.8, sigma=0.1, source="explicit",  description="python"),
                CapabilityEntry(_enc("academic paper writing"), mu=0.4, sigma=0.3, source="implicit", description="paper_writing"),
            ],
            needs=[
                NeedEntry(_enc("bayesian statistics"),  intensity=0.8, description="bayesian"),
                NeedEntry(_enc("python programming"),   intensity=0.1, description="python"),
                NeedEntry(_enc("academic paper writing"), intensity=0.7, description="paper_writing"),
            ],
        ),
        "bob": UserState(
            user_id="bob",
            capabilities=[
                CapabilityEntry(_enc("bayesian statistics"),  mu=0.9, sigma=0.1, source="explicit",  description="bayesian"),
                CapabilityEntry(_enc("python programming"),   mu=0.5, sigma=0.2, source="explicit",  description="python"),
                CapabilityEntry(_enc("academic paper writing"), mu=0.8, sigma=0.1, source="explicit", description="paper_writing"),
            ],
            needs=[
                NeedEntry(_enc("bayesian statistics"),  intensity=0.2, description="bayesian"),
                NeedEntry(_enc("python programming"),   intensity=0.7, description="python"),
                NeedEntry(_enc("academic paper writing"), intensity=0.9, description="paper_writing"),
            ],
        ),
        "carol": UserState(
            user_id="carol",
            capabilities=[
                CapabilityEntry(_enc("bayesian statistics"),  mu=0.6, sigma=0.4, source="meta", description="bayesian"),
                CapabilityEntry(_enc("python programming"),   mu=0.7, sigma=0.35, source="meta", description="python"),
                CapabilityEntry(_enc("academic paper writing"), mu=0.5, sigma=0.4, source="meta", description="paper_writing"),
            ],
            needs=[
                NeedEntry(_enc("bayesian statistics"),  intensity=0.5, description="bayesian"),
                NeedEntry(_enc("python programming"),   intensity=0.3, description="python"),
                NeedEntry(_enc("academic paper writing"), intensity=0.8, description="paper_writing"),
            ],
        ),
    }
    return profiles.get(user_id)


def dummy_create_task() -> Task:
    """Synthetic task for testing. All requirements are soft (attention matching)."""
    return Task(
        task_id="task_001",
        goal="Build Bayesian churn model, target NeurIPS",
        requirements=[
            TaskRequirement(_enc("bayesian statistics"),    level=0.8, constraint_type="soft", description="bayesian"),
            TaskRequirement(_enc("python programming"),     level=0.6, constraint_type="soft", description="python"),
            TaskRequirement(_enc("academic paper writing"), level=0.7, constraint_type="soft", description="paper_writing"),
        ],
        offers=[
            TaskOffer(_enc("research collaboration"),  strength=0.8, source="explicit",  description="research collaboration"),
            TaskOffer(_enc("academic authorship"),     strength=0.7, source="explicit",  description="co-authorship"),
            TaskOffer(_enc("python programming"),      strength=0.5, source="inferred",  description="python practice"),
        ],
    )


def dummy_user_feedback(match_result: MatchResult) -> dict:
    """
    Simulate user actions and collaboration outcome.
    In production: collected from real user interactions (Layer 4).

    Returns raw data that RewardFunction will process.
    """
    noise = np.random.normal(0, 0.05)
    acceptance_prob = np.clip(match_result.match_score + noise, 0, 1)

    r_u = 1.0 if acceptance_prob > 0.6 else (0.7 if acceptance_prob > 0.4 else 0.3)
    r_v = 1.0 if acceptance_prob > 0.5 else (0.7 if acceptance_prob > 0.3 else 0.3)

    n_rounds = max(3, int(3 + (1 - match_result.match_score) * 25 + np.random.randint(-2, 3)))

    completion_prob = match_result.match_score * 0.9 + 0.1
    if np.random.random() < completion_prob:
        f_completion = 1.0
    elif np.random.random() < 0.5:
        f_completion = 0.5
    else:
        f_completion = 0.0

    stars_u = np.clip(int(match_result.match_score * 5 + np.random.normal(0, 0.5) + 0.5), 1, 5)
    stars_v = np.clip(int(match_result.match_score * 5 + np.random.normal(0, 0.5) + 0.5), 1, 5)

    return {
        "r_u": r_u,
        "r_v": r_v,
        "n_rounds": n_rounds,
        "f_completion": f_completion,
        "stars_u": stars_u,
        "stars_v": stars_v,
    }
