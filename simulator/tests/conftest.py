"""
simulator/tests/conftest.py
===========================
Shared pytest fixtures for the simulator test suite.
"""
from __future__ import annotations

import pytest

from simulator.config import SimulatorConfig
from simulator.mock_backend import RuleBasedBackend
from simulator.types import (
    CandidateCard,
    MatchingContext,
    TaskSpec,
    UserProfile,
)


@pytest.fixture
def config() -> SimulatorConfig:
    cfg = SimulatorConfig(
        backend_type="mock",
        random_seed=999,
        trace_verbose=False,
        num_requester_personas=4,
        num_candidate_personas=4,
    )
    cfg.apply_seed()
    return cfg


@pytest.fixture
def backend(config: SimulatorConfig) -> RuleBasedBackend:
    return RuleBasedBackend(config)


@pytest.fixture
def context() -> MatchingContext:
    return MatchingContext(
        requester=UserProfile(
            user_id="req_bi",
            role="pm",
            capabilities={"python": 0.8, "ml_systems": 0.7},
            needs={"execution": 0.9},
        ),
        candidate=UserProfile(
            user_id="cand_bi",
            role="engineer",
            capabilities={"python": 0.95, "ml_systems": 0.85},
            needs={},
            preferences={"interests": "ml_systems python"},
        ),
        task=TaskSpec(
            task_id="task_bi",
            title="Build ML System",
            description="Python ML system development",
            required_skills={"python": 0.8, "ml_systems": 0.7},
            metadata={"urgency": "high", "budget": "competitive"},
        ),
        card=CandidateCard(
            candidate_id="cand_bi",
            summary="Senior ML engineer",
            highlighted_strengths=["Python", "ML systems"],
            highlighted_risks=["busy schedule"],
            explanation="Strong technical fit",
        ),
    )
