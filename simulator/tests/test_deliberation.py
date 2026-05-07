"""
simulator/tests/test_deliberation.py
====================================
Tests for the deliberation engine.
"""
from __future__ import annotations

import pytest

from simulator.config import SimulatorConfig
from simulator.deliberation import DeliberationEngine
from simulator.mock_backend import RuleBasedBackend
from simulator.persona_generator import RequesterPersonaGenerator
from simulator.types import (
    CandidateCard,
    MatchingContext,
    SideType,
    TaskSpec,
    UserProfile,
)


@pytest.fixture
def config() -> SimulatorConfig:
    cfg = SimulatorConfig(backend_type="mock", random_seed=123)
    cfg.apply_seed()
    return cfg


@pytest.fixture
def backend(config: SimulatorConfig) -> RuleBasedBackend:
    return RuleBasedBackend(config)


@pytest.fixture
def context() -> MatchingContext:
    return MatchingContext(
        requester=UserProfile(
            user_id="req_d",
            role="pm",
            capabilities={"python": 0.8},
            needs={"ml": 0.9},
        ),
        candidate=UserProfile(
            user_id="cand_d",
            role="engineer",
            capabilities={"python": 0.9, "ml_systems": 0.7},
            preferences={"interests": "ml_systems python"},
        ),
        task=TaskSpec(
            task_id="task_d",
            title="ML Pipeline",
            description="Python ML pipeline",
            required_skills={"python": 0.7},
            metadata={"urgency": "medium"},
        ),
        card=CandidateCard(
            candidate_id="cand_d",
            summary="ML engineer",
            highlighted_strengths=["Python"],
            highlighted_risks=[],
            explanation="Fit",
        ),
    )


class TestDeliberationEngine:
    def test_deliberation_returns_valid_trace(
        self, backend, config, context
    ):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()

        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        assert trace.stop_reason != ""
        assert isinstance(trace.initial_opinions, dict)
        assert isinstance(trace.final_opinions, dict)
        assert len(trace.initial_opinions) == len(personas)

    def test_deliberation_stops_at_max_rounds(
        self, backend, context
    ):
        config = SimulatorConfig(
            backend_type="mock",
            random_seed=42,
            max_deliberation_rounds=2,
            early_stop_threshold=0.99,
        )
        config.apply_seed()
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()

        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        assert "max_rounds_reached" in trace.stop_reason
        assert len(trace.per_round_updates) <= config.max_deliberation_rounds

    def test_deliberation_early_stop_on_majority(
        self, backend, context
    ):
        config = SimulatorConfig(
            backend_type="mock",
            random_seed=42,
            max_deliberation_rounds=5,
            early_stop_threshold=0.90,  # Very high threshold — may not trigger
        )
        config.apply_seed()
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()

        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        # Should either stop early or hit max rounds
        if "early_stop" in trace.stop_reason:
            assert len(trace.per_round_updates) < config.max_deliberation_rounds
        else:
            assert "max_rounds_reached" in trace.stop_reason

    def test_initial_and_final_opinions_both_populated(
        self, backend, config, context
    ):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()

        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        assert set(trace.initial_opinions.keys()) == set(trace.final_opinions.keys())
        for pid, op in trace.final_opinions.items():
            assert -1.0 <= op.utility_score <= 1.0
            assert 0.0 <= op.confidence <= 1.0

    def test_round_updates_sequential_round_numbers(
        self, backend, config, context
    ):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()

        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        for i, update in enumerate(trace.per_round_updates):
            assert update.round_idx == i + 1  # Rounds start at 1

    def test_deliberation_with_2_personas(
        self, backend, context
    ):
        """Edge case: deliberation with minimum 2 personas."""
        config = SimulatorConfig(backend_type="mock", random_seed=99)
        config.apply_seed()
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)[:2]
        engine = DeliberationEngine()

        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        assert len(trace.initial_opinions) == 2
        assert len(trace.final_opinions) == 2
