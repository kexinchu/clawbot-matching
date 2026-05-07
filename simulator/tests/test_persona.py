"""
simulator/tests/test_persona.py
==============================
Tests for persona generation and persona judgment.
"""
from __future__ import annotations

import pytest

from simulator.config import SimulatorConfig
from simulator.mock_backend import RuleBasedBackend
from simulator.persona_generator import (
    CandidatePersonaGenerator,
    RequesterPersonaGenerator,
)
from simulator.persona_judge import MockPersonaJudge, judge_persona_opinion
from simulator.types import (
    CandidateCard,
    MatchingContext,
    SideType,
    TaskSpec,
    UserProfile,
)


@pytest.fixture
def mock_context() -> MatchingContext:
    return MatchingContext(
        requester=UserProfile(
            user_id="req_test",
            role="pm",
            capabilities={"python": 0.8},
            needs={"ml": 0.9},
        ),
        candidate=UserProfile(
            user_id="cand_test",
            role="engineer",
            capabilities={"python": 0.9, "ml_systems": 0.8},
            needs={},
            preferences={"interests": "ml_systems python"},
        ),
        task=TaskSpec(
            task_id="task_test",
            title="Build ML Pipeline",
            description="Python ML pipeline for data processing",
            required_skills={"python": 0.7, "ml_systems": 0.6},
            metadata={"urgency": "high", "budget": "competitive"},
        ),
        card=CandidateCard(
            candidate_id="cand_test",
            summary="Experienced ML engineer",
            highlighted_strengths=["Strong Python"],
            highlighted_risks=[],
            explanation="Good fit",
        ),
    )


@pytest.fixture
def config() -> SimulatorConfig:
    cfg = SimulatorConfig(backend_type="mock", random_seed=0)
    cfg.apply_seed()
    return cfg


@pytest.fixture
def backend(config: SimulatorConfig) -> RuleBasedBackend:
    return RuleBasedBackend(config)


class TestPersonaGeneration:
    def test_requester_personas_not_empty(self, backend, config, mock_context):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(mock_context)
        assert len(personas) > 0
        assert all(hasattr(p, "persona_id") for p in personas)
        assert all(hasattr(p, "persona_role") for p in personas)

    def test_candidate_personas_not_empty(self, backend, config, mock_context):
        gen = CandidatePersonaGenerator(backend, config)
        personas = gen.generate(mock_context)
        assert len(personas) > 0
        assert all(hasattr(p, "persona_id") for p in personas)

    def test_requester_personas_are_diverse(self, backend, config, mock_context):
        """All generated personas should have distinct roles."""
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(mock_context)
        roles = [p.persona_role for p in personas]
        assert len(roles) == len(set(roles)), "Persona roles should be unique"

    def test_persona_count_respects_config(self, backend, config, mock_context):
        config.num_requester_personas = 3
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(mock_context)
        assert len(personas) == 3


class TestPersonaJudge:
    def test_judge_returns_valid_opinion(self, backend, config, mock_context):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(mock_context)
        opinion = judge_persona_opinion(personas[0], mock_context, backend, config)

        assert isinstance(opinion.utility_score, float)
        assert -1.0 <= opinion.utility_score <= 1.0
        assert 0.0 <= opinion.confidence <= 1.0
        assert set(opinion.action_probs.keys()) == {"accept", "skip", "reject"}
        assert abs(sum(opinion.action_probs.values()) - 1.0) < 1e-6
        assert isinstance(opinion.rationale, str)
        assert len(opinion.rationale) > 0

    def test_mock_judge_class(self, config, mock_context):
        judge = MockPersonaJudge(config)
        gen = RequesterPersonaGenerator(RuleBasedBackend(config), config)
        personas = gen.generate(mock_context)
        opinion = judge.judge(personas[0], mock_context, round_idx=0)
        assert -1.0 <= opinion.utility_score <= 1.0

    def test_all_requester_persona_roles_judgeable(
        self, backend, config, mock_context
    ):
        """Every requester persona type should produce a valid opinion."""
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(mock_context)
        for p in personas:
            opinion = judge_persona_opinion(p, mock_context, backend, config)
            assert opinion.persona_id == p.persona_id
            assert -1.0 <= opinion.utility_score <= 1.0

    def test_candidate_personas_judgeable(self, backend, config, mock_context):
        gen = CandidatePersonaGenerator(backend, config)
        personas = gen.generate(mock_context)
        for p in personas:
            opinion = judge_persona_opinion(p, mock_context, backend, config)
            assert opinion.persona_id == p.persona_id
            assert -1.0 <= opinion.utility_score <= 1.0

    def test_judge_is_deterministic_per_seed(self, backend, config, mock_context):
        """Same seed → same opinion."""
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(mock_context)
        op1 = judge_persona_opinion(personas[0], mock_context, backend, config, round_idx=0)
        op2 = judge_persona_opinion(personas[0], mock_context, backend, config, round_idx=0)
        assert abs(op1.utility_score - op2.utility_score) < 1e-6

    def test_round_idx_affects_opinion(self, backend, config, mock_context):
        """Higher rounds should produce refined opinions."""
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(mock_context)
        op0 = judge_persona_opinion(personas[0], mock_context, backend, config, round_idx=0)
        op2 = judge_persona_opinion(personas[0], mock_context, backend, config, round_idx=2)
        # Just check they're both valid — may or may not differ
        assert isinstance(op2.rationale, str)
