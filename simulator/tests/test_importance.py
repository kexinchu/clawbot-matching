"""
simulator/tests/test_importance.py
==================================
Tests for persona importance ranking and top-k pruning.
"""
from __future__ import annotations

import pytest

from simulator.config import SimulatorConfig
from simulator.deliberation import DeliberationEngine
from simulator.importance_ranker import (
    _compute_decision_flip_impact,
    rank_persona_importance,
    select_top_k,
)
from simulator.mock_backend import RuleBasedBackend
from simulator.persona_generator import RequesterPersonaGenerator
from simulator.persona_judge import judge_persona_opinion
from simulator.types import (
    CandidateCard,
    DeliberationTrace,
    MatchingContext,
    PersonaImportance,
    PersonaOpinion,
    SideType,
    TaskSpec,
    UserProfile,
)


@pytest.fixture
def config() -> SimulatorConfig:
    cfg = SimulatorConfig(backend_type="mock", random_seed=77)
    cfg.apply_seed()
    return cfg


@pytest.fixture
def backend(config: SimulatorConfig) -> RuleBasedBackend:
    return RuleBasedBackend(config)


@pytest.fixture
def context() -> MatchingContext:
    return MatchingContext(
        requester=UserProfile(
            user_id="req_imp",
            role="pm",
            capabilities={"python": 0.8},
            needs={"ml": 0.9},
        ),
        candidate=UserProfile(
            user_id="cand_imp",
            role="engineer",
            capabilities={"python": 0.9, "ml_systems": 0.7},
        ),
        task=TaskSpec(
            task_id="task_imp",
            title="ML Pipeline",
            description="Python ML pipeline",
            required_skills={"python": 0.7, "ml_systems": 0.6},
            metadata={"urgency": "high"},
        ),
        card=CandidateCard(
            candidate_id="cand_imp",
            summary="ML engineer",
            highlighted_strengths=["Python"],
            highlighted_risks=["availability"],
            explanation="Fit",
        ),
    )


class TestImportanceRanking:
    def test_weights_sum_to_one(self, backend, config, context):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()
        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        importance = rank_persona_importance(
            trace.final_opinions, context, trace, config
        )

        total = sum(imp.normalized_weight for imp in importance)
        assert abs(total - 1.0) < 1e-5, f"Weights sum to {total}, expected 1.0"

    def test_all_personas_get_weight(self, backend, config, context):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()
        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        importance = rank_persona_importance(
            trace.final_opinions, context, trace, config
        )

        opinion_ids = set(trace.final_opinions.keys())
        ranked_ids = set(imp.persona_id for imp in importance)
        assert opinion_ids == ranked_ids, "Every persona should get a weight"

    def test_weights_are_sorted_descending(
        self, backend, config, context
    ):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()
        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        importance = rank_persona_importance(
            trace.final_opinions, context, trace, config
        )

        weights = [imp.normalized_weight for imp in importance]
        assert weights == sorted(weights, reverse=True), "Weights should be descending"

    def test_top_k_selects_k_personas(self, backend, config, context):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()
        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        importance = rank_persona_importance(
            trace.final_opinions, context, trace, config
        )

        test_cfg = SimulatorConfig(backend_type="mock", random_seed=42)
        test_cfg.top_k_personas = 2
        selected, pruned = select_top_k(personas, importance, test_cfg)

        assert len(selected) == 2
        assert len(pruned) == len(personas) - 2
        assert set(p.persona_id for p in selected + pruned) == set(p.persona_id for p in personas)

    def test_selected_sorted_by_weight(self, backend, config, context):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()
        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        importance = rank_persona_importance(
            trace.final_opinions, context, trace, config
        )

        config.top_k_personas = len(personas)
        selected, _ = select_top_k(personas, importance, config)
        imp_map = {imp.persona_id: imp for imp in importance}

        selected_weights = [imp_map[p.persona_id].normalized_weight for p in selected]
        assert selected_weights == sorted(selected_weights, reverse=True)

    def test_empty_opinions_returns_empty(self, config, context):
        importance = rank_persona_importance({}, context, DeliberationTrace(), config)
        assert importance == []

    def test_raw_scores_are_nonnegative(self, backend, config, context):
        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()
        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)

        importance = rank_persona_importance(
            trace.final_opinions, context, trace, config
        )

        for imp in importance:
            assert imp.raw_score >= 0, f"Raw score for {imp.persona_id} is negative"
            assert 0 <= imp.normalized_weight <= 1

    def test_decision_flip_impact(self, config):
        """
        Verify _compute_decision_flip_impact produces reasonable non-negative values.

        Cases:
        - Single persona: no flip possible → 0.0
        - All equal opinions: removing one changes nothing → 0.0
        - Dominant persona with high weight: removing it shifts utility
          enough to cross the accept threshold → > 0
        """
        def make_opinion(pid: str, utility: float) -> PersonaOpinion:
            return PersonaOpinion(
                persona_id=pid,
                persona_role="skill_match_evaluator",
                utility_score=utility,
                action_probs={"accept": 0.5, "skip": 0.3, "reject": 0.2},
                confidence=0.8,
                rationale="",
                extracted_concerns=[],
            )

        # Case 1: single persona → 0.0
        single = {"p1": make_opinion("p1", 0.5)}
        single_weights = {"p1": 1.0}
        flip = _compute_decision_flip_impact("p1", single["p1"], single, single_weights, config)
        assert flip == 0.0, "Single persona cannot flip a decision"

        # Case 2: identical utilities → removing one changes nothing
        identical = {
            "a": make_opinion("a", 0.5),
            "b": make_opinion("b", 0.5),
            "c": make_opinion("c", 0.5),
        }
        ident_weights = {pid: 1.0 for pid in identical}
        for pid in identical:
            f = _compute_decision_flip_impact(pid, identical[pid], identical, ident_weights, config)
            assert f == 0.0, f"Identical opinions should give zero flip impact for {pid}"

        # Case 3: dominant persona pushes utility into ACCEPT band
        # accept_th = 0.3, so 0.5 is in ACCEPT
        # After removing dominant (utility 0.5), remaining mean ≈ -0.05 (in SKIP band)
        dominant = {
            "dom": make_opinion("dom", 0.5),
            "low": make_opinion("low", -0.1),
        }
        # dom has higher weight (2x), so full_util is closer to 0.5
        dom_weights = {"dom": 2.0, "low": 1.0}
        flip_dom = _compute_decision_flip_impact(
            "dom", dominant["dom"], dominant, dom_weights, config
        )
        assert flip_dom > 0.0, (
            f"Removing dominant persona should have non-zero flip impact, got {flip_dom}"
        )
        assert flip_dom <= 1.0, f"Flip impact should be capped at 1.0, got {flip_dom}"

        # Verify all non-zero flip impacts are in (0, 1]
        all_opinions = {**dominant, **identical}
        all_weights = {**dom_weights, **ident_weights}
        for pid, op in all_opinions.items():
            f = _compute_decision_flip_impact(pid, op, all_opinions, all_weights, config)
            assert f >= 0.0, f"Flip impact must be non-negative for {pid}, got {f}"
