"""
simulator/tests/test_bilateral.py
=================================
Tests for the bilateral simulator and decision model.
"""
from __future__ import annotations

import pytest

from simulator.bilateral_simulator import BilateralSimulator
from simulator.config import SimulatorConfig
from simulator.decision_model import SideDecisionEngine, action_from_probs
from simulator.mock_backend import RuleBasedBackend
from simulator.types import (
    Action,
    CandidateCard,
    JointAction,
    MatchingContext,
    PersonaImportance,
    PersonaOpinion,
    PersonaSpec,
    SideType,
    TaskSpec,
    UserProfile,
)
from simulator.utils import TraceLogger


class TestSideDecision:
    def test_action_probs_sum_to_one(self):
        """Action probability distributions should sum to ~1.0."""
        probs = {"accept": 0.4, "skip": 0.3, "reject": 0.3}
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_action_from_probs_accept(self):
        probs = {"accept": 0.7, "skip": 0.2, "reject": 0.1}
        assert action_from_probs(probs) == Action.ACCEPT

    def test_action_from_probs_skip(self):
        probs = {"accept": 0.3, "skip": 0.5, "reject": 0.2}
        assert action_from_probs(probs) == Action.SKIP

    def test_action_from_probs_reject(self):
        probs = {"accept": 0.1, "skip": 0.2, "reject": 0.7}
        assert action_from_probs(probs) == Action.REJECT

    def test_probabilistic_mode_uses_action_probs(self, context):
        """
        Verify probabilistic mode actually samples from the action_probs distribution
        rather than falling back to deterministic threshold logic.

        We construct a context where the utility is in the SKIP band (near 0)
        but the action_probs strongly favor ACCEPT (0.99 accept, 0.01 skip, 0.0 reject).
        In deterministic mode the action would be SKIP (utility in [-0.3, 0.3]).
        In probabilistic mode the action should be ACCEPT (sampled from probs).
        """
        from simulator.decision_model import SideDecisionEngine
        from simulator.deliberation import DeliberationEngine
        from simulator.importance_ranker import rank_persona_importance, select_top_k
        from simulator.persona_generator import RequesterPersonaGenerator
        from simulator.types import DeliberationTrace, PersonaOpinion

        cfg_prob = SimulatorConfig(
            backend_type="mock", random_seed=42,
            decision_mode="probabilistic",
        )
        cfg_prob.apply_seed()
        bk = RuleBasedBackend(cfg_prob)

        # Run deliberation normally
        gen = RequesterPersonaGenerator(bk, cfg_prob)
        personas = gen.generate(context)
        engine = DeliberationEngine()
        trace = engine.run(personas, context, bk, cfg_prob, SideType.REQUESTER)
        importance = rank_persona_importance(trace.final_opinions, context, trace, cfg_prob)
        selected, _ = select_top_k(personas, importance, cfg_prob)

        # Inject a persona opinion with SKIP-band utility but ACCEPT-biased probs
        dominant_pid = selected[0].persona_id
        mixed_opinions = dict(trace.final_opinions)
        mixed_opinions[dominant_pid] = PersonaOpinion(
            persona_id=dominant_pid,
            persona_role="skill_match_evaluator",
            utility_score=0.05,  # in SKIP band: [-0.3, 0.3]
            action_probs={"accept": 0.99, "skip": 0.01, "reject": 0.0},
            confidence=0.5,
            rationale="",
            extracted_concerns=[],
        )

        dec = SideDecisionEngine()
        decision = dec.decide(
            personas=personas,
            final_opinions=mixed_opinions,
            importance_scores=importance,
            selected_personas=selected,
            context=context,
            config=cfg_prob,
            side=SideType.REQUESTER,
            trace=trace,
        )

        # With near-0.99 accept prob, probabilistic mode should return ACCEPT
        # (Deterministic threshold would return SKIP since utility ≈ 0.05 < 0.3)
        assert decision.action == Action.ACCEPT, (
            f"probabilistic mode must sample from action_probs; "
            f"got {decision.action} (utility={decision.utility:.3f}, "
            f"probs={mixed_opinions[dominant_pid].action_probs})"
        )

    def test_decision_result_has_all_fields(self, backend, config, context):
        from simulator.deliberation import DeliberationEngine
        from simulator.importance_ranker import rank_persona_importance, select_top_k
        from simulator.persona_generator import RequesterPersonaGenerator

        from simulator.types import DeliberationTrace

        gen = RequesterPersonaGenerator(backend, config)
        personas = gen.generate(context)
        engine = DeliberationEngine()
        trace = engine.run(personas, context, backend, config, SideType.REQUESTER)
        importance = rank_persona_importance(
            trace.final_opinions, context, trace, config
        )
        selected, _ = select_top_k(personas, importance, config)

        dec_engine = SideDecisionEngine()
        decision = dec_engine.decide(
            personas=personas,
            final_opinions=trace.final_opinions,
            importance_scores=importance,
            selected_personas=selected,
            context=context,
            config=config,
            side=SideType.REQUESTER,
            trace=trace,
        )

        assert decision.side == SideType.REQUESTER
        assert -1.0 <= decision.utility <= 1.0
        assert decision.action in {Action.ACCEPT, Action.SKIP, Action.REJECT}
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.selected_personas) <= config.top_k_personas
        assert isinstance(decision.final_rationale, str)


class TestBilateralSimulator:
    def test_bilateral_simulator_runs(self, backend, config, context):
        logger = TraceLogger(verbose=False)
        sim = BilateralSimulator(backend, config, logger)
        result = sim.run(context)

        assert result.requester_decision is not None
        assert result.candidate_decision is not None
        assert result.joint_action in list(JointAction)

    def test_joint_accept_prob_is_valid(self, backend, config, context):
        sim = BilateralSimulator(backend, config, TraceLogger(verbose=False))
        result = sim.run(context)
        assert 0.0 <= result.joint_accept_prob <= 1.0

    def test_requester_and_candidate_decisions_independent(
        self, backend, config, context
    ):
        sim = BilateralSimulator(backend, config, TraceLogger(verbose=False))
        result = sim.run(context)
        # Both should be valid decisions, possibly different
        assert result.requester_decision.side == SideType.REQUESTER
        assert result.candidate_decision.side == SideType.CANDIDATE

    def test_logger_collects_phases(self, backend, config, context):
        logger = TraceLogger(verbose=False)
        sim = BilateralSimulator(backend, config, logger)
        sim.run(context)

        phases = {e["phase"] for e in logger.entries}
        expected = {
            "persona_generation",
            "requester_deliberation",
            "candidate_deliberation",
            "importance_ranking",
            "side_decisions",
            "bilateral_result",
        }
        assert expected.issubset(phases), f"Missing phases: {expected - phases}"

    def test_joint_action_mapping(self, backend, config):
        """Test that joint actions are assigned correctly."""
        from simulator.decision_model import SideDecisionEngine
        from simulator.types import DecisionResult

        req = DecisionResult(
            side=SideType.REQUESTER,
            utility=0.5,
            action=Action.ACCEPT,
            action_probs={"accept": 0.7, "skip": 0.2, "reject": 0.1},
            confidence=0.8,
            selected_personas=[],
            all_persona_opinions={},
            importance_scores={},
            final_rationale="",
        )
        cand = DecisionResult(
            side=SideType.CANDIDATE,
            utility=0.6,
            action=Action.ACCEPT,
            action_probs={"accept": 0.8, "skip": 0.1, "reject": 0.1},
            confidence=0.7,
            selected_personas=[],
            all_persona_opinions={},
            importance_scores={},
            final_rationale="",
        )
        sim = BilateralSimulator(backend, config, TraceLogger(verbose=False))
        joint, prob = sim._compute_joint_action(req, cand)
        assert joint == JointAction.MUTUAL_ACCEPT
        assert abs(prob - 0.7 * 0.8) < 1e-6
