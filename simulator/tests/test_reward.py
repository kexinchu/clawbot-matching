"""
simulator/tests/test_reward.py
==============================
Tests for the reward construction module.
"""
from __future__ import annotations

import pytest

from simulator.config import SimulatorConfig
from simulator.mock_backend import RuleBasedBackend
from simulator.outcome_simulator import OutcomeSimulator
from simulator.reward import (
    _action_value,
    _clamp,
    _efficiency_reward,
    _feedback_reward,
    _quality_reward,
    compute_reward,
)
from simulator.types import (
    Action,
    BilateralDecisionResult,
    DecisionResult,
    JointAction,
    MatchingContext,
    OutcomeResult,
    SideType,
    CandidateCard,
    TaskSpec,
    UserProfile,
)


@pytest.fixture
def config() -> SimulatorConfig:
    return SimulatorConfig(backend_type="mock", random_seed=321)


@pytest.fixture
def bilateral_mutual_accept() -> BilateralDecisionResult:
    req = DecisionResult(
        side=SideType.REQUESTER,
        utility=0.6,
        action=Action.ACCEPT,
        action_probs={"accept": 0.8, "skip": 0.1, "reject": 0.1},
        confidence=0.75,
        selected_personas=["rq_skill"],
        all_persona_opinions={},
        importance_scores={"rq_skill": 1.0},
        final_rationale="Good match",
    )
    cand = DecisionResult(
        side=SideType.CANDIDATE,
        utility=0.7,
        action=Action.ACCEPT,
        action_probs={"accept": 0.9, "skip": 0.05, "reject": 0.05},
        confidence=0.8,
        selected_personas=["cd_fit"],
        all_persona_opinions={},
        importance_scores={"cd_fit": 1.0},
        final_rationale="Good fit",
    )
    return BilateralDecisionResult(
        requester_decision=req,
        candidate_decision=cand,
        joint_action=JointAction.MUTUAL_ACCEPT,
        joint_accept_prob=0.8 * 0.9,
    )


@pytest.fixture
def outcome_good() -> OutcomeResult:
    return OutcomeResult(
        agreement_probability=0.85,
        expected_rounds=2.0,
        completion_probability=0.80,
        requester_satisfaction=0.78,
        candidate_satisfaction=0.75,
        outcome_rationale="Good outcome",
    )


@pytest.fixture
def outcome_poor() -> OutcomeResult:
    return OutcomeResult(
        agreement_probability=0.20,
        expected_rounds=6.0,
        completion_probability=0.15,
        requester_satisfaction=0.10,
        candidate_satisfaction=0.15,
        outcome_rationale="Poor outcome",
    )


class TestActionValue:
    def test_action_accept_value(self):
        assert _action_value(Action.ACCEPT) == 1.0

    def test_action_skip_value(self):
        assert _action_value(Action.SKIP) == 0.0

    def test_action_reject_value(self):
        assert _action_value(Action.REJECT) == -1.0


class TestClamp:
    def test_clamp_preserves_mid(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_clamps_high(self):
        assert _clamp(2.0) == 1.0

    def test_clamp_clamps_low(self):
        assert _clamp(-0.5) == 0.0


class TestFeedbackReward:
    def test_mutual_accept_high_reward(
        self, bilateral_mutual_accept: BilateralDecisionResult, config
    ):
        reward = _feedback_reward(bilateral_mutual_accept, config)
        assert 0.0 <= reward <= 1.0

    def test_mutual_reject_low_reward(
        self, bilateral_mutual_accept: BilateralDecisionResult, config
    ):
        req = DecisionResult(
            side=SideType.REQUESTER, utility=-0.5, action=Action.REJECT,
            action_probs={"accept": 0.1, "skip": 0.1, "reject": 0.8},
            confidence=0.8, selected_personas=[], all_persona_opinions={},
            importance_scores={}, final_rationale="",
        )
        cand = DecisionResult(
            side=SideType.CANDIDATE, utility=-0.5, action=Action.REJECT,
            action_probs={"accept": 0.1, "skip": 0.1, "reject": 0.8},
            confidence=0.8, selected_personas=[], all_persona_opinions={},
            importance_scores={}, final_rationale="",
        )
        bilateral = BilateralDecisionResult(
            requester_decision=req, candidate_decision=cand,
            joint_action=JointAction.MUTUAL_REJECT, joint_accept_prob=0.08,
        )
        reward = _feedback_reward(bilateral, config)
        # Mutual reject should give lower reward than mutual accept
        assert reward < _feedback_reward(bilateral_mutual_accept, config)

    def test_one_reject_penalized(
        self, bilateral_mutual_accept: BilateralDecisionResult, config
    ):
        req = bilateral_mutual_accept.requester_decision
        cand_new = DecisionResult(
            side=SideType.CANDIDATE, utility=-0.3, action=Action.REJECT,
            action_probs={"accept": 0.1, "skip": 0.1, "reject": 0.8},
            confidence=0.6, selected_personas=[], all_persona_opinions={},
            importance_scores={}, final_rationale="",
        )
        bilateral = BilateralDecisionResult(
            requester_decision=req, candidate_decision=cand_new,
            joint_action=JointAction.ONE_REJECT,
            joint_accept_prob=0.8 * 0.1,
        )
        reward = _feedback_reward(bilateral, config)
        # One reject should be penalized vs mutual accept
        assert reward < _feedback_reward(bilateral_mutual_accept, config)

    def test_all_joint_actions_have_distinct_rewards(self, config):
        """All 6 joint actions map to distinct feedback reward values."""
        from simulator.types import DecisionResult, BilateralDecisionResult
        req_base = dict(
            side=SideType.REQUESTER, utility=0.5, action=Action.ACCEPT,
            action_probs={"accept": 0.8, "skip": 0.1, "reject": 0.1},
            confidence=0.75, selected_personas=[], all_persona_opinions={},
            importance_scores={}, final_rationale="",
        )
        cand_base = dict(
            side=SideType.CANDIDATE, utility=0.5, action=Action.ACCEPT,
            action_probs={"accept": 0.8, "skip": 0.1, "reject": 0.1},
            confidence=0.75, selected_personas=[], all_persona_opinions={},
            importance_scores={}, final_rationale="",
        )

        cases = [
            (JointAction.MUTUAL_ACCEPT,                  Action.ACCEPT, Action.ACCEPT),
            (JointAction.REQUESTER_ACCEPT_CANDIDATE_SKIP, Action.ACCEPT, Action.SKIP),
            (JointAction.REQUESTER_SKIP_CANDIDATE_ACCEPT, Action.SKIP,   Action.ACCEPT),
            (JointAction.MUTUAL_SKIP,                    Action.SKIP,   Action.SKIP),
            (JointAction.ONE_REJECT,                     Action.ACCEPT, Action.REJECT),
            (JointAction.MUTUAL_REJECT,                   Action.REJECT, Action.REJECT),
        ]
        rewards = []
        for joint_action, req_act, cand_act in cases:
            req = DecisionResult(**req_base | {"action": req_act})
            cand = DecisionResult(**cand_base | {"action": cand_act})
            bilateral = BilateralDecisionResult(
                requester_decision=req, candidate_decision=cand,
                joint_action=joint_action, joint_accept_prob=0.5,
            )
            rewards.append(_feedback_reward(bilateral, config))

        # All values must be distinct
        assert len(set(rewards)) == len(rewards), (
            f"Joint actions collapsed onto same reward: {rewards}"
        )
        # Verify the semantic ordering: MUTUAL_ACCEPT > asymmetric > SKIP > ONE_REJECT > MUTUAL_REJECT
        assert rewards[0] > rewards[1]   # MUTUAL_ACCEPT > REQ_SKIP
        assert rewards[0] > rewards[4]   # MUTUAL_ACCEPT > ONE_REJECT
        assert rewards[0] > rewards[5]  # MUTUAL_ACCEPT > MUTUAL_REJECT
        assert rewards[3] > rewards[4]   # MUTUAL_SKIP > ONE_REJECT
        assert rewards[4] > rewards[5]   # ONE_REJECT > MUTUAL_REJECT


class TestEfficiencyReward:
    def test_few_rounds_high_reward(self, outcome_good: OutcomeResult, config):
        reward = _efficiency_reward(outcome_good, config)
        assert 0.0 <= reward <= 1.0

    def test_many_rounds_low_reward(self, outcome_good: OutcomeResult, outcome_poor: OutcomeResult, config):
        reward = _efficiency_reward(outcome_poor, config)
        assert 0.0 <= reward <= 1.0
        # More rounds → lower reward
        assert reward < _efficiency_reward(outcome_good, config)


class TestQualityReward:
    def test_quality_in_01(self, outcome_good: OutcomeResult, config):
        reward = _quality_reward(outcome_good, config)
        assert 0.0 <= reward <= 1.0

    def test_better_outcome_higher_quality(
        self, outcome_good: OutcomeResult, outcome_poor: OutcomeResult, config
    ):
        good_q = _quality_reward(outcome_good, config)
        poor_q = _quality_reward(outcome_poor, config)
        assert good_q > poor_q


class TestComputeReward:
    def test_total_reward_in_01(
        self,
        bilateral_mutual_accept: BilateralDecisionResult,
        outcome_good: OutcomeResult,
        config,
    ):
        result = compute_reward(bilateral_mutual_accept, outcome_good, config)
        assert 0.0 <= result.total_reward <= 1.0

    def test_total_reward_components_sum_weighted(
        self,
        bilateral_mutual_accept: BilateralDecisionResult,
        outcome_good: OutcomeResult,
        config,
    ):
        result = compute_reward(bilateral_mutual_accept, outcome_good, config)
        expected = (
            config.reward_feedback_weight * result.feedback_reward
            + config.reward_efficiency_weight * result.efficiency_reward
            + config.reward_quality_weight * result.quality_reward
        )
        assert abs(result.total_reward - expected) < 1e-6

    def test_breakdown_contains_all_keys(
        self,
        bilateral_mutual_accept: BilateralDecisionResult,
        outcome_good: OutcomeResult,
        config,
    ):
        result = compute_reward(bilateral_mutual_accept, outcome_good, config)
        expected_keys = {
            "feedback_reward", "efficiency_reward", "quality_reward",
            "joint_action", "joint_accept_prob", "agreement_prob",
            "completion_prob", "req_satisfaction", "cand_satisfaction",
        }
        assert expected_keys.issubset(result.breakdown.keys())

    def test_good_vs_poor_reward_ordering(
        self,
        bilateral_mutual_accept: BilateralDecisionResult,
        outcome_good: OutcomeResult,
        outcome_poor: OutcomeResult,
        config,
    ):
        result_good = compute_reward(bilateral_mutual_accept, outcome_good, config)
        # Build a mutual reject case
        req = DecisionResult(
            side=SideType.REQUESTER, utility=-0.5, action=Action.REJECT,
            action_probs={"accept": 0.1, "skip": 0.1, "reject": 0.8},
            confidence=0.7, selected_personas=[], all_persona_opinions={},
            importance_scores={}, final_rationale="",
        )
        cand = DecisionResult(
            side=SideType.CANDIDATE, utility=-0.5, action=Action.REJECT,
            action_probs={"accept": 0.1, "skip": 0.1, "reject": 0.8},
            confidence=0.7, selected_personas=[], all_persona_opinions={},
            importance_scores={}, final_rationale="",
        )
        bilateral_bad = BilateralDecisionResult(
            requester_decision=req, candidate_decision=cand,
            joint_action=JointAction.MUTUAL_REJECT, joint_accept_prob=0.01,
        )
        result_bad = compute_reward(bilateral_bad, outcome_poor, config)
        assert result_good.total_reward > result_bad.total_reward


class TestOutcomeSimulator:
    def test_outcome_simulator_produces_valid_result(self):
        config = SimulatorConfig(backend_type="mock", random_seed=42)
        backend = RuleBasedBackend(config)

        context = MatchingContext(
            requester=UserProfile(user_id="r1", role="pm"),
            candidate=UserProfile(user_id="c1", role="eng", capabilities={"python": 0.9}),
            task=TaskSpec(
                task_id="t1", title="Test Task", description="Test",
                required_skills={"python": 0.7},
            ),
            card=CandidateCard(candidate_id="c1", summary="Good eng",
                               highlighted_strengths=[], highlighted_risks=[]),
        )

        req = DecisionResult(
            side=SideType.REQUESTER, utility=0.5, action=Action.ACCEPT,
            action_probs={"accept": 0.7, "skip": 0.2, "reject": 0.1},
            confidence=0.7, selected_personas=[], all_persona_opinions={},
            importance_scores={}, final_rationale="",
        )
        cand = DecisionResult(
            side=SideType.CANDIDATE, utility=0.6, action=Action.ACCEPT,
            action_probs={"accept": 0.8, "skip": 0.1, "reject": 0.1},
            confidence=0.7, selected_personas=[], all_persona_opinions={},
            importance_scores={}, final_rationale="",
        )
        from simulator.types import BilateralDecisionResult
        bilateral = BilateralDecisionResult(
            requester_decision=req, candidate_decision=cand,
            joint_action=JointAction.MUTUAL_ACCEPT, joint_accept_prob=0.56,
        )

        sim = OutcomeSimulator(config)
        outcome = sim.simulate(context, bilateral)

        assert 0.0 <= outcome.agreement_probability <= 1.0
        assert 0.0 <= outcome.completion_probability <= 1.0
        assert 0.0 <= outcome.requester_satisfaction <= 1.0
        assert 0.0 <= outcome.candidate_satisfaction <= 1.0
        assert outcome.expected_rounds >= 1.0
        assert len(outcome.outcome_rationale) > 0
