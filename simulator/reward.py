"""
simulator/reward.py
===================
Constructs Layer-5 reward signals from bilateral decisions and outcomes.

RewardResult:
  - feedback_reward: accept/skip/reject mapped to numeric, bilateral product
  - efficiency_reward: based on expected_rounds
  - quality_reward: based on completion_probability and satisfaction
  - total_reward: weighted combination of the above
"""
from __future__ import annotations

from simulator.config import SimulatorConfig
from simulator.types import (
    Action,
    BilateralDecisionResult,
    JointAction,
    OutcomeResult,
    RewardResult,
)


def compute_reward(
    bilateral: BilateralDecisionResult,
    outcome: OutcomeResult,
    config: SimulatorConfig,
) -> RewardResult:
    """
    Main entry point for reward construction.

    Returns a RewardResult with all components and total.
    """
    feedback_reward = _feedback_reward(bilateral, config)
    efficiency_reward = _efficiency_reward(outcome, config)
    quality_reward = _quality_reward(outcome, config)

    total = (
        config.reward_feedback_weight * feedback_reward
        + config.reward_efficiency_weight * efficiency_reward
        + config.reward_quality_weight * quality_reward
    )

    breakdown = {
        "feedback_reward": feedback_reward,
        "efficiency_reward": efficiency_reward,
        "quality_reward": quality_reward,
        "joint_action": bilateral.joint_action.value,
        "joint_accept_prob": bilateral.joint_accept_prob,
        "agreement_prob": outcome.agreement_probability,
        "completion_prob": outcome.completion_probability,
        "req_satisfaction": outcome.requester_satisfaction,
        "cand_satisfaction": outcome.candidate_satisfaction,
    }

    return RewardResult(
        feedback_reward=feedback_reward,
        efficiency_reward=efficiency_reward,
        quality_reward=quality_reward,
        total_reward=total,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Sub-reward functions
# ---------------------------------------------------------------------------

def _action_value(action: Action) -> float:
    """Map a single-side action to a numeric signal."""
    return {
        Action.ACCEPT: 1.0,
        Action.SKIP: 0.0,
        Action.REJECT: -1.0,
    }[action]


def _feedback_reward(bilateral: BilateralDecisionResult, config: SimulatorConfig) -> float:
    """
    feedback_reward ∈ [-0.5, 1].

    Uses a JointAction lookup table so all 6 joint outcomes receive distinct
    reward values.  The table reflects a clear semantic gradient:
      MUTUAL_ACCEPT               → fully aligned, +1.0
      REQUESTER_ACCEPT_CANDIDATE_SKIP  → requester moves, +0.55
      REQUESTER_SKIP_CANDIDATE_ACCEPT  → candidate moves, +0.30
      MUTUAL_SKIP                 → neither commits, 0.0
      ONE_REJECT                  → one side blocks, -0.45
      MUTUAL_REJECT               → both block, strongly negative, -0.95

    A soft boost proportional to joint_accept_prob is added so that a
    mutual accept with higher P(both accept) is rewarded slightly more.
    """
    # Base reward per joint action — all 6 outcomes have distinct values
    _JOINT_REWARD: dict[JointAction, float] = {
        JointAction.MUTUAL_ACCEPT: 1.00,
        JointAction.REQUESTER_ACCEPT_CANDIDATE_SKIP: 0.55,
        JointAction.REQUESTER_SKIP_CANDIDATE_ACCEPT: 0.30,
        JointAction.MUTUAL_SKIP: 0.00,
        JointAction.ONE_REJECT: -0.45,
        JointAction.MUTUAL_REJECT: -0.95,
    }

    base = _JOINT_REWARD.get(bilateral.joint_action, 0.0)

    # Soft boost: the stronger the independent accept signals, the better
    boost = 0.08 * bilateral.joint_accept_prob
    reward = base + boost

    # feedback_reward spans [-0.5, 1] (rejects can go slightly negative)
    return float(_clamp(reward, lo=-0.5, hi=1.0))


def _efficiency_reward(outcome: OutcomeResult, config: SimulatorConfig) -> float:
    """
    efficiency_reward ∈ [0, 1].

    Based on expected_rounds:
      - min_rounds = 1 (immediate agreement)
      - max_rounds = 10 (very slow negotiation)
    Reward decreases as expected_rounds increases.
    """
    min_r, max_r = 1.0, 10.0
    rounds = max(min_r, min(max_r, outcome.expected_rounds))
    normalized = 1.0 - (rounds - min_r) / (max_r - min_r)
    return float(_clamp(normalized))


def _quality_reward(outcome: OutcomeResult, config: SimulatorConfig) -> float:
    """
    quality_reward ∈ [0, 1].

    Based on:
      - completion_probability (40%)
      - requester satisfaction (30%)
      - candidate satisfaction (30%)
    """
    reward = (
        0.40 * outcome.completion_probability
        + 0.30 * outcome.requester_satisfaction
        + 0.30 * outcome.candidate_satisfaction
    )
    return float(_clamp(reward))


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
