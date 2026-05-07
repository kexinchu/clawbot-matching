"""Feedback provider contract tests.

All FeedbackProvider implementations must return the same six-field dict
so RewardFunction can consume any of them interchangeably.
"""

import pytest

from test_helpers import CFG, make_user, make_task
from WorldModel import WorldModel
from feedback_provider import (
    FeedbackProvider,
    DummyFeedback,
    SimulatorFeedback,
    HumanFeedback,
)
from Reward_function import RewardFunction


REQUIRED_FIELDS = {"r_u", "r_v", "n_rounds", "f_completion", "stars_u", "stars_v"}


@pytest.fixture
def actors_match():
    alice = make_user("alice",
        caps=[("python programming", 0.3, 0.3, "explicit")],
        needs=[("collab", 0.5)])
    bob = make_user("bob",
        caps=[("python programming", 0.8, 0.1, "explicit"),
              ("data analysis", 0.7, 0.1, "explicit")],
        needs=[("mentorship", 0.5)])
    task = make_task(
        reqs=[("python programming", 0.8, "soft"),
              ("data analysis", 0.6, "soft")],
        offers=[("research collab", 0.7, "explicit")])
    wm = WorldModel(config=CFG)
    match = wm.compute_match(alice, bob, task)
    return alice, bob, task, match


# ── Contract: all providers return the six-field dict ────────────────

def test_dummy_feedback_returns_required_fields(actors_match):
    alice, bob, task, match = actors_match
    fb = DummyFeedback().collect(alice, bob, task, match)
    assert REQUIRED_FIELDS.issubset(fb.keys())


def test_simulator_feedback_returns_required_fields(actors_match):
    alice, bob, task, match = actors_match
    fb = SimulatorFeedback().collect(alice, bob, task, match)
    assert REQUIRED_FIELDS.issubset(fb.keys())


def test_human_feedback_raises_until_phase3(actors_match):
    alice, bob, task, match = actors_match
    with pytest.raises(NotImplementedError):
        HumanFeedback().collect(alice, bob, task, match)


# ── Field type/range invariants ──────────────────────────────────────

@pytest.mark.parametrize("Provider", [DummyFeedback, SimulatorFeedback])
def test_feedback_field_ranges(Provider, actors_match):
    alice, bob, task, match = actors_match
    fb = Provider().collect(alice, bob, task, match)
    assert 0.0 <= fb["r_u"] <= 1.0
    assert 0.0 <= fb["r_v"] <= 1.0
    assert isinstance(fb["n_rounds"], int)
    assert fb["n_rounds"] >= 3
    assert fb["f_completion"] in (0.0, 0.5, 1.0)
    assert 1 <= fb["stars_u"] <= 5
    assert 1 <= fb["stars_v"] <= 5


# ── Contract with downstream RewardFunction ──────────────────────────

@pytest.mark.parametrize("Provider", [DummyFeedback, SimulatorFeedback])
def test_reward_function_consumes_any_provider(Provider, actors_match):
    alice, bob, task, match = actors_match
    fb = Provider().collect(alice, bob, task, match)
    R = RewardFunction().compute(fb)
    assert 0.0 <= R.R <= 1.0
    assert 0.0 <= R.r_feedback <= 1.0
    assert 0.0 <= R.r_efficiency <= 1.0
    assert 0.0 <= R.r_quality <= 1.0


# ── Simulator-specific: extra trace fields preserved ─────────────────

def test_simulator_feedback_includes_trace_fields(actors_match):
    alice, bob, task, match = actors_match
    fb = SimulatorFeedback().collect(alice, bob, task, match)
    assert "_sim_joint_action" in fb
    assert "_sim_agreement_prob" in fb
    assert "_sim_joint_accept_prob" in fb
    valid_actions = {
        "mutual_accept",
        "requester_accept_candidate_skip",
        "requester_skip_candidate_accept",
        "mutual_skip",
        "one_reject",
        "mutual_reject",
    }
    assert fb["_sim_joint_action"] in valid_actions
    assert 0.0 <= fb["_sim_agreement_prob"] <= 1.0
    assert 0.0 <= fb["_sim_joint_accept_prob"] <= 1.0
