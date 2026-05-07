"""L5 closed loop driven by SimulatorFeedback (instead of dummy).

Confirms the integration L1 → L2 → L4(simulator) → L5.1 → L5.2-4 actually
closes when the feedback source is the multi-persona simulator.

Existing test_l5_integration.py keeps using DummyFeedback for fast/precise
causal tests; this file is the integration-with-real-L4 smoke verification.
"""

import numpy as np
import pytest

from test_helpers import CFG, make_user, make_task
from WorldModel import WorldModel
from Online_learning import OnlineLearning
from feedback_provider import SimulatorFeedback
from ol_utils import find_cap


@pytest.fixture
def setup():
    # Pick μ/σ values that keep the match score well below 1.0 so updates
    # are observable. (When S_cap=1 and S_need=1 the score saturates and
    # subsequent μ changes can't move M.)
    alice = make_user("alice",
        caps=[("python programming", 0.2, 0.3, "explicit"),
              ("data analysis",      0.2, 0.3, "explicit")],
        needs=[("research collab", 0.6)])
    bob = make_user("bob",
        caps=[("python programming", 0.5, 0.4, "explicit"),
              ("data analysis",      0.4, 0.4, "explicit")],
        needs=[("mentorship", 0.5)])
    task = make_task(
        # High demand so even bob's μ leaves residual gaps → S_cap < 1
        reqs=[("python programming", 0.95, "soft"),
              ("data analysis",      0.95, "soft")],
        offers=[("research collab", 0.7, "explicit")])
    wm = WorldModel(config=CFG, theta_c=0.4, theta_n=-0.1)
    engine = OnlineLearning(wm, feedback_provider=SimulatorFeedback())
    return alice, bob, task, wm, engine


# ── Closed-loop with real simulator ───────────────────────────────────

def test_simulator_driven_loop_runs_without_crash(setup):
    alice, bob, task, wm, engine = setup
    for _ in range(5):
        report = engine.run_one_round(alice, bob, task)
    assert len(engine.history) == 5


def test_simulator_driven_reward_in_unit_interval(setup):
    """Original reward formula keeps R ∈ [0, 1]."""
    alice, bob, task, wm, engine = setup
    for _ in range(5):
        report = engine.run_one_round(alice, bob, task)
        R = report["reward"]["R"]
        assert 0.0 <= R <= 1.0, f"R={R} outside [0,1]"


def test_simulator_driven_state_propagates(setup):
    """Simulator feedback must drive μ/σ updates that L2 then sees."""
    alice, bob, task, wm, engine = setup
    score_before = wm.compute_match(alice, bob, task).match_score
    sigma_before = find_cap(bob, "python programming").sigma

    for _ in range(8):
        engine.run_one_round(alice, bob, task)

    score_after = wm.compute_match(alice, bob, task).match_score
    sigma_after = find_cap(bob, "python programming").sigma

    assert sigma_after < sigma_before, "σ must shrink under simulator feedback"
    assert score_before != score_after, "L2 score must reflect L5 updates"


def test_simulator_driven_invariants_hold(setup):
    """All standard invariants still hold under simulator feedback."""
    alice, bob, task, wm, engine = setup
    prev_sigma = {c.description: c.sigma for c in bob.capabilities}
    for _ in range(8):
        report = engine.run_one_round(alice, bob, task)
        # Weights
        assert abs(wm.w_c + wm.w_n - 1.0) < 1e-6
        assert 0 < wm.w_c < 1 and 0 < wm.w_n < 1
        # Match score in [0, 1]
        assert 0 <= report["match_score"]["M (no UCB)"] <= 1
        # σ never grows
        for cap in bob.capabilities:
            assert cap.sigma <= prev_sigma[cap.description] + 1e-6
            prev_sigma[cap.description] = cap.sigma


def test_simulator_trace_in_report(setup):
    """SimulatorFeedback's trace fields survive into report['feedback'] keys
    that the report dict pulls.

    The report only pulls the six standard fields, so the trace fields are
    visible only by inspecting the underlying provider — verify they at
    least don't break the loop and are accessible directly.
    """
    alice, bob, task, _, engine = setup
    fb = engine.feedback_provider.collect(
        alice, bob, task, engine.world_model.compute_match(alice, bob, task)
    )
    assert "_sim_joint_action" in fb
