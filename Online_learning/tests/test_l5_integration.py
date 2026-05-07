"""L5 integration tests — verifies closed-loop correctness.

Covers:
  1. Closed loop:    L5 update → L2 next call sees different state
  2. Causal direction: high R → μ↑,  low R → μ↓
  3. σ monotonically non-increasing per round
  4. Weight invariants: sum=1, both in (0, 1)
  5. M always in [0, 1]
  6. Gate=0 handled gracefully (no crash, M=0)
  7. UCB bonus shrinks as σ shrinks over rounds
"""

import numpy as np
import pytest

from test_helpers import CFG, make_user, make_task
from WorldModel import WorldModel
from Online_learning import OnlineLearning
from ol_utils import find_cap

np.random.seed(0)

# ── Shared feedback overrides ─────────────────────────────────────────

HIGH_REWARD = {"r_u": 1.0, "r_v": 1.0, "n_rounds": 3,
               "f_completion": 1.0, "stars_u": 5, "stars_v": 5}
LOW_REWARD  = {"r_u": 0.0, "r_v": 0.0, "n_rounds": 25,
               "f_completion": 0.0, "stars_u": 1, "stars_v": 1}


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def actors():
    """Standard requester + candidate + task used across multiple tests."""
    alice = make_user("alice",
        caps=[("bayesian statistics", 0.3, 0.3, "explicit"),
              ("python programming", 0.8, 0.1, "explicit")],
        needs=[("bayesian statistics mentorship", 0.8)])
    bob = make_user("bob",
        caps=[("bayesian statistics", 0.5, 0.5, "explicit"),
              ("python programming", 0.5, 0.5, "explicit")],
        needs=[("python programming practice", 0.7)])
    task = make_task(
        reqs=[("bayesian statistics", 0.8, "soft"),
              ("python programming", 0.6, "soft")],
        offers=[("research collaboration", 0.8, "explicit")])
    return alice, bob, task


@pytest.fixture
def fresh_engine(actors):
    alice, bob, task = actors
    wm = WorldModel(config=CFG, theta_c=0.4, theta_n=-0.1)
    engine = OnlineLearning(wm)
    return alice, bob, task, wm, engine


# ── 1. Closed-loop propagation ────────────────────────────────────────

def test_l5_updates_propagate_to_l2(fresh_engine):
    """After learning rounds, μ and L2 score must reflect L5 state changes."""
    alice, bob, task, wm, engine = fresh_engine
    score_before = wm.compute_match(alice, bob, task).match_score
    mu_before = find_cap(bob, "bayesian statistics").mu

    for _ in range(20):
        engine.run_one_round(alice, bob, task)

    score_after = wm.compute_match(alice, bob, task).match_score
    mu_after = find_cap(bob, "bayesian statistics").mu

    assert mu_before != mu_after,     "μ must change after Bayesian updates"
    assert score_before != score_after, "L2 score must reflect L5 state changes"


def test_sgd_updates_theta_when_scap_neq_sneed():
    """θ changes only when S_cap ≠ S_need (softmax Jacobian is zero when equal).

    Use a task with no offers so S_need=0 while S_cap>0, ensuring non-zero gradient.
    """
    alice = make_user("alice",
        caps=[("bayesian statistics", 0.3, 0.3, "explicit")],
        needs=[("bayesian statistics mentorship", 0.8)])
    bob = make_user("bob",
        caps=[("bayesian statistics", 0.7, 0.3, "explicit")],
        needs=[])
    # No offers → S_need=0, S_cap>0 → guaranteed non-zero gradient
    task_no_offers = make_task(
        reqs=[("bayesian statistics", 0.8, "soft")],
        offers=[])
    wm = WorldModel(config=CFG, theta_c=0.4, theta_n=-0.1)
    engine = OnlineLearning(wm)
    theta_before = wm.theta.copy()

    for _ in range(30):
        engine.run_one_round(alice, bob, task_no_offers)

    assert not np.allclose(wm.theta, theta_before), (
        f"θ must change when S_cap≠S_need: before={theta_before}, after={wm.theta}"
    )


def test_l5_mutation_is_in_place(fresh_engine):
    """L5 mutates the candidate UserState in-place; same object reflects updates."""
    alice, bob, task, wm, engine = fresh_engine
    cap = find_cap(bob, "bayesian statistics")
    mu0, sig0 = cap.mu, cap.sigma

    engine.run_one_round(alice, bob, task)

    # Same object, different values
    assert cap.mu != mu0 or cap.sigma != sig0, "CapabilityEntry must be mutated in-place"


# ── 2. Causal direction ───────────────────────────────────────────────

def test_high_reward_raises_mu(actors):
    """Sustained high reward should push μ upward from a low starting point."""
    alice, _, task = actors
    # Bob starts with low μ=0.2
    bob_low = make_user("bob_low",
        caps=[("bayesian statistics", 0.2, 0.4, "explicit"),
              ("python programming", 0.2, 0.4, "explicit")],
        needs=[])
    wm = WorldModel(config=CFG)
    engine = OnlineLearning(wm)

    mu0 = find_cap(bob_low, "bayesian statistics").mu
    for _ in range(30):
        engine.run_one_round(alice, bob_low, task, feedback_override=HIGH_REWARD)

    mu1 = find_cap(bob_low, "bayesian statistics").mu
    assert mu1 > mu0, f"High reward must increase μ: {mu0:.4f} → {mu1:.4f}"


def test_low_reward_lowers_mu(actors):
    """Sustained low reward should push μ downward from a high starting point."""
    alice, _, task = actors
    # Bob starts with high μ=0.9
    bob_high = make_user("bob_high",
        caps=[("bayesian statistics", 0.9, 0.4, "explicit"),
              ("python programming", 0.9, 0.4, "explicit")],
        needs=[])
    wm = WorldModel(config=CFG)
    engine = OnlineLearning(wm)

    mu0 = find_cap(bob_high, "bayesian statistics").mu
    for _ in range(30):
        engine.run_one_round(alice, bob_high, task, feedback_override=LOW_REWARD)

    mu1 = find_cap(bob_high, "bayesian statistics").mu
    assert mu1 < mu0, f"Low reward must decrease μ: {mu0:.4f} → {mu1:.4f}"


def test_high_reward_raises_weight_toward_cap(actors):
    """When S_cap >> S_need, high reward should not decrease w_c."""
    alice, bob, task = actors
    # Use a cap-dominant task (no offers → S_need≈0)
    task_cap_only = make_task(
        reqs=[("bayesian statistics", 0.8, "soft")],
        offers=[])
    wm = WorldModel(config=CFG, theta_c=0.4, theta_n=-0.1)
    engine = OnlineLearning(wm)
    wc0 = wm.w_c

    for _ in range(30):
        engine.run_one_round(alice, bob, task_cap_only, feedback_override=HIGH_REWARD)

    # With S_need≈0, the gradient pushes w_c up (or keeps it stable near max)
    assert wm.w_c >= wc0 - 0.05, f"w_c should not fall when S_cap dominates: {wc0:.4f} → {wm.w_c:.4f}"


# ── 3. σ monotonically non-increasing ────────────────────────────────

def test_sigma_never_increases(actors):
    """σ must decrease or stay the same every round (precision only adds)."""
    alice, bob, task = actors
    wm = WorldModel(config=CFG)
    engine = OnlineLearning(wm)

    prev_sigma = {cap.description: cap.sigma for cap in bob.capabilities}

    for _ in range(20):
        engine.run_one_round(alice, bob, task)
        for cap in bob.capabilities:
            assert cap.sigma <= prev_sigma[cap.description] + 1e-6, (
                f"σ increased for '{cap.description}': "
                f"{prev_sigma[cap.description]:.4f} → {cap.sigma:.4f}"
            )
            prev_sigma[cap.description] = cap.sigma


def test_sigma_strictly_decreases_over_many_rounds(actors):
    """After many rounds, σ must be meaningfully smaller than initial."""
    alice, bob, task = actors
    wm = WorldModel(config=CFG)
    engine = OnlineLearning(wm)

    sig0 = find_cap(bob, "bayesian statistics").sigma
    for _ in range(30):
        engine.run_one_round(alice, bob, task)
    sig1 = find_cap(bob, "bayesian statistics").sigma

    assert sig1 < sig0 * 0.5, f"σ should halve after 30 rounds: {sig0:.4f} → {sig1:.4f}"


# ── 4. Weight invariants ──────────────────────────────────────────────

def test_weights_always_sum_to_one(fresh_engine):
    """w_c + w_n must equal 1.0 after every round."""
    alice, bob, task, wm, engine = fresh_engine
    for _ in range(30):
        engine.run_one_round(alice, bob, task)
        assert abs(wm.w_c + wm.w_n - 1.0) < 1e-6, (
            f"weights don't sum to 1: w_c={wm.w_c:.6f} w_n={wm.w_n:.6f}"
        )


def test_weights_stay_in_open_interval(fresh_engine):
    """Both weights must remain strictly between 0 and 1."""
    alice, bob, task, wm, engine = fresh_engine
    for _ in range(50):
        engine.run_one_round(alice, bob, task)
        assert 0 < wm.w_c < 1, f"w_c out of range: {wm.w_c}"
        assert 0 < wm.w_n < 1, f"w_n out of range: {wm.w_n}"


# ── 5. M always in [0, 1] ─────────────────────────────────────────────

def test_match_score_always_in_range(fresh_engine):
    """match_score must be in [0, 1] every round, both with and without UCB."""
    alice, bob, task, wm, engine = fresh_engine
    for i in range(20):
        report = engine.run_one_round(alice, bob, task)
        m_no_ucb  = report["match_score"]["M (no UCB)"]
        m_ucb     = report["match_score"]["M (with UCB)"]
        assert 0 <= m_no_ucb <= 1,  f"Round {i}: M (no UCB) = {m_no_ucb}"
        assert 0 <= m_ucb    <= 1,  f"Round {i}: M (UCB)    = {m_ucb}"


# ── 6. Gate=0 handled gracefully ─────────────────────────────────────

def test_gate_fail_gives_zero_score_and_no_crash(actors):
    """When gate fails, match_score=0 and L5 loop doesn't crash."""
    alice, bob, task = actors
    # Task requires clearance 5; bob has clearance 0 → gate fails
    task_secure = make_task(
        reqs=[("bayesian statistics", 0.5, "soft")],
        offers=[("research collaboration", 0.8, "explicit")],
        data_clearance=5,
    )
    wm = WorldModel(config=CFG)

    result = wm.compute_match(alice, bob, task_secure)
    assert result.sigma_gate == 0
    assert result.match_score == 0.0

    # L5 must not crash on a zero-score result
    engine = OnlineLearning(wm)
    report = engine.run_one_round(alice, bob, task_secure)
    assert report["match_score"]["M (no UCB)"] == 0.0


def test_hard_constraint_fail_gives_zero_score(actors):
    """Hard constraint mismatch → gate=0 → M=0."""
    alice, bob, task = actors
    task_hard = make_task(
        reqs=[("quantum computing", 0.9, "hard")],
        offers=[])
    wm = WorldModel(config=CFG)

    result = wm.compute_match(alice, bob, task_hard)
    assert result.sigma_gate == 0
    assert result.match_score == 0.0


# ── 7. UCB bonus shrinks with σ ───────────────────────────────────────

def test_ucb_bonus_shrinks_over_rounds(actors):
    """UCB exploration bonus β·σ must decrease as σ shrinks."""
    alice, bob, task = actors
    wm = WorldModel(config=CFG)
    engine = OnlineLearning(wm)

    bonuses = []
    for _ in range(20):
        report = engine.run_one_round(alice, bob, task)
        ucb = report["path3_ucb"]
        bonus = ucb["bayesian statistics"]["beta_sigma"]
        bonuses.append(bonus)

    # Bonus in the last 5 rounds should be smaller than in first 5
    assert np.mean(bonuses[-5:]) < np.mean(bonuses[:5]), (
        f"UCB bonus should shrink over time: "
        f"first5={np.mean(bonuses[:5]):.4f} last5={np.mean(bonuses[-5:]):.4f}"
    )


def test_ucb_score_not_below_mu(actors):
    """UCB score μ̃ = μ + β·σ must be >= μ and <= 1.0."""
    alice, bob, task = actors
    wm = WorldModel(config=CFG)
    engine = OnlineLearning(wm)

    for _ in range(10):
        report = engine.run_one_round(alice, bob, task)
        for cap_key, scores in report["path3_ucb"].items():
            assert scores["ucb"] >= scores["mu"] - 1e-6, \
                f"UCB score below μ for {cap_key}"
            assert scores["ucb"] <= 1.0 + 1e-6, \
                f"UCB score exceeds 1.0 for {cap_key}"
