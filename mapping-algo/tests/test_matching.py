"""Tests for the MapScore matching algorithm.

Loads JSON fixtures from tests/fixtures/ and validates:
  1. Vector utils (cosine_sim, softmax, attention)
  2. Gate logic (clearance, hard constraints)
  3. S_cap scoring
  4. S_need scoring
  5. MapScore composition
  6. UCB enhancement
  7. 1-1 pipeline (ranking)
  8. 1-N pipeline (team building, submodularity)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────
ALGO_DIR = Path(__file__).resolve().parents[1]   # mapping-algo/
REPO_ROOT = ALGO_DIR.parent                       # clawbot-matching/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ALGO_DIR))

from datatypes import (  # noqa: E402
    CapabilityEntry, NeedEntry, UserState,
    TaskRequirement, TaskOffer, Task,
    GapDetail, NeedDetail, MatchResult, TeamResult,
)
from config import MatchConfig  # noqa: E402
from scoring import (  # noqa: E402
    compute_gate, compute_s_cap, compute_s_need, compute_match_score,
)
from pipeline import match_one_to_one, match_one_to_n  # noqa: E402
from utils import (  # noqa: E402
    attention_weighted_value, cosine_sim, cosine_sim_batch, softmax,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ══════════════════════════════════════════════════════════════════════
#  Helpers: load JSON fixtures → Python objects
# ══════════════════════════════════════════════════════════════════════

def _load_json(name: str) -> dict:
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


def _arr(data: list) -> np.ndarray:
    return np.array(data, dtype=np.float64)


def _make_user(d: dict) -> UserState:
    return UserState(
        user_id=d["user_id"],
        capabilities=[
            CapabilityEntry(
                embedding=_arr(c["embedding"]),
                mu=c["mu"], sigma=c["sigma"],
                source=c.get("source", "explicit"),
                description=c["description"],
            )
            for c in d.get("capabilities", [])
        ],
        needs=[
            NeedEntry(
                embedding=_arr(n["embedding"]),
                intensity=n["intensity"],
                description=n["description"],
            )
            for n in d.get("needs", [])
        ],
        clearance_level=d.get("clearance_level", 0),
    )


def _make_task(d: dict) -> Task:
    return Task(
        task_id=d["task_id"],
        goal=d["goal"],
        requirements=[
            TaskRequirement(
                embedding=_arr(r["embedding"]),
                level=r["level"],
                constraint_type=r["constraint_type"],
                description=r["description"],
            )
            for r in d.get("requirements", [])
        ],
        offers=[
            TaskOffer(
                embedding=_arr(o["embedding"]),
                strength=o["strength"],
                source=o.get("source", "explicit"),
                description=o["description"],
            )
            for o in d.get("offers", [])
        ],
        data_clearance=d.get("data_clearance", 0),
    )


@pytest.fixture
def academic_scenario():
    data = _load_json("scenario_academic.json")
    return {
        "requester": _make_user(data["requester"]),
        "task": _make_task(data["task"]),
        "candidates": [_make_user(c) for c in data["candidates"]],
        "expected": data["expected"],
        "dim": data["embedding_dim"],
    }


@pytest.fixture
def candidate_pool():
    data = _load_json("candidates_pool.json")
    return [_make_user(c) for c in data["candidates"]]


@pytest.fixture
def cfg(academic_scenario):
    # tau_hard=0.63 calibrated for BAAI/bge-base-en-v1.5 embeddings:
    #   "clinical data analysis" vs "clinical research"    ≈ 0.758  → Bob PASS
    #   "clinical data analysis" vs "clinical trials"      ≈ 0.700  → Carol PASS
    #   "clinical data analysis" vs "natural language processing" ≈ 0.598 → Eve FAIL
    # Range (0.598, 0.679) gives clean separation; 0.63 sits in the middle.
    return MatchConfig(embedding_dim=academic_scenario["dim"], tau_hard=0.63)


@pytest.fixture
def theta():
    return np.array([0.4, -0.4])  # w_c ≈ 0.69, w_n ≈ 0.31


# ══════════════════════════════════════════════════════════════════════
#  1. Vector utils
# ══════════════════════════════════════════════════════════════════════

class TestVectorUtils:
    def test_cosine_sim_identical(self):
        v = np.random.randn(64)
        assert cosine_sim(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_sim_orthogonal(self):
        v1 = np.zeros(64); v1[0] = 1.0
        v2 = np.zeros(64); v2[1] = 1.0
        assert cosine_sim(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_sim_zero_vector(self):
        assert cosine_sim(np.zeros(64), np.ones(64)) == 0.0

    def test_cosine_sim_batch_shape(self):
        q = np.random.randn(64)
        keys = np.random.randn(10, 64)
        result = cosine_sim_batch(q, keys)
        assert result.shape == (10,)

    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert s.sum() == pytest.approx(1.0, abs=1e-8)

    def test_softmax_equal_inputs(self):
        s = softmax(np.array([0.0, 0.0]))
        assert s[0] == pytest.approx(0.5, abs=1e-8)
        assert s[1] == pytest.approx(0.5, abs=1e-8)

    def test_attention_empty_keys(self):
        q = np.random.randn(64)
        assert attention_weighted_value(
            q, np.empty((0, 64)), np.array([]), 0.1
        ) == 0.0

    def test_attention_single_key(self):
        q = np.random.randn(64)
        # Single key → attention weight = 1.0, result = that value
        val = attention_weighted_value(
            q, q.reshape(1, -1), np.array([0.8]), 0.1
        )
        assert val == pytest.approx(0.8, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════
#  2. Gate logic
# ══════════════════════════════════════════════════════════════════════

class TestGate:
    def test_bob_passes_gate(self, academic_scenario, cfg):
        u = academic_scenario["requester"]
        bob = academic_scenario["candidates"][0]
        task = academic_scenario["task"]
        sigma, reason = compute_gate(u, bob, task, cfg)
        assert sigma == 1
        assert reason == ""

    def test_carol_passes_gate(self, academic_scenario, cfg):
        u = academic_scenario["requester"]
        carol = academic_scenario["candidates"][1]
        task = academic_scenario["task"]
        sigma, reason = compute_gate(u, carol, task, cfg)
        assert sigma == 1

    def test_dave_fails_clearance(self, academic_scenario, cfg):
        u = academic_scenario["requester"]
        dave = academic_scenario["candidates"][2]
        task = academic_scenario["task"]
        sigma, reason = compute_gate(u, dave, task, cfg)
        assert sigma == 0
        assert "clearance" in reason

    def test_eve_fails_hard_constraint(self, academic_scenario, cfg):
        """Eve has NLP + paper writing but no clinical capability."""
        u = academic_scenario["requester"]
        eve = academic_scenario["candidates"][3]
        task = academic_scenario["task"]
        sigma, reason = compute_gate(u, eve, task, cfg)
        assert sigma == 0
        assert "hard" in reason


# ══════════════════════════════════════════════════════════════════════
#  3. S_cap scoring
# ══════════════════════════════════════════════════════════════════════

class TestScap:
    def test_range(self, academic_scenario, cfg):
        u = academic_scenario["requester"]
        bob = academic_scenario["candidates"][0]
        task = academic_scenario["task"]
        score, details = compute_s_cap(bob, u, task, cfg)
        assert 0.0 <= score <= 1.0

    def test_no_soft_reqs_returns_one(self, cfg):
        u = UserState(user_id="u")
        v = UserState(user_id="v")
        task = Task(
            task_id="t", goal="test",
            requirements=[
                TaskRequirement(
                    embedding=np.random.randn(cfg.embedding_dim),
                    level=0.5, constraint_type="hard", description="x",
                )
            ],
        )
        score, details = compute_s_cap(v, u, task, cfg)
        assert score == 1.0
        assert details == []

    def test_details_have_coverage(self, academic_scenario, cfg):
        u = academic_scenario["requester"]
        bob = academic_scenario["candidates"][0]
        task = academic_scenario["task"]
        _, details = compute_s_cap(bob, u, task, cfg)
        assert len(details) == 2  # 2 soft requirements
        for d in details:
            assert hasattr(d, "gap")
            assert hasattr(d, "coverage")

    def test_perfect_coverage_when_v_exceeds_gap(self, cfg):
        """If v's capability exceeds u's gap, coverage should = gap (not exceed)."""
        dim = cfg.embedding_dim
        emb = np.random.randn(dim)
        emb /= np.linalg.norm(emb)

        u = UserState("u", capabilities=[
            CapabilityEntry(emb, mu=0.2, sigma=0.1, description="skill"),
        ])
        v = UserState("v", capabilities=[
            CapabilityEntry(emb, mu=0.9, sigma=0.1, description="skill"),
        ])
        task = Task("t", "test", requirements=[
            TaskRequirement(emb, level=0.5, constraint_type="soft",
                            description="skill"),
        ])

        score, details = compute_s_cap(v, u, task, cfg)
        assert score == pytest.approx(1.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════
#  4. S_need scoring
# ══════════════════════════════════════════════════════════════════════

class TestSneed:
    def test_range(self, academic_scenario, cfg):
        bob = academic_scenario["candidates"][0]
        task = academic_scenario["task"]
        score, details = compute_s_need(bob, task, cfg)
        assert 0.0 <= score <= 1.0

    def test_no_needs_returns_neutral(self, cfg):
        v = UserState(user_id="v")
        task = Task(task_id="t", goal="test")
        score, _ = compute_s_need(v, task, cfg)
        assert score == pytest.approx(0.5)

    def test_no_offers_returns_zero(self, cfg):
        v = UserState("v", needs=[
            NeedEntry(np.random.randn(cfg.embedding_dim), 0.8, "something"),
        ])
        task = Task(task_id="t", goal="test")  # no offers
        score, _ = compute_s_need(v, task, cfg)
        assert score == 0.0

    def test_high_match_gives_high_score(self, cfg):
        """When offer directly matches need, score should be high."""
        dim = cfg.embedding_dim
        emb = np.random.randn(dim)
        emb /= np.linalg.norm(emb)

        v = UserState("v", needs=[
            NeedEntry(emb, intensity=0.8, description="X"),
        ])
        task = Task("t", "test", offers=[
            TaskOffer(emb, strength=0.9, source="explicit", description="X"),
        ])
        score, details = compute_s_need(v, task, cfg)
        assert score > 0.8


# ══════════════════════════════════════════════════════════════════════
#  5. MapScore composition
# ══════════════════════════════════════════════════════════════════════

class TestMapScore:
    def test_gated_candidate_gets_zero(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        dave = academic_scenario["candidates"][2]  # fails clearance
        task = academic_scenario["task"]
        result = compute_match_score(u, dave, task, theta, cfg)
        assert result.match_score == 0.0
        assert result.sigma_gate == 0

    def test_passing_candidate_positive_score(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        bob = academic_scenario["candidates"][0]
        task = academic_scenario["task"]
        result = compute_match_score(u, bob, task, theta, cfg)
        assert result.match_score > 0
        assert result.sigma_gate == 1

    def test_weights_sum_to_one(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        bob = academic_scenario["candidates"][0]
        task = academic_scenario["task"]
        result = compute_match_score(u, bob, task, theta, cfg)
        assert result.w_c + result.w_n == pytest.approx(1.0, abs=1e-6)

    def test_score_equals_formula(self, academic_scenario, cfg, theta):
        """M = σ · (w_c · S_cap + w_n · S_need)"""
        u = academic_scenario["requester"]
        bob = academic_scenario["candidates"][0]
        task = academic_scenario["task"]
        result = compute_match_score(u, bob, task, theta, cfg)
        expected = result.sigma_gate * (
            result.w_c * result.s_cap + result.w_n * result.s_need
        )
        assert result.match_score == pytest.approx(expected, abs=1e-6)

    def test_to_dict_format(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        bob = academic_scenario["candidates"][0]
        task = academic_scenario["task"]
        result = compute_match_score(u, bob, task, theta, cfg)
        d = result.to_dict()
        assert "candidate_id" in d
        assert "match_score" in d
        assert "components" in d
        assert "S_cap" in d["components"]
        assert "S_need" in d["components"]
        assert "gap_coverage_detail" in d
        assert "need_satisfaction_detail" in d


# ══════════════════════════════════════════════════════════════════════
#  6. UCB enhancement
# ══════════════════════════════════════════════════════════════════════

class TestUCB:
    def test_ucb_increases_score(self, academic_scenario, cfg, theta):
        """UCB-enhanced score should be >= expected score."""
        u = academic_scenario["requester"]
        bob = academic_scenario["candidates"][0]
        task = academic_scenario["task"]

        r_base = compute_match_score(u, bob, task, theta, cfg, use_ucb=False)
        r_ucb = compute_match_score(u, bob, task, theta, cfg,
                                    use_ucb=True, round_t=10)
        assert r_ucb.match_score >= r_base.match_score - 1e-6
        assert r_ucb.ucb_bonus >= 0

    def test_ucb_bonus_decreases_with_low_sigma(self, cfg, theta):
        """Candidate with low σ should get small UCB bonus."""
        dim = cfg.embedding_dim
        emb = np.random.randn(dim)
        emb /= np.linalg.norm(emb)

        u = UserState("u")
        v_certain = UserState("v_certain", capabilities=[
            CapabilityEntry(emb, mu=0.5, sigma=0.01, description="X"),
        ])
        v_uncertain = UserState("v_uncertain", capabilities=[
            CapabilityEntry(emb, mu=0.5, sigma=0.5, description="X"),
        ])
        task = Task("t", "test", requirements=[
            TaskRequirement(emb, 0.8, "soft", "X"),
        ])

        r1 = compute_match_score(u, v_certain, task, theta, cfg,
                                 use_ucb=True, round_t=10)
        r2 = compute_match_score(u, v_uncertain, task, theta, cfg,
                                 use_ucb=True, round_t=10)
        assert r2.ucb_bonus > r1.ucb_bonus


# ══════════════════════════════════════════════════════════════════════
#  7. 1-1 pipeline
# ══════════════════════════════════════════════════════════════════════

class TestOneToOne:
    def test_returns_ranked_list(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        candidates = academic_scenario["candidates"]
        results = match_one_to_one(u, task, candidates, theta, cfg, top_k=5)
        # Only Bob and Carol pass gate
        assert len(results) == 2
        # Sorted descending
        for i in range(len(results) - 1):
            assert results[i].match_score >= results[i + 1].match_score

    def test_top_candidate_passed_gate(self, academic_scenario, cfg, theta):
        """Top-ranked candidate must be one that passed the gate."""
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        candidates = academic_scenario["candidates"]
        results = match_one_to_one(u, task, candidates, theta, cfg, top_k=5)
        assert results[0].candidate_id in {"bob_002", "carol_003"}

    def test_top_k_limits_output(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        candidates = academic_scenario["candidates"]
        results = match_one_to_one(u, task, candidates, theta, cfg, top_k=1)
        assert len(results) == 1

    def test_empty_pool(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        results = match_one_to_one(u, task, [], theta, cfg)
        assert results == []

    def test_larger_pool(self, academic_scenario, candidate_pool, cfg, theta):
        """Test with the larger 10-candidate pool."""
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        results = match_one_to_one(u, task, candidate_pool, theta, cfg, top_k=5)
        assert len(results) <= 5
        for r in results:
            assert r.sigma_gate == 1
            assert r.match_score > 0


# ══════════════════════════════════════════════════════════════════════
#  8. 1-N team building
# ══════════════════════════════════════════════════════════════════════

class TestOneToN:
    def test_builds_team_or_gap_already_covered(self, academic_scenario, cfg, theta):
        """If u already has no gap, termination = gap_covered with 0 members.
        Otherwise, at least one member is selected."""
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        candidates = academic_scenario["candidates"]
        result = match_one_to_n(u, task, candidates, theta, cfg)
        # Either a team was built or gap was already covered
        if result.termination_reason == "gap_covered":
            assert result.collective_coverage >= 0.0
        else:
            assert len(result.team_members) > 0
            assert result.collective_coverage > 0

    def test_team_respects_n_max(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        candidates = academic_scenario["candidates"]
        cfg.n_max = 1
        result = match_one_to_n(u, task, candidates, theta, cfg)
        assert len(result.team_members) <= 1
        cfg.n_max = 5  # reset

    def test_selection_order_decreasing_gain(self, academic_scenario, cfg, theta):
        """Submodularity ⇒ marginal gains are non-increasing."""
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        candidates = academic_scenario["candidates"]
        result = match_one_to_n(u, task, candidates, theta, cfg)
        gains = [g for _, g in result.selection_order]
        for i in range(len(gains) - 1):
            assert gains[i] >= gains[i + 1] - 1e-6

    def test_to_dict_format(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        candidates = academic_scenario["candidates"]
        result = match_one_to_n(u, task, candidates, theta, cfg)
        d = result.to_dict()
        assert "team" in d
        assert "members" in d["team"]
        assert "collective_coverage" in d["team"]
        assert "residual_gaps" in d["team"]
        assert "termination_reason" in d["team"]
        assert "per_member" in d

    def test_no_eligible_returns_empty(self, cfg, theta):
        """All candidates fail gate → empty team."""
        dim = cfg.embedding_dim
        emb = np.random.randn(dim)
        u = UserState("u")
        task = Task("t", "test", data_clearance=3, requirements=[
            TaskRequirement(emb, 0.5, "hard", "hard_X"),
            TaskRequirement(emb, 0.5, "soft", "soft_X"),
        ])
        candidates = [
            UserState("v1", clearance_level=0),
            UserState("v2", clearance_level=1),
        ]
        result = match_one_to_n(u, task, candidates, theta, cfg)
        assert result.team_members == []
        assert result.collective_coverage == 0.0

    def test_collective_coverage_in_range(self, academic_scenario, cfg, theta):
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        candidates = academic_scenario["candidates"]
        result = match_one_to_n(u, task, candidates, theta, cfg)
        assert 0.0 <= result.collective_coverage <= 1.0

    def test_larger_pool_team(self, academic_scenario, candidate_pool, cfg, theta):
        """Team building with the larger 10-candidate pool."""
        u = academic_scenario["requester"]
        task = academic_scenario["task"]
        cfg.n_max = 3
        result = match_one_to_n(u, task, candidate_pool, theta, cfg)
        assert len(result.team_members) <= 3
        cfg.n_max = 5
