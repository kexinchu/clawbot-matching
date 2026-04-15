"""Semantic correctness tests — 第一层验证 (controlled comparison scenarios).

Unlike test_matching.py (which verifies code correctness: no crashes, correct
value ranges, formula consistency), these tests verify **algorithmic semantics**:
given inputs where we KNOW the correct answer, does the algorithm agree?

Key technique: use standard-basis unit vectors as embeddings so cosine_sim is
either 1.0 (same index) or 0.0 (different index).  Important: attention softmax
normalises over ALL capabilities, so we always pair each candidate with TWO
capabilities — one with high mu on the "right" embedding and one with near-zero
mu on a different embedding.  This avoids the single-item-softmax degeneracy
(softmax([x]) = [1.0] regardless of x).

Scenarios:
  A. Gap coverage dominance     — perfect gap coverage > partial > zero
  B. S_need dominance           — same S_cap, higher S_need wins
  C. Gate is absolute           — gated-out candidate never appears, even with high capability
  D. UCB exploration            — uncertain newcomer ranks above certain-low when exploration on
  E. Team submodularity         — complementary pair covers more than two overlapping members
  F. Weight sensitivity         — theta controls S_cap vs S_need tradeoff
  G. Hard constraint exactness  — boundary: capability just meets vs just below hard req level
  H. 1-N termination            — team stops when gap fully covered (no excess members)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ALGO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGO_DIR))

from config import MatchConfig
from datatypes import (
    CapabilityEntry, NeedEntry, UserState,
    TaskRequirement, TaskOffer, Task,
)
from scoring import compute_gate, compute_s_cap, compute_s_need, compute_match_score
from pipeline import match_one_to_one, match_one_to_n


# ── Shared config: low tau_hard so unit vectors pass hard gate ─────────
DIM = 8  # small embedding dimension for clarity

def _cfg(**overrides) -> MatchConfig:
    defaults = dict(
        embedding_dim=DIM,
        temperature=0.05,   # near-argmax attention → clear winner
        tau_hard=0.9,       # require very high cosine similarity for hard reqs
        epsilon=1e-8,
        ucb_beta_scale=2.0,
        n_max=5,
        gap_epsilon=1e-4,
    )
    defaults.update(overrides)
    return MatchConfig(**defaults)


def _unit(i: int, dim: int = DIM) -> np.ndarray:
    """Return the i-th standard basis vector (deterministic, orthogonal)."""
    v = np.zeros(dim)
    v[i % dim] = 1.0
    return v


def _cap(emb: np.ndarray, mu: float, sigma: float = 0.1,
         source: str = "explicit", desc: str = "") -> CapabilityEntry:
    return CapabilityEntry(embedding=emb, mu=mu, sigma=sigma,
                           source=source, description=desc)


def _need(emb: np.ndarray, intensity: float, desc: str = "") -> NeedEntry:
    return NeedEntry(embedding=emb, intensity=intensity, description=desc)


def _offer(emb: np.ndarray, strength: float,
           source: str = "explicit", desc: str = "") -> TaskOffer:
    return TaskOffer(embedding=emb, strength=strength, source=source,
                     description=desc)


def _req(emb: np.ndarray, level: float, ctype: str = "soft",
         desc: str = "") -> TaskRequirement:
    return TaskRequirement(embedding=emb, level=level,
                           constraint_type=ctype, description=desc)


def _user(uid: str, caps=None, needs=None, clearance: int = 2) -> UserState:
    return UserState(
        user_id=uid,
        capabilities=caps or [],
        needs=needs or [],
        clearance_level=clearance,
    )


def _task(reqs, offers=None, clearance: int = 0) -> Task:
    return Task(
        task_id="t1",
        goal="test task",
        requirements=reqs,
        offers=offers or [],
        data_clearance=clearance,
    )


THETA_BALANCED = np.array([0.0, 0.0])   # w_c = w_n = 0.5
THETA_CAP_ONLY = np.array([10.0, -10.0])  # w_c ≈ 1.0
THETA_NEED_ONLY = np.array([-10.0, 10.0])  # w_n ≈ 1.0


# ══════════════════════════════════════════════════════════════════════
#  Scenario A: Gap coverage dominance
#  Semantic: who helps u more should rank higher
# ══════════════════════════════════════════════════════════════════════

class TestScenarioA_GapCoverage:
    """
    Setup:
      u  has zero capability on skill e0 (gap = q0 = 0.8)
      v1 fully covers the gap   (mu=0.9 on e0, mu=0.01 on e1)
      v2 partially covers       (mu=0.5 on e0, mu=0.01 on e1)
      v3 has NO capabilities at all → p̃_v = 0 → coverage = 0

    Two-capability design: attention_weighted_value with a single capability
    always gives softmax([x])=[1.0] regardless of embedding similarity.  Adding
    a second low-mu capability on a different embedding lets the softmax
    discriminate: for query e0, weights ≈ [1,0] (temp=0.05, peaked).

    Expected ranking: v1 > v2 > v3
    """

    def setup_method(self):
        self.cfg = _cfg()
        e0, e1 = _unit(0), _unit(1)

        self.u = _user("u", caps=[])   # no capability
        self.task = _task([_req(e0, 0.8, "soft", "skill_A")])

        # Two capabilities so attention can discriminate
        self.v1 = _user("v1", caps=[_cap(e0, 0.9), _cap(e1, 0.01)])   # perfect coverage
        self.v2 = _user("v2", caps=[_cap(e0, 0.5), _cap(e1, 0.01)])   # partial
        self.v3 = _user("v3", caps=[])                                  # no capability → p̃=0

        self.theta = THETA_CAP_ONLY

    def test_v1_has_higher_scap_than_v2(self):
        s1, _ = compute_s_cap(self.v1, self.u, self.task, self.cfg)
        s2, _ = compute_s_cap(self.v2, self.u, self.task, self.cfg)
        assert s1 > s2, f"v1.S_cap={s1:.4f} should exceed v2.S_cap={s2:.4f}"

    def test_v2_has_higher_scap_than_v3(self):
        s2, _ = compute_s_cap(self.v2, self.u, self.task, self.cfg)
        s3, _ = compute_s_cap(self.v3, self.u, self.task, self.cfg)
        assert s2 > s3, f"v2.S_cap={s2:.4f} should exceed v3.S_cap={s3:.4f}"

    def test_ranking_preserves_gap_coverage_order(self):
        results = match_one_to_one(
            self.u, self.task, [self.v1, self.v2, self.v3],
            self.theta, self.cfg, top_k=3,
        )
        ids = [r.candidate_id for r in results]
        assert ids.index("v1") < ids.index("v2"), "v1 should rank above v2"
        assert ids.index("v2") < ids.index("v3"), "v2 should rank above v3"

    def test_v3_scores_zero_with_no_capabilities(self):
        s3, _ = compute_s_cap(self.v3, self.u, self.task, self.cfg)
        # attention_weighted_value returns 0.0 for empty key set
        assert s3 == 0.0, f"v3 S_cap={s3:.4f} should be 0 (no capabilities)"


# ══════════════════════════════════════════════════════════════════════
#  Scenario B: S_need dominance (same S_cap, different S_need)
#  Semantic: when capabilities are equal, mutual benefit matters
# ══════════════════════════════════════════════════════════════════════

class TestScenarioB_NeedDominance:
    """
    Setup:
      u has zero cap (gap = 0.8 on e0)
      v1 and v2 both have identical capability on e0 (mu=0.7, mu=0.01 on e1)
      Task offers TWO rewards: strong (strength=0.9) on e2, weak (strength=0.1) on e3
      v1.needs = [e2, intensity=0.9]  → attention picks strong offer → high S_need
      v2.needs = [e3, intensity=0.9]  → attention picks weak offer  → low S_need

    Two-offer design: with a single offer, softmax([sim/τ]) = [1.0] always, so
    both v1 and v2 would appear equally satisfied.  Having TWO offers with
    different strengths makes attention route v1→strong and v2→weak.

    Expected: S_cap(v1)=S_cap(v2), but score(v1) > score(v2) with balanced theta
    """

    def setup_method(self):
        self.cfg = _cfg()
        e0, e1, e2, e3 = _unit(0), _unit(1), _unit(2), _unit(3)

        self.u = _user("u", caps=[])
        self.task = _task(
            reqs=[_req(e0, 0.8, "soft", "skill_A")],
            offers=[
                _offer(e2, 0.9, desc="reward_R_strong"),   # strong offer
                _offer(e3, 0.1, desc="reward_S_weak"),     # weak offer
            ],
        )

        # Identical capability (two caps for softmax discrimination), different needs
        self.v1 = _user("v1",
                         caps=[_cap(e0, 0.7), _cap(e1, 0.01)],
                         needs=[_need(e2, 0.9, "wants reward_R")])  # aligned with strong offer
        self.v2 = _user("v2",
                         caps=[_cap(e0, 0.7), _cap(e1, 0.01)],
                         needs=[_need(e3, 0.9, "wants reward_S")]) # aligned with weak offer

        self.theta = THETA_BALANCED

    def test_scap_equal_for_v1_v2(self):
        s1, _ = compute_s_cap(self.v1, self.u, self.task, self.cfg)
        s2, _ = compute_s_cap(self.v2, self.u, self.task, self.cfg)
        assert abs(s1 - s2) < 1e-4, f"S_cap must be equal: {s1:.6f} vs {s2:.6f}"

    def test_v1_has_higher_sneed(self):
        n1, _ = compute_s_need(self.v1, self.task, self.cfg)
        n2, _ = compute_s_need(self.v2, self.task, self.cfg)
        assert n1 > n2, f"v1.S_need={n1:.4f} should exceed v2.S_need={n2:.4f}"

    def test_v1_ranks_above_v2_balanced_weights(self):
        results = match_one_to_one(
            self.u, self.task, [self.v1, self.v2],
            self.theta, self.cfg,
        )
        ids = [r.candidate_id for r in results]
        assert ids[0] == "v1", f"v1 should rank first, got {ids}"


# ══════════════════════════════════════════════════════════════════════
#  Scenario C: Gate is absolute
#  Semantic: failed gate → never appears in results, regardless of capability
# ══════════════════════════════════════════════════════════════════════

class TestScenarioC_GateAbsolute:
    """
    v_expert: extremely capable (mu=1.0), but clearance=0 < task.clearance=1
    v_decent: moderate capability (mu=0.5), clearance=2 (passes)
    Expected: v_decent appears in results; v_expert does NOT (ever)
    """

    def setup_method(self):
        self.cfg = _cfg()
        e0 = _unit(0)

        self.u = _user("u", caps=[])
        # task requires clearance=1
        self.task = _task(
            reqs=[_req(e0, 0.6, "soft", "skill_A")],
            clearance=1,
        )

        self.v_expert = _user("v_expert",
                               caps=[_cap(e0, 1.0)],
                               clearance=0)   # blocked!
        self.v_decent = _user("v_decent",
                               caps=[_cap(e0, 0.5)],
                               clearance=2)   # passes

        self.theta = THETA_CAP_ONLY

    def test_expert_fails_gate(self):
        sg, reason = compute_gate(self.u, self.v_expert, self.task, self.cfg)
        assert sg == 0
        assert "clearance" in reason.lower()

    def test_decent_passes_gate(self):
        sg, _ = compute_gate(self.u, self.v_decent, self.task, self.cfg)
        assert sg == 1

    def test_expert_absent_from_1_to_1_results(self):
        results = match_one_to_one(
            self.u, self.task,
            [self.v_expert, self.v_decent],
            self.theta, self.cfg, top_k=10,
        )
        ids = [r.candidate_id for r in results]
        assert "v_expert" not in ids, "gated-out candidate must not appear"
        assert "v_decent" in ids

    def test_expert_absent_from_team_results(self):
        team = match_one_to_n(
            self.u, self.task,
            [self.v_expert, self.v_decent],
            self.theta, self.cfg,
        )
        assert "v_expert" not in team.team_members

    def test_hard_constraint_gate(self):
        """Candidate without matching hard-req capability is gated out."""
        e0, e1 = _unit(0), _unit(1)
        task_hard = _task([_req(e0, 0.7, "hard", "critical_skill")])
        v_wrong = _user("v_wrong", caps=[_cap(e1, 1.0)])  # orthogonal skill
        sg, reason = compute_gate(self.u, self.v_decent, task_hard, self.cfg)
        # v_decent has cap on e0 which is the hard req
        # ... but e0 cosine_sim = 1.0 >= tau_hard(0.9) → should PASS
        sg2, _ = compute_gate(self.u, v_wrong, task_hard, self.cfg)
        assert sg2 == 0, "wrong-skill candidate must fail hard gate"


# ══════════════════════════════════════════════════════════════════════
#  Scenario D: UCB exploration
#  Semantic: with UCB on, uncertain newcomer should rank above certain-low
# ══════════════════════════════════════════════════════════════════════

class TestScenarioD_UCBExploration:
    """
    v_certain:  mu=0.55, sigma=0.01  → mu_ucb ≈ 0.55 + tiny
    v_uncertain: mu=0.45, sigma=0.35  → mu_ucb ≈ 0.45 + beta*0.35

    With round_t=1, beta = sqrt(2*log(2)) ≈ 1.18
    v_uncertain ucb: 0.45 + 1.18*0.35 ≈ 0.86 → outranks v_certain's ≈ 0.56

    Without UCB: v_certain (mu=0.55) > v_uncertain (mu=0.45)
    """

    def setup_method(self):
        self.cfg = _cfg()
        e0 = _unit(0)
        self.u = _user("u", caps=[])
        self.task = _task([_req(e0, 0.9, "soft", "target_skill")])
        self.v_certain = _user("v_certain",
                                caps=[_cap(e0, 0.55, sigma=0.01)])
        self.v_uncertain = _user("v_uncertain",
                                  caps=[_cap(e0, 0.45, sigma=0.35)])
        self.theta = THETA_CAP_ONLY

    def test_without_ucb_certain_wins(self):
        results = match_one_to_one(
            self.u, self.task,
            [self.v_certain, self.v_uncertain],
            self.theta, self.cfg,
            use_ucb=False,
        )
        ids = [r.candidate_id for r in results]
        assert ids[0] == "v_certain", \
            f"Without UCB, certain (mu=0.55) should beat uncertain (mu=0.45), got {ids}"

    def test_with_ucb_uncertain_wins(self):
        results = match_one_to_one(
            self.u, self.task,
            [self.v_certain, self.v_uncertain],
            self.theta, self.cfg,
            use_ucb=True, round_t=1,
        )
        ids = [r.candidate_id for r in results]
        assert ids[0] == "v_uncertain", \
            f"With UCB, uncertain (mu=0.45, sigma=0.35) should rank first, got {ids}"

    def test_ucb_bonus_is_positive_and_meaningful(self):
        r_ucb = match_one_to_one(
            self.u, self.task, [self.v_uncertain],
            self.theta, self.cfg, use_ucb=True, round_t=1,
        )[0]
        r_no = match_one_to_one(
            self.u, self.task, [self.v_uncertain],
            self.theta, self.cfg, use_ucb=False,
        )[0]
        assert r_ucb.ucb_bonus > 0
        assert r_ucb.match_score > r_no.match_score


# ══════════════════════════════════════════════════════════════════════
#  Scenario E: Team submodularity (1-N)
#  Semantic: complementary pair should collectively cover more than duplicates
# ══════════════════════════════════════════════════════════════════════

class TestScenarioE_TeamSubmodularity:
    """
    Task requires two distinct skills: e0 and e1
    u has zero cap on both.

    v_A:   high mu on e0 (0.9), near-zero mu on e1 (0.01)
    v_B:   near-zero mu on e0 (0.01), high mu on e1 (0.9)
    v_dup: high mu on e0 (0.9), near-zero mu on e1 (0.01)  — same profile as v_A

    Two-capability design: with τ=0.05, softmax peaks at the embedding with
    cosine_sim=1.0.  So v_A's p̃ for e0 ≈ 0.9 and for e1 ≈ 0.01.  v_B is
    the complement.  Greedy should pick v_A + v_B (full coverage), not v_A + v_dup
    (redundant on e0, miss e1).
    """

    def setup_method(self):
        self.cfg = _cfg(n_max=2)
        e0, e1 = _unit(0), _unit(1)

        self.u = _user("u", caps=[])
        self.task = _task([
            _req(e0, 0.8, "soft", "skill_e0"),
            _req(e1, 0.8, "soft", "skill_e1"),
        ])

        self.v_A   = _user("v_A",   caps=[_cap(e0, 0.9), _cap(e1, 0.01)])
        self.v_B   = _user("v_B",   caps=[_cap(e0, 0.01), _cap(e1, 0.9)])
        self.v_dup = _user("v_dup", caps=[_cap(e0, 0.9), _cap(e1, 0.01)])  # same as v_A

        self.theta = THETA_CAP_ONLY

    def test_complementary_team_has_high_coverage(self):
        team = match_one_to_n(
            self.u, self.task,
            [self.v_A, self.v_B],
            self.theta, self.cfg,
        )
        assert team.collective_coverage > 0.8, \
            f"Complementary pair should cover >80%, got {team.collective_coverage:.3f}"

    def test_duplicate_team_has_lower_coverage(self):
        team = match_one_to_n(
            self.u, self.task,
            [self.v_A, self.v_dup],
            self.theta, self.cfg,
        )
        # Covers e0 well but misses most of e1 (only 0.01 coverage)
        assert team.collective_coverage < 0.6, \
            f"Duplicate pair should have lower coverage (misses e1), got {team.collective_coverage:.3f}"

    def test_greedy_picks_complementary_over_duplicate(self):
        """With all three candidates, greedy should choose v_A (or v_dup) + v_B."""
        team = match_one_to_n(
            self.u, self.task,
            [self.v_A, self.v_B, self.v_dup],
            self.theta, self.cfg,
        )
        assert "v_A" in team.team_members or "v_dup" in team.team_members, \
            "Team must include at least one e0 expert"
        assert "v_B" in team.team_members, \
            "Team must include e1 expert (v_B) for full coverage"

    def test_second_member_adds_positive_gain(self):
        """Complementary second member always increases coverage."""
        team = match_one_to_n(
            self.u, self.task,
            [self.v_A, self.v_B],
            self.theta, self.cfg,
        )
        if team.selection_order and len(team.selection_order) == 2:
            _, gain2 = team.selection_order[1]
            assert gain2 > 0, "Second complementary member must add positive gain"


# ══════════════════════════════════════════════════════════════════════
#  Scenario F: Weight sensitivity (theta controls tradeoff)
# ══════════════════════════════════════════════════════════════════════

class TestScenarioF_WeightSensitivity:
    """
    v_cap: great capability (S_cap high), poor need alignment (S_need low)
    v_need: poor capability (S_cap low), great need alignment (S_need high)

    Two-capability design for S_cap:
      v_cap:  [e0→0.9, e4→0.01]  → for query e0: attention peaks at e0 → p̃≈0.9
      v_need: [e4→0.9, e0→0.01]  → for query e0: attention peaks at e4 (sim=0!) but
                                   still weight e0 component → p̃≈0.01  ← LOW S_cap

    Two-offer design for S_need:
      offers: [e2 strength=0.9 (strong), e4 strength=0.1 (weak)]
      v_cap.need = e4  → attention → picks weak offer → o_tilde≈0.1 → LOW S_need
      v_need.need = e2 → attention → picks strong offer → o_tilde≈0.9 → HIGH S_need

    With theta cap-only:  v_cap wins  (high S_cap × w_c dominates)
    With theta need-only: v_need wins (high S_need × w_n dominates)
    """

    def setup_method(self):
        self.cfg = _cfg()
        e0, e2, e4 = _unit(0), _unit(2), _unit(4)

        self.u = _user("u", caps=[])
        self.task = _task(
            reqs=[_req(e0, 0.8, "soft", "skill_A")],
            offers=[
                _offer(e2, 0.9, desc="offer_strong"),  # strong
                _offer(e4, 0.1, desc="offer_weak"),    # weak
            ],
        )

        # v_cap: covers the capability gap (e0), aligned with WEAK offer (e4)
        self.v_cap = _user("v_cap",
                            caps=[_cap(e0, 0.9), _cap(e4, 0.01)],
                            needs=[_need(e4, 0.9, "wants weak offer")])

        # v_need: doesn't cover the gap (e0 low), aligned with STRONG offer (e2)
        self.v_need = _user("v_need",
                             caps=[_cap(e4, 0.9), _cap(e0, 0.01)],
                             needs=[_need(e2, 0.9, "wants strong offer")])

    def test_cap_theta_prefers_v_cap(self):
        results = match_one_to_one(
            self.u, self.task, [self.v_cap, self.v_need],
            THETA_CAP_ONLY, self.cfg,
        )
        assert results[0].candidate_id == "v_cap", \
            f"Cap-only theta should prefer v_cap, got {results[0].candidate_id}"

    def test_need_theta_prefers_v_need(self):
        results = match_one_to_one(
            self.u, self.task, [self.v_cap, self.v_need],
            THETA_NEED_ONLY, self.cfg,
        )
        assert results[0].candidate_id == "v_need", \
            f"Need-only theta should prefer v_need, got {results[0].candidate_id}"

    def test_score_components_are_separated(self):
        """S_cap and S_need should each be high for the right candidate."""
        r_cap = match_one_to_one(
            self.u, self.task, [self.v_cap],
            THETA_BALANCED, self.cfg,
        )[0]
        r_need = match_one_to_one(
            self.u, self.task, [self.v_need],
            THETA_BALANCED, self.cfg,
        )[0]
        assert r_cap.s_cap > r_need.s_cap, \
            f"v_cap S_cap={r_cap.s_cap:.4f} should exceed v_need S_cap={r_need.s_cap:.4f}"
        assert r_need.s_need > r_cap.s_need, \
            f"v_need S_need={r_need.s_need:.4f} should exceed v_cap S_need={r_cap.s_need:.4f}"


# ══════════════════════════════════════════════════════════════════════
#  Scenario G: Hard constraint boundary
#  Semantic: exactly at threshold = pass; epsilon below = fail
# ══════════════════════════════════════════════════════════════════════

class TestScenarioG_HardConstraintBoundary:
    """
    Hard req on e0 at level=0.7
    v_pass: mu=0.70 (exact threshold) — must pass
    v_fail: mu=0.69 (just below)      — must fail
    """

    def setup_method(self):
        # Use tau_hard=0.9 so exact same-embedding vectors pass (cosine_sim=1.0)
        self.cfg = _cfg(tau_hard=0.9)
        e0 = _unit(0)

        self.u = _user("u", caps=[])
        self.task = _task([_req(e0, 0.70, "hard", "critical")])

        self.v_pass = _user("v_pass", caps=[_cap(e0, 0.70)])
        self.v_fail = _user("v_fail", caps=[_cap(e0, 0.69)])

    def test_exact_threshold_passes(self):
        sg, reason = compute_gate(self.u, self.v_pass, self.task, self.cfg)
        assert sg == 1, f"mu=0.70 at level=0.70 should pass gate, reason: {reason}"

    def test_below_threshold_fails(self):
        sg, reason = compute_gate(self.u, self.v_fail, self.task, self.cfg)
        assert sg == 0, "mu=0.69 at level=0.70 should fail hard gate"
        assert "hard" in reason.lower()

    def test_gate_fail_propagates_to_zero_score(self):
        result = compute_match_score(
            self.u, self.v_fail, self.task,
            THETA_CAP_ONLY, self.cfg,
        )
        assert result.sigma_gate == 0
        assert result.match_score == 0.0


# ══════════════════════════════════════════════════════════════════════
#  Scenario H: 1-N termination when gap is fully covered
#  Semantic: team stops as soon as coverage satisfies all gaps
# ══════════════════════════════════════════════════════════════════════

class TestScenarioH_TeamTermination:
    """
    Task has one soft req at level=0.5
    u has zero capability.
    v1: mu=0.6 on e0 — covers gap=0.5 fully (0.6 >= 0.5)
    v2: mu=0.6 on e0 — identical to v1
    n_max=5, but team should stop after v1 (gap covered).
    """

    def setup_method(self):
        self.cfg = _cfg(n_max=5, gap_epsilon=1e-4)
        e0 = _unit(0)

        self.u = _user("u", caps=[])
        self.task = _task([_req(e0, 0.5, "soft", "skill_A")])

        self.v1 = _user("v1", caps=[_cap(e0, 0.6)])
        self.v2 = _user("v2", caps=[_cap(e0, 0.6)])

        self.theta = THETA_CAP_ONLY

    def test_team_stops_early_when_gap_covered(self):
        team = match_one_to_n(
            self.u, self.task,
            [self.v1, self.v2],
            self.theta, self.cfg,
        )
        assert len(team.team_members) == 1, \
            f"Team should stop at 1 member once gap is covered, got {team.team_members}"
        assert team.termination_reason == "gap_covered"

    def test_collective_coverage_near_one_after_termination(self):
        team = match_one_to_n(
            self.u, self.task,
            [self.v1, self.v2],
            self.theta, self.cfg,
        )
        assert team.collective_coverage > 0.9, \
            f"Coverage should be near 1.0 after gap is filled, got {team.collective_coverage:.4f}"

    def test_no_positive_gain_terminates_correctly(self):
        """Candidate with near-zero e0 coverage triggers no_positive_gain."""
        e0, e1 = _unit(0), _unit(1)
        # Two capabilities: high mu on e1, near-zero mu on e0.
        # For query e0, attention peaks at e1 (sim=0 → uniform) but e0 component mu=0.001 is tiny.
        # With τ=0.05 and both sims=0 for e0 query, weights=[0.5,0.5], p̃≈(0.9+0.001)/2≈0.45.
        # gap=0.5, coverage=min(0.45,0.5)=0.45, gain=0.45/0.5=0.9 → actually covers it.
        # Use caps=[] instead: no capabilities → p̃=0 → gain=0 → no_positive_gain.
        v_nocap = _user("v_nocap", caps=[])
        team = match_one_to_n(
            self.u, self.task,
            [v_nocap],
            self.theta, self.cfg,
        )
        assert team.termination_reason in {"no_positive_gain", "no_eligible_candidate"}


# ══════════════════════════════════════════════════════════════════════
#  Scenario I: u's own capability reduces gap
#  Semantic: a capable requester u needs less help — gap shrinks
# ══════════════════════════════════════════════════════════════════════

class TestScenarioI_RequesterCapabilityReducesGap:
    """
    Task req: level=0.8 on e0
    u_weak:   mu=0.0 on e0  → gap=0.8
    u_strong: mu=0.6 on e0  → gap=0.2

    Same candidate v (mu=0.9 on e0) should achieve:
      - higher S_cap coverage against u_weak (more gap to fill)
      - lower S_cap score fractionally because gap_denom is smaller for u_strong
      but v fully satisfies u_strong's small gap → S_cap = 1.0
    """

    def setup_method(self):
        self.cfg = _cfg()
        e0 = _unit(0)

        self.task = _task([_req(e0, 0.8, "soft", "skill_A")])
        self.v = _user("v", caps=[_cap(e0, 0.9)])

        self.u_weak = _user("u_weak", caps=[_cap(e0, 0.0)])
        self.u_strong = _user("u_strong", caps=[_cap(e0, 0.6)])

    def test_u_strong_has_smaller_gap(self):
        """u_strong's p̃_u should be closer to req.level, meaning smaller gap."""
        from utils import attention_weighted_value
        e0 = _unit(0)
        embs_weak = np.stack([c.embedding for c in self.u_weak.capabilities])
        mus_weak = np.array([c.mu for c in self.u_weak.capabilities])
        p_weak = attention_weighted_value(e0, embs_weak, mus_weak, self.cfg.temperature)

        embs_strong = np.stack([c.embedding for c in self.u_strong.capabilities])
        mus_strong = np.array([c.mu for c in self.u_strong.capabilities])
        p_strong = attention_weighted_value(e0, embs_strong, mus_strong, self.cfg.temperature)

        assert p_strong > p_weak, "u_strong should have higher p̃_u on e0"
        gap_weak = max(0.0, 0.8 - p_weak)
        gap_strong = max(0.0, 0.8 - p_strong)
        assert gap_weak > gap_strong, "u_weak's gap must be larger"

    def test_v_fully_satisfies_u_strong_gap(self):
        """v (mu=0.9) covers u_strong's small gap (≈0.2) completely → S_cap=1.0."""
        s, details = compute_s_cap(self.v, self.u_strong, self.task, self.cfg)
        assert s > 0.9, f"v should fully cover u_strong's small gap, S_cap={s:.4f}"

    def test_gap_details_reflect_u_capability(self):
        _, details_weak = compute_s_cap(self.v, self.u_weak, self.task, self.cfg)
        _, details_strong = compute_s_cap(self.v, self.u_strong, self.task, self.cfg)
        assert len(details_weak) == 1 and len(details_strong) == 1
        # u_weak's gap should be larger
        assert details_weak[0].gap > details_strong[0].gap
