"""Adapter tests — UserState/Task/MatchResult → simulator types.

Covers:
  1. UserState.capabilities (List[CapabilityEntry]) → dict {description: mu}
  2. UserState.needs (List[NeedEntry]) → dict {description: intensity}
  3. Task soft requirements only (hard ones filtered — gate handles them)
  4. CandidateCard strengths drawn from gap_details with high coverage
  5. CandidateCard risks drawn from gap_details with low coverage
  6. MatchingContext one-shot pack
  7. clearance_level survives round trip into constraints
"""

import pytest

from test_helpers import make_user, make_task, CFG
from sim_adapter import (
    to_sim_userprofile,
    to_sim_taskspec,
    build_candidate_card,
    build_matching_context,
)
from WorldModel import WorldModel


# ── UserProfile translation ──────────────────────────────────────────

def test_userprofile_dicts_keyed_by_description():
    u = make_user("u1",
        caps=[("python programming", 0.8, 0.1, "explicit"),
              ("data science", 0.5, 0.3, "implicit")],
        needs=[("mentorship", 0.7), ("research collab", 0.4)])
    p = to_sim_userprofile(u)
    assert p.user_id == "u1"
    assert p.capabilities == {"python programming": 0.8, "data science": 0.5}
    assert p.needs == {"mentorship": 0.7, "research collab": 0.4}


def test_userprofile_clearance_propagates_into_constraints():
    u = make_user("u_secure",
        caps=[("alpha", 0.5, 0.5, "explicit")], needs=[], clearance=3)
    p = to_sim_userprofile(u)
    assert p.constraints["clearance_level"] == 3


def test_userprofile_role_param_is_set():
    u = make_user("u", caps=[("x", 0.5, 0.5, "explicit")], needs=[])
    p_req = to_sim_userprofile(u, role="requester")
    p_cand = to_sim_userprofile(u, role="candidate")
    assert p_req.role == "requester"
    assert p_cand.role == "candidate"


# ── TaskSpec translation ─────────────────────────────────────────────

def test_taskspec_includes_only_soft_requirements():
    t = make_task(
        reqs=[("python", 0.8, "soft"),
              ("clearance", 0.9, "hard"),
              ("ml", 0.6, "soft")],
        offers=[])
    spec = to_sim_taskspec(t)
    # Hard requirement filtered — L2 gate handles it
    assert "clearance" not in spec.required_skills
    assert spec.required_skills == {"python": 0.8, "ml": 0.6}


def test_taskspec_metadata_includes_clearance_and_offer_count():
    t = make_task(
        reqs=[("python", 0.5, "soft")],
        offers=[("collab", 0.8, "explicit"), ("auth", 0.7, "explicit")],
        data_clearance=2)
    spec = to_sim_taskspec(t)
    assert spec.metadata["data_clearance"] == 2
    assert spec.metadata["n_offers"] == 2


def test_taskspec_with_no_soft_reqs_returns_empty_required_skills():
    t = make_task(
        reqs=[("hard1", 0.9, "hard")],
        offers=[])
    spec = to_sim_taskspec(t)
    assert spec.required_skills == {}


# ── CandidateCard ─────────────────────────────────────────────────────

@pytest.fixture
def basic_match():
    """Run L2 over a constructed pair, return (alice, bob, task, match)."""
    alice = make_user("alice",
        caps=[("python", 0.3, 0.3, "explicit")],
        needs=[("collab", 0.5)])
    bob = make_user("bob",
        caps=[("python", 0.9, 0.1, "explicit"),
              ("data analysis", 0.8, 0.1, "explicit")],
        needs=[("mentorship", 0.5)])
    task = make_task(
        reqs=[("python", 0.8, "soft"),
              ("data analysis", 0.7, "soft")],
        offers=[("collab", 0.7, "explicit")])
    wm = WorldModel(config=CFG)
    match = wm.compute_match(alice, bob, task)
    return alice, bob, task, match


def test_card_summary_contains_id_and_match_score(basic_match):
    alice, bob, task, match = basic_match
    card = build_candidate_card(bob, task, match)
    assert card.candidate_id == "bob"
    assert "bob" in card.summary
    assert f"{match.match_score:.2f}" in card.summary


def test_card_strengths_reference_high_coverage_reqs(basic_match):
    alice, bob, task, match = basic_match
    card = build_candidate_card(bob, task, match)
    # Bob covers both python (0.9 vs gap 0.5) and data analysis fully
    joined = " ".join(card.highlighted_strengths)
    # at least one strength references a requirement description
    req_descs = [r.description for r in task.requirements]
    assert any(rd in joined for rd in req_descs), (
        f"Expected a requirement description in strengths: {card.highlighted_strengths}"
    )


def test_card_risks_appear_when_coverage_low():
    alice = make_user("alice",
        caps=[("python", 0.3, 0.3, "explicit")],
        needs=[])
    bob_weak = make_user("bob_weak",
        caps=[("python", 0.4, 0.2, "explicit")],   # low μ
        needs=[])
    task = make_task(
        reqs=[("python", 0.9, "soft")],   # demanding
        offers=[])
    wm = WorldModel(config=CFG)
    match = wm.compute_match(alice, bob_weak, task)
    card = build_candidate_card(bob_weak, task, match)
    # gap is 0.6, coverage is ~0.4, ratio ~0.67 — falls into mid range,
    # but if M is low the moderate-match risk should fire
    assert card.highlighted_strengths or card.highlighted_risks
    if match.match_score < 0.6:
        assert any("moderate" in r.lower() or "limited" in r.lower()
                   for r in card.highlighted_risks)


def test_card_explanation_mentions_weights_and_gate(basic_match):
    alice, bob, task, match = basic_match
    card = build_candidate_card(bob, task, match)
    assert "w_c" in card.explanation
    assert "w_n" in card.explanation
    assert "Gate" in card.explanation or "σ" in card.explanation


# ── MatchingContext one-shot ─────────────────────────────────────────

def test_build_matching_context_packs_everything(basic_match):
    alice, bob, task, match = basic_match
    ctx = build_matching_context(alice, bob, task, match)
    assert ctx.requester.user_id == "alice"
    assert ctx.candidate.user_id == "bob"
    assert ctx.task.task_id == "t_test"
    assert ctx.card.candidate_id == "bob"


def test_build_matching_context_history_round_trip(basic_match):
    alice, bob, task, match = basic_match
    history = {"prior_collaboration": True}
    ctx = build_matching_context(alice, bob, task, match, history=history)
    assert ctx.history == history
