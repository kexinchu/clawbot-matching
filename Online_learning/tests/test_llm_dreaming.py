"""LLM Dreaming integration tests — Layer 3 dream simulation.

Runs entirely in mock mode (no ANTHROPIC_API_KEY required).
Verifies:
  1. Test fixtures use new types (UserState / CapabilityEntry / TaskRequirement)
  2. build_agent_persona renders descriptions and intensities from list iteration
  3. DreamSimulator mock returns parseable JSON
  4. PlanningLayer combines analytical + dream scores correctly
  5. Output ranking is consistent
  6. Refined candidates expose new field names (s_cap / s_need)
"""

import sys
import os
import json

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DREAM = os.path.join(_HERE, '..', '..', 'LLM_Dreaming')
if _DREAM not in sys.path:
    sys.path.append(_DREAM)

from LLM_dreaming import (
    SoftProfile, ExtendedProfile,
    create_test_candidates, build_agent_persona,
    DreamSimulator, PlanningLayer, RefinedCandidate,
    JUDGE_SYSTEM,
)
from datatypes import UserState, CapabilityEntry, NeedEntry, Task, TaskRequirement
from config import MatchConfig
from WorldModel import WorldModel


# ── 1. Fixtures use new types ─────────────────────────────────────────

def test_create_test_candidates_uses_new_types():
    alice, task, candidates = create_test_candidates()

    # Alice
    assert isinstance(alice, ExtendedProfile)
    assert isinstance(alice.profile, UserState)
    assert all(isinstance(c, CapabilityEntry) for c in alice.profile.capabilities)
    assert all(isinstance(n, NeedEntry) for n in alice.profile.needs)

    # Task
    assert isinstance(task, Task)
    assert all(isinstance(r, TaskRequirement) for r in task.requirements)
    assert len(task.requirements) > 0

    # Candidates
    assert len(candidates) == 5
    for cand in candidates:
        assert isinstance(cand, ExtendedProfile)
        assert isinstance(cand.profile, UserState)
        assert all(isinstance(c, CapabilityEntry) for c in cand.profile.capabilities)


def test_candidates_have_distinct_ids():
    _, _, candidates = create_test_candidates()
    ids = [c.user_id for c in candidates]
    assert len(set(ids)) == len(ids), "all candidate ids must be unique"
    assert {"bob", "carol", "dave", "eve", "frank"} == set(ids)


def test_capability_embeddings_have_correct_shape():
    alice, task, candidates = create_test_candidates()
    for cand in [alice] + candidates:
        for cap in cand.profile.capabilities:
            assert cap.embedding.shape == (64,)
        for need in cand.profile.needs:
            assert need.embedding.shape == (64,)
    for req in task.requirements:
        assert req.embedding.shape == (64,)


# ── 2. Persona builder iterates lists, not dicts ──────────────────────

def test_persona_includes_all_capabilities():
    alice, task, _ = create_test_candidates()
    persona = build_agent_persona(alice, "requester", task)
    for cap in alice.profile.capabilities:
        assert cap.description in persona, f"missing capability '{cap.description}' in persona"


def test_persona_includes_all_needs():
    alice, task, _ = create_test_candidates()
    persona = build_agent_persona(alice, "requester", task)
    for need in alice.profile.needs:
        assert need.description in persona


def test_persona_role_distinguishes_requester_vs_candidate():
    alice, task, candidates = create_test_candidates()
    bob = candidates[0]
    p_req  = build_agent_persona(alice, "requester", task)
    p_cand = build_agent_persona(bob,   "candidate", task)
    assert "posted the task" in p_req
    assert "being\n" in p_cand or "considered" in p_cand


def test_persona_includes_soft_profile_fields():
    alice, task, _ = create_test_candidates()
    persona = build_agent_persona(alice, "requester", task)
    assert alice.soft.availability in persona
    assert alice.soft.timezone in persona
    assert alice.soft.collab_style in persona


# ── 3. DreamSimulator mock mode ───────────────────────────────────────

def test_simulator_runs_in_mock_mode_without_key():
    """Without ANTHROPIC_API_KEY the simulator must not attempt network calls.

    API_KEY is captured at module-import time, so we don't reload (would create
    duplicate classes); instead we assert this matches the env at import time.
    """
    import LLM_dreaming as _ld
    expected_mock = not bool(os.environ.get("ANTHROPIC_API_KEY", ""))
    sim = _ld.DreamSimulator(n_turns=2)
    assert sim.mock_mode is expected_mock


def test_simulator_judge_returns_valid_json():
    sim = DreamSimulator(n_turns=2)
    result = sim._mock_judge([{"role": "user", "content": "candidate available 40h fully dedicated, eager to learn"}])
    parsed = json.loads(result)
    assert "time_energy" in parsed
    assert "priority_alignment" in parsed
    assert "collab_style" in parsed
    assert "personality_fit" in parsed
    assert "overall_compatibility" in parsed
    assert "recommendation" in parsed
    assert 0.0 <= parsed["overall_compatibility"] <= 1.0
    assert parsed["recommendation"] in {"strong_match", "good_match", "risky_match", "poor_match"}


def test_simulator_judge_high_score_for_strong_signals():
    sim = DreamSimulator(n_turns=2)
    high_text = "fully dedicated 40h, eager to learn, pair programming, neurips publication motivation"
    result = sim._mock_judge([{"role": "user", "content": high_text}])
    parsed = json.loads(result)
    assert parsed["overall_compatibility"] >= 0.6


def test_simulator_judge_low_score_for_weak_signals():
    sim = DreamSimulator(n_turns=2)
    low_text = "only 5h stretched thin, async only, minimal time, other projects"
    result = sim._mock_judge([{"role": "user", "content": low_text}])
    parsed = json.loads(result)
    assert parsed["overall_compatibility"] <= 0.5


def test_simulate_conversation_full_loop():
    """End-to-end: persona → conversation → judge → parsed scores."""
    alice, task, candidates = create_test_candidates()
    bob = candidates[0]
    sim = DreamSimulator(n_turns=2)
    result = sim.simulate_conversation(alice, bob, task)
    assert result["candidate_id"] == "bob"
    assert "transcript" in result
    assert "compatibility" in result
    assert len(result["transcript"]) >= 2
    compat = result["compatibility"]
    assert 0.0 <= compat["overall_compatibility"] <= 1.0


# ── 4. PlanningLayer pipeline ────────────────────────────────────────

@pytest.fixture
def planner_setup():
    alice, task, candidates = create_test_candidates()
    wm = WorldModel(config=MatchConfig(embedding_dim=64))
    dreamer = DreamSimulator(n_turns=2)
    planner = PlanningLayer(wm, dreamer, top_k=3, top_n=2,
                             analytical_weight=0.5, dream_weight=0.5)
    return planner, alice, task, candidates


def test_planning_layer_returns_top_n(planner_setup):
    planner, alice, task, candidates = planner_setup
    top = planner.run(alice, candidates, task)
    assert len(top) == 2


def test_planning_layer_output_types(planner_setup):
    planner, alice, task, candidates = planner_setup
    top = planner.run(alice, candidates, task)
    for r in top:
        assert isinstance(r, RefinedCandidate)
        # New lowercase field names from migration
        assert hasattr(r, "s_cap")
        assert hasattr(r, "s_need")
        assert not hasattr(r, "S_cap"), "old uppercase field should be removed"


def test_combined_score_is_weighted_sum(planner_setup):
    """combined_score = w_analytical · analytical + w_dream · dream."""
    planner, alice, task, candidates = planner_setup
    top = planner.run(alice, candidates, task)
    for r in top:
        expected = (planner.w_analytical * r.analytical_score
                    + planner.w_dream * r.dream_score)
        assert abs(r.combined_score - expected) < 1e-6


def test_top_n_sorted_descending_by_combined(planner_setup):
    planner, alice, task, candidates = planner_setup
    top = planner.run(alice, candidates, task)
    scores = [r.combined_score for r in top]
    assert scores == sorted(scores, reverse=True), "top-N must be sorted by combined score"


def test_all_recommendations_are_valid_categories(planner_setup):
    planner, alice, task, candidates = planner_setup
    top = planner.run(alice, candidates, task)
    valid = {"strong_match", "good_match", "risky_match", "poor_match"}
    for r in top:
        assert r.recommendation in valid


# ── 5. Cross-layer field-name consistency ─────────────────────────────

def test_no_legacy_field_references_in_planning(planner_setup):
    """RefinedCandidate must not expose old M/S_cap/S_need names."""
    planner, alice, task, candidates = planner_setup
    top = planner.run(alice, candidates, task)
    for r in top:
        assert not hasattr(r, "M"),     "MatchResult.M is gone (now match_score)"
        assert not hasattr(r, "S_cap"), "MatchResult.S_cap is gone (now s_cap)"
        assert not hasattr(r, "S_need"),"MatchResult.S_need is gone (now s_need)"
