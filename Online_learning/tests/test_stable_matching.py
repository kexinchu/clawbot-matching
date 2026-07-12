"""Unit tests for batch stable matching invariants."""

from __future__ import annotations

import copy
import random
from unittest.mock import patch

import pytest

from Experiments.stable_matching.batch_generator_utils import (
    handcrafted_competition_batch,
    verify_shared_profiles,
)
from Experiments.stable_matching.deferred_acceptance import (
    count_blocking_pairs,
    deferred_acceptance,
    is_stable_matching,
)
from Experiments.stable_matching.evaluation import (
    build_oracle_utilities,
    oracle_max_weight_assignment,
)
from Experiments.stable_matching.preferences import (
    METHOD_CW,
    METHOD_GS,
    FORBIDDEN_PREF_KEYS,
    build_coweaver_preferences,
    build_gs_preferences,
    build_preferences,
    collect_feasible_edges,
    discrete_weighted_skill_coverage,
)
from simulator.generate_batch_matching_testset import build_batch_testset


@pytest.fixture(scope="module")
def small_testset():
    return build_batch_testset(
        seed=42,
        num_batches=2,
        tasks_per_batch=5,
        candidates_per_batch=7,
        candidate_capacity=1,
    )


def test_shared_candidate_profile_identical_across_tasks(small_testset):
    batch = small_testset["batches"][0]
    errors = verify_shared_profiles(batch)
    assert errors == []

    # Same candidate object fields across every task pair entry.
    by_cand = {}
    for te in batch["tasks"]:
        for pe in te["pair_entries"]:
            by_cand.setdefault(pe["candidate_id"], []).append(pe["candidate_profile"])
    for cid, profiles in by_cand.items():
        assert len(profiles) == len(batch["tasks"])
        for p in profiles[1:]:
            assert p == profiles[0], f"{cid} profile drifted across tasks"


def test_capacity_one_never_double_assigns():
    batch = handcrafted_competition_batch(candidate_capacity=1)
    edges = collect_feasible_edges(batch)
    prefs = build_gs_preferences(batch, edges, tie_break_seed=0)
    da = deferred_acceptance(
        prefs.task_ids,
        prefs.candidate_ids,
        prefs.task_prefs,
        prefs.candidate_prefs,
        candidate_capacity=1,
    )
    assigned = [cid for cid in da.matching.values() if cid is not None]
    assert len(assigned) == len(set(assigned))
    for cid, tasks in da.candidate_loads.items():
        assert len(tasks) <= 1


def test_capacity_k_respected():
    batch = handcrafted_competition_batch(candidate_capacity=2)
    edges = collect_feasible_edges(batch)
    prefs = build_gs_preferences(batch, edges, tie_break_seed=0)
    da = deferred_acceptance(
        prefs.task_ids,
        prefs.candidate_ids,
        prefs.task_prefs,
        prefs.candidate_prefs,
        candidate_capacity=2,
    )
    for cid, tasks in da.candidate_loads.items():
        assert len(tasks) <= 2


def test_da_stable_under_own_preferences():
    batch = handcrafted_competition_batch()
    edges = collect_feasible_edges(batch)
    for method in (METHOD_GS, METHOD_CW):
        prefs = build_preferences(method, batch, feasible_edges=edges, tie_break_seed=7)
        da = deferred_acceptance(
            prefs.task_ids,
            prefs.candidate_ids,
            prefs.task_prefs,
            prefs.candidate_prefs,
            candidate_capacity=1,
            tie_break_seed=7,
        )
        assert is_stable_matching(
            da.matching,
            prefs.task_prefs,
            prefs.candidate_prefs,
            candidate_capacity=1,
        )
        assert count_blocking_pairs(
            da.matching,
            prefs.task_prefs,
            prefs.candidate_prefs,
            candidate_capacity=1,
        ) == 0


def test_infeasible_pairs_never_matched(small_testset):
    batch = small_testset["batches"][0]
    edges = collect_feasible_edges(batch)
    infeasible = set()
    for te in batch["tasks"]:
        tid = te["task"]["task_id"]
        for pe in te["pair_entries"]:
            if not pe.get("public_feasible", True):
                infeasible.add((tid, pe["candidate_id"]))

    for method in (METHOD_GS, METHOD_CW):
        prefs = build_preferences(method, batch, feasible_edges=edges, tie_break_seed=0)
        da = deferred_acceptance(
            prefs.task_ids,
            prefs.candidate_ids,
            prefs.task_prefs,
            prefs.candidate_prefs,
            candidate_capacity=1,
        )
        for tid, cid in da.matching.items():
            if cid is None:
                continue
            assert (tid, cid) not in infeasible
            assert (tid, cid) in edges


def test_gs_preferences_do_not_call_mapscore():
    batch = handcrafted_competition_batch()
    edges = collect_feasible_edges(batch)
    with patch(
        "Experiments.stable_matching.preferences.WorldModel.compute_match",
        side_effect=AssertionError("MapScore must not be called for GS"),
    ):
        prefs = build_gs_preferences(batch, edges, tie_break_seed=0)
    assert prefs.diagnostics["uses_mapscore"] is False
    # Coverage should still be populated.
    assert any(prefs.task_scores[tid] for tid in prefs.task_ids)


def test_coweaver_preferences_do_not_read_hidden_latents():
    batch = handcrafted_competition_batch()
    # Inject sentinel hidden values; preference builder must ignore them.
    for te in batch["tasks"]:
        for pe in te["pair_entries"]:
            pe["context_latents"]["latent_interpersonal_affinity"] = 0.99
            pe["oracle_total_reward"] = 999.0

    edges = collect_feasible_edges(batch)
    # build_coweaver uses _public_pair_view which strips latents; also assert
    # forbidden keys are not traversed when scoring public views.
    prefs = build_coweaver_preferences(batch, edges, tie_break_seed=0)
    assert prefs.diagnostics["uses_hidden_latents"] is False
    assert prefs.diagnostics["uses_ucb"] is False
    assert prefs.diagnostics["uses_dreaming"] is False

    # Ensure forbidden keys are documented.
    assert "context_latents" in FORBIDDEN_PREF_KEYS
    assert "total_reward" in FORBIDDEN_PREF_KEYS


def test_both_methods_share_feasible_edges_and_da_allocator(small_testset):
    batch = small_testset["batches"][0]
    edges = collect_feasible_edges(batch)
    gs = build_preferences(METHOD_GS, batch, feasible_edges=edges, tie_break_seed=1)
    cw = build_preferences(METHOD_CW, batch, feasible_edges=edges, tie_break_seed=1)
    assert gs.feasible_edges == cw.feasible_edges == edges

    # Same allocator function object is used by both (imported once).
    da_gs = deferred_acceptance(
        gs.task_ids, gs.candidate_ids, gs.task_prefs, gs.candidate_prefs,
        candidate_capacity=1, tie_break_seed=1,
    )
    da_cw = deferred_acceptance(
        cw.task_ids, cw.candidate_ids, cw.task_prefs, cw.candidate_prefs,
        candidate_capacity=1, tie_break_seed=1,
    )
    assert isinstance(da_gs.matching, dict) and isinstance(da_cw.matching, dict)


def test_handcrafted_batch_creates_real_capacity_competition():
    batch = handcrafted_competition_batch(candidate_capacity=1)
    edges = collect_feasible_edges(batch)
    prefs = build_gs_preferences(batch, edges, tie_break_seed=0)

    # Both tasks should rank the star candidate first (or at least top).
    for tid in prefs.task_ids:
        assert prefs.task_prefs[tid][0] == "cand_star"

    da = deferred_acceptance(
        prefs.task_ids,
        prefs.candidate_ids,
        prefs.task_prefs,
        prefs.candidate_prefs,
        candidate_capacity=1,
    )
    holders = [cid for cid in da.matching.values() if cid == "cand_star"]
    assert len(holders) == 1, "star must be assigned to exactly one task under capacity=1"
    # The other task should get backup (or remain unmatched), not the star.
    assert sum(1 for c in da.matching.values() if c == "cand_star") == 1


def test_oracle_reads_hidden_outcome_only_and_not_used_in_selection(small_testset):
    batch = small_testset["batches"][0]
    utilities = build_oracle_utilities(batch, seed=42)
    assert utilities, "oracle utilities should be non-empty"

    task_ids = [t["task"]["task_id"] for t in batch["tasks"]]
    cand_ids = [c["candidate_profile"]["user_id"] for c in batch["shared_candidates"]]
    matching, obj = oracle_max_weight_assignment(
        task_ids, cand_ids, utilities, candidate_capacity=1
    )
    assert obj == pytest.approx(
        sum(
            utilities[(tid, cid)]
            for tid, cid in matching.items()
            if cid is not None
        )
    )

    # GS / CoWeaver selection must not consult utilities.
    edges = collect_feasible_edges(batch)
    with patch(
        "Experiments.stable_matching.evaluation.simulate_pair_outcome",
        side_effect=AssertionError("selection must not call oracle outcome"),
    ):
        # Preference construction should not call simulate_pair_outcome.
        build_gs_preferences(batch, edges, tie_break_seed=0)
        build_coweaver_preferences(batch, edges, tie_break_seed=0)


def test_identical_seed_is_reproducible():
    a = build_batch_testset(seed=123, num_batches=2, tasks_per_batch=3, candidates_per_batch=4)
    b = build_batch_testset(seed=123, num_batches=2, tasks_per_batch=3, candidates_per_batch=4)
    assert a == b

    batch = a["batches"][0]
    edges = collect_feasible_edges(batch)
    prefs1 = build_gs_preferences(batch, edges, tie_break_seed=0)
    prefs2 = build_gs_preferences(batch, edges, tie_break_seed=0)
    assert prefs1.task_prefs == prefs2.task_prefs
    assert prefs1.candidate_prefs == prefs2.candidate_prefs

    da1 = deferred_acceptance(
        prefs1.task_ids, prefs1.candidate_ids, prefs1.task_prefs, prefs1.candidate_prefs,
        candidate_capacity=1, tie_break_seed=0,
    )
    da2 = deferred_acceptance(
        prefs2.task_ids, prefs2.candidate_ids, prefs2.task_prefs, prefs2.candidate_prefs,
        candidate_capacity=1, tie_break_seed=0,
    )
    assert da1.matching == da2.matching
    assert da1.trace == da2.trace


def test_discrete_coverage_matches_expected():
    required = {"python": 0.8, "statistics": 0.5}
    assert discrete_weighted_skill_coverage(required, {"python": 0.8, "statistics": 0.5}) == 1.0
    assert discrete_weighted_skill_coverage(required, {"python": 0.7, "statistics": 0.5}) == pytest.approx(
        0.5 / 1.3
    )
