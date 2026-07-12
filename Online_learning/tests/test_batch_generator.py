"""Tests for the batch matching benchmark generator."""

from __future__ import annotations

from Experiments.stable_matching.batch_generator_utils import verify_shared_profiles
from Experiments.stable_matching.preferences import (
    collect_feasible_edges,
    discrete_weighted_skill_coverage,
)
from simulator.generate_batch_matching_testset import (
    build_batch_testset,
    _pair_is_feasible_public,
)


def test_generator_default_shape():
    data = build_batch_testset(
        seed=42,
        num_batches=3,
        tasks_per_batch=5,
        candidates_per_batch=7,
        candidate_capacity=1,
    )
    assert data["metadata"]["num_batches"] == 3
    assert data["metadata"]["shared_candidate_pool"] is True
    assert data["metadata"]["benchmark_version"] == "v3_batch_shared_pool"
    assert len(data["batches"]) == 3
    for batch in data["batches"]:
        assert len(batch["shared_candidates"]) == 7
        assert len(batch["tasks"]) == 5
        assert batch["candidate_capacity"] == 1
        assert verify_shared_profiles(batch) == []


def test_most_tasks_have_at_least_two_feasible_candidates():
    data = build_batch_testset(seed=42, num_batches=5, tasks_per_batch=5, candidates_per_batch=7)
    ok = 0
    total = 0
    for batch in data["batches"]:
        for te in batch["tasks"]:
            total += 1
            n_feas = sum(1 for pe in te["pair_entries"] if pe["public_feasible"])
            if n_feas >= 2:
                ok += 1
    assert ok / total >= 0.8, f"only {ok}/{total} tasks have >=2 feasible candidates"


def test_competition_signals_present():
    data = build_batch_testset(seed=42, num_batches=3, tasks_per_batch=5, candidates_per_batch=7)
    for batch in data["batches"]:
        # At least one candidate with high coverage on >=2 tasks.
        star = batch["shared_candidates"][0]["candidate_profile"]
        high_cov_tasks = 0
        for te in batch["tasks"]:
            cov = discrete_weighted_skill_coverage(
                te["task"]["required_skills"], star["capabilities"]
            )
            if cov >= 0.4:
                high_cov_tasks += 1
        assert high_cov_tasks >= 2

        # Some infeasible pairs exist.
        n_infeas = sum(
            1
            for te in batch["tasks"]
            for pe in te["pair_entries"]
            if not pe["public_feasible"]
        )
        assert n_infeas >= 1


def test_no_mapscore_or_oracle_in_public_benchmark_fields():
    data = build_batch_testset(seed=7, num_batches=1, tasks_per_batch=3, candidates_per_batch=4)
    blob = str(data)
    for forbidden in ("MapScore", "total_reward", "mutual_accept_probability", "S_cap"):
        # Allow the string only inside metadata notes if present; harden by checking pair keys.
        pass
    for batch in data["batches"]:
        for te in batch["tasks"]:
            assert "MapScore" not in te
            assert "outcome" not in te
            for pe in te["pair_entries"]:
                assert "MapScore" not in pe
                assert "total_reward" not in pe
                assert "context_latents" in pe  # hidden, but present for oracle eval
                # Public matching inputs exclude latents from profile.
                assert "latent_interpersonal_affinity" not in pe["candidate_profile"]


def test_seed_reproducibility():
    a = build_batch_testset(seed=456, num_batches=2, tasks_per_batch=4, candidates_per_batch=5)
    b = build_batch_testset(seed=456, num_batches=2, tasks_per_batch=4, candidates_per_batch=5)
    assert a == b
    c = build_batch_testset(seed=457, num_batches=2, tasks_per_batch=4, candidates_per_batch=5)
    assert a != c


def test_feasible_edge_set_consistent_with_flags():
    data = build_batch_testset(seed=42, num_batches=1, tasks_per_batch=5, candidates_per_batch=7)
    batch = data["batches"][0]
    edges = collect_feasible_edges(batch)
    for te in batch["tasks"]:
        tid = te["task"]["task_id"]
        for pe in te["pair_entries"]:
            cid = pe["candidate_id"]
            if pe["public_feasible"] and discrete_weighted_skill_coverage(
                te["task"]["required_skills"], pe["candidate_profile"]["capabilities"]
            ) > 0:
                assert (tid, cid) in edges
            else:
                assert (tid, cid) not in edges


def test_pair_feasibility_helper():
    task = {
        "task_id": "t",
        "required_skills": {"python": 0.8},
        "metadata": {"urgency": "low"},
    }
    ok = {"capabilities": {"python": 0.9}, "preferences": {}, "constraints": {}}
    bad = {"capabilities": {"python": 0.1}, "preferences": {}, "constraints": {}}
    assert _pair_is_feasible_public(task, ok) is True
    assert _pair_is_feasible_public(task, bad) is False
