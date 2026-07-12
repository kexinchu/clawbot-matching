"""Unit tests for PeopleJoin-Reactive baseline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from Experiments.peoplejoin.bm25_retriever import BM25CandidateRetriever, DirectoryEntry
from Experiments.peoplejoin.candidate_responder import CandidateResponder, public_candidate_profile
from Experiments.peoplejoin.isolation import assert_peoplejoin_public_inputs
from Experiments.peoplejoin.llm_client import PeopleJoinLLMClient
from Experiments.peoplejoin.reactive_controller import (
    PeopleJoinBudget,
    PeopleJoinReactiveController,
    run_peoplejoin_reactive,
)
from run_20_tasks_evaluation import (
    build_learning_state_from_priors,
    build_requester_for_match,
    build_task,
    compute_ground_truth_M,
)
from run_v3_dreaming_ablation import analytical_matches, compute_bilateral_outcome
from run_v3_peoplejoin_baseline import evaluate_peoplejoin, evaluate_task


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_PATH = REPO_ROOT / "simulator" / "20_Tasks_Testset_v3_tiered.json"


def _load_tasks():
    return json.loads(V3_PATH.read_text(encoding="utf-8"))["tasks"]


def _first_task():
    return _load_tasks()[0]


def test_bm25_returns_sorted_candidate_ids():
    entries = [
        DirectoryEntry(
            candidate_id="c_python",
            role="engineer",
            summary="python systems engineer",
            highlighted_strengths=["python", "systems"],
            highlighted_risks=["timezone"],
        ),
        DirectoryEntry(
            candidate_id="c_design",
            role="designer",
            summary="product design and UX",
            highlighted_strengths=["product_design", "ux"],
            highlighted_risks=["workload"],
        ),
        DirectoryEntry(
            candidate_id="c_python_ml",
            role="scientist",
            summary="python ml systems and evaluation",
            highlighted_strengths=["python", "ml_systems", "evaluation"],
            highlighted_risks=[],
        ),
    ]
    retriever = BM25CandidateRetriever(entries)
    hits = retriever.search("python ml_systems evaluation", top_k=5)
    ids = [h.candidate_id for h in hits]
    scores = [h.score for h in hits]

    assert set(ids) <= {"c_python", "c_design", "c_python_ml"}
    assert ids[0] in {"c_python_ml", "c_python"}
    assert scores == sorted(scores, reverse=True)
    assert hits[0].rank == 1
    assert all(h.candidate_id for h in hits)


def test_candidate_responder_prompt_excludes_hidden_latents():
    task_entry = _first_task()
    cand = task_entry["candidates"][0]
    # Ensure latents exist in the raw entry but not in responder prompt.
    assert "context_latents" in cand
    llm = PeopleJoinLLMClient(mode="mock", seed=0)
    responder = CandidateResponder.from_task_entry(task_entry, llm)
    cid = cand["candidate_profile"]["user_id"]
    prompt = responder.build_system_prompt(cid)
    assert_peoplejoin_public_inputs(prompt)
    for bad in (
        "context_latents",
        "latent_interpersonal_affinity",
        "latent_risk_tolerance",
        "latent_opportunity_bias",
    ):
        assert bad not in prompt
    # Profile JSON must be present; instructional bans may mention MapScore in prose.
    assert '"capabilities"' in prompt or "capabilities" in prompt
    answer = responder.answer(cid, "What are your top capabilities and availability?")
    assert len(answer) > 10


def test_controller_cannot_select_out_of_pool_candidate():
    task_entry = _first_task()
    llm = PeopleJoinLLMClient(mode="mock", seed=0)
    controller = PeopleJoinReactiveController(
        task_entry=task_entry,
        llm=llm,
        budget=PeopleJoinBudget(max_search=1, max_ask=1, max_actions=3),
    )
    pool = {c["candidate_profile"]["user_id"] for c in task_entry["candidates"]}

    # Force an invalid finish then continue; final selection must be in pool.
    obs, err = controller._execute(
        {
            "action": "finish",
            "selected_candidate_id": "not_a_real_candidate",
            "rationale": "bad",
        }
    )
    assert err == "finish_candidate_not_in_pool"
    assert "error" in obs

    result = controller.run()
    assert result.selected_candidate_id in pool


def test_controller_respects_search_ask_action_budgets():
    task_entry = _first_task()
    budget = PeopleJoinBudget(max_search=2, max_ask=2, max_actions=5, search_top_k=5)
    llm = PeopleJoinLLMClient(mode="mock", seed=1)
    result = run_peoplejoin_reactive(task_entry, llm=llm, budget=budget, seed=1, mode="mock")
    cost = result.interaction_cost
    assert cost["search_calls"] <= budget.max_search
    assert cost["candidate_messages"] <= budget.max_ask
    # finish counts as an action; total executed actions <= max_actions
    action_names = [step["action"]["action"] for step in result.trace]
    assert len(action_names) <= budget.max_actions + 2  # parse-only retries may add steps rarely
    assert sum(1 for a in action_names if a == "search_relevant_people") <= budget.max_search
    assert sum(1 for a in action_names if a == "ask_candidate") <= budget.max_ask


def test_malformed_action_retries_or_falls_back():
    task_entry = _first_task()
    llm = PeopleJoinLLMClient(mode="mock", seed=0)
    controller = PeopleJoinReactiveController(
        task_entry=task_entry,
        llm=llm,
        budget=PeopleJoinBudget(max_search=2, max_ask=2, max_actions=4),
    )

    call_count = {"n": 0}
    real_complete = llm.complete

    def flaky_complete(system, messages, mock_fn=None):
        call_count["n"] += 1
        # First controller call returns garbage; subsequent use real mock policy.
        if call_count["n"] == 1:
            llm.usage.api_calls += 1
            llm.usage.input_tokens += 10
            llm.usage.output_tokens += 5
            return "this is not json at all"
        return real_complete(system, messages, mock_fn=mock_fn)

    with mock.patch.object(llm, "complete", side_effect=flaky_complete):
        result = controller.run()

    pool = {c["candidate_profile"]["user_id"] for c in task_entry["candidates"]}
    assert result.selected_candidate_id in pool
    assert result.retrieval_diagnostics["malformed_json_count"] >= 1 or result.errors


def test_mock_mode_runs_at_least_two_tasks():
    tasks = _load_tasks()[:2]
    budget = PeopleJoinBudget(max_search=2, max_ask=3, max_actions=6)
    for task_entry in tasks:
        result = run_peoplejoin_reactive(
            task_entry,
            budget=budget,
            seed=0,
            mode="mock",
        )
        pool = {c["candidate_profile"]["user_id"] for c in task_entry["candidates"]}
        assert result.selected_candidate_id in pool
        assert result.interaction_cost["search_calls"] >= 1
        assert result.trace


def test_same_seed_mock_mode_is_reproducible():
    task_entry = _first_task()
    budget = PeopleJoinBudget(max_search=2, max_ask=3, max_actions=6)
    r1 = run_peoplejoin_reactive(task_entry, budget=budget, seed=7, mode="mock")
    r2 = run_peoplejoin_reactive(task_entry, budget=budget, seed=7, mode="mock")
    assert r1.selected_candidate_id == r2.selected_candidate_id
    assert [s["action"] for s in r1.trace] == [s["action"] for s in r2.trace]


def test_peoplejoin_does_not_use_mapscore_for_selection():
    task_entry = _first_task()
    budget = PeopleJoinBudget(max_search=2, max_ask=2, max_actions=5)

    with mock.patch("run_v3_peoplejoin_baseline.analytical_matches") as mocked_matches:
        # If PeopleJoin selection called MapScore, this would raise during run.
        mocked_matches.side_effect = AssertionError("MapScore must not be used for selection")
        # Selection path itself should not call analytical_matches.
        result = run_peoplejoin_reactive(task_entry, budget=budget, seed=0, mode="mock")
        assert result.method == "peoplejoin_reactive"
        mocked_matches.assert_not_called()

    # Explicitly ensure controller module never imports WorldModel scoring for choice.
    import Experiments.peoplejoin.reactive_controller as rc

    source = Path(rc.__file__).read_text(encoding="utf-8")
    assert "WorldModel" not in source
    assert "compute_match" not in source
    assert "MapScore" not in source or "do NOT have MapScore" in source


def test_posthoc_outcome_evaluation_is_isolated_from_controller():
    task_entry = _first_task()
    task = build_task(task_entry["task"])
    requester = build_requester_for_match(task_entry["proposer_profile"])
    pool = [build_learning_state_from_priors(c) for c in task_entry["candidates"]]
    m_true, optimum_id, m_opt = compute_ground_truth_M(
        requester, task_entry["candidates"], task
    )

    # Run controller without outcome simulation.
    pj = run_peoplejoin_reactive(
        task_entry,
        budget=PeopleJoinBudget(max_search=1, max_ask=2, max_actions=4),
        seed=0,
        mode="mock",
    )
    assert "outcome" not in json.dumps(pj.to_dict())
    assert "context_latents" not in json.dumps(pj.to_dict())

    # Post-hoc oracle evaluation happens separately.
    outcome, _ = compute_bilateral_outcome(task_entry, pj.selected_candidate_id)
    assert "mutual_accept_probability" in outcome
    assert "completion_probability" in outcome

    # Full evaluate_peoplejoin wires post-hoc diagnostics after selection.
    evaluated = evaluate_peoplejoin(
        task_entry,
        requester,
        pool,
        task,
        m_true,
        m_opt,
        seed=0,
        mode="mock",
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        budget=PeopleJoinBudget(max_search=1, max_ask=2, max_actions=4),
        oracle_best_id=optimum_id,
    )
    assert evaluated["selected_candidate_id"] == pj.selected_candidate_id
    assert "outcome" in evaluated
    assert "selected_match" in evaluated  # diagnostic only


def test_assert_peoplejoin_public_inputs_catches_leaks():
    with pytest.raises(AssertionError):
        assert_peoplejoin_public_inputs({"foo": 1, "latent_risk_tolerance": 0.2})
    with pytest.raises(AssertionError):
        assert_peoplejoin_public_inputs('{"MapScore": 0.42, "candidate_id": "x"}')
    # Instructional prose may mention forbidden concepts without field leakage.
    assert_peoplejoin_public_inputs(
        "You do NOT have MapScore, S_cap, oracle outcomes, or rewards."
    )
    assert_peoplejoin_public_inputs({"candidate_id": "x", "summary": "ok"})


def test_evaluate_task_smoke_mock():
    task_entry = _first_task()
    result = evaluate_task(
        task_entry,
        seeds=[0],
        mode="mock",
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        budget=PeopleJoinBudget(max_search=1, max_ask=2, max_actions=4),
    )
    assert "lappas_coverage" in result["conditions"]
    assert "mapscore_greedy" in result["conditions"]
    assert "peoplejoin_reactive" in result["conditions"]
    assert len(result["conditions"]["peoplejoin_reactive"]) == 1
