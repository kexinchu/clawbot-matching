import json
from pathlib import Path

from LLM_Dreaming.LLM_dreaming import build_agent_persona

from run_20_tasks_evaluation import (
    build_learning_state_from_priors,
    build_requester_for_match,
    build_task,
    compute_ground_truth_M,
)
from run_v3_dreaming_ablation import (
    CountingDreamSimulator,
    analytical_matches,
    analytical_top_k,
    build_public_dream_profiles,
    evaluate_mapscore_dreaming,
    evaluate_static_method,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_PATH = REPO_ROOT / "simulator" / "20_Tasks_Testset_v3_tiered.json"


def _first_task_entry():
    return json.loads(V3_PATH.read_text(encoding="utf-8"))["tasks"][0]


def _states(task_entry):
    task = build_task(task_entry["task"])
    requester = build_requester_for_match(task_entry["proposer_profile"])
    pool = [build_learning_state_from_priors(c) for c in task_entry["candidates"]]
    true_m, optimum_id, optimum_m = compute_ground_truth_M(requester, task_entry["candidates"], task)
    return task, requester, pool, true_m, optimum_id, optimum_m


def test_dreaming_public_profiles_do_not_include_hidden_latents():
    task_entry = _first_task_entry()
    _, requester, pool, *_ = _states(task_entry)

    requester_ext, candidate_exts, public_payload = build_public_dream_profiles(
        task_entry,
        requester,
        pool,
    )

    serialized = json.dumps(public_payload, sort_keys=True)
    assert "context_latents" not in serialized
    assert "latent_interpersonal_affinity" not in serialized
    assert requester_ext.user_id == task_entry["proposer_profile"]["user_id"]
    assert candidate_exts


def test_shortlist_contains_only_analytical_top_k():
    task_entry = _first_task_entry()
    task, requester, pool, *_ = _states(task_entry)
    matches = analytical_matches(requester, pool, task, "mapscore")

    top3 = analytical_top_k(matches, 3)
    top3_ids = [m.candidate_id for m in top3]
    expected = [
        m.candidate_id
        for m in sorted(matches.values(), key=lambda m: (m.match_score, m.candidate_id), reverse=True)[:3]
    ]

    assert top3_ids == expected
    assert len(top3_ids) == 3


def test_dreaming_disabled_equals_mapscore_greedy():
    task_entry = _first_task_entry()
    task, requester, pool, true_m, optimum_id, optimum_m = _states(task_entry)

    mapscore = evaluate_static_method(
        "mapscore_greedy",
        task_entry,
        requester,
        pool,
        task,
        optimum_id,
        true_m,
        optimum_m,
    )
    disabled = evaluate_mapscore_dreaming(
        task_entry,
        requester,
        pool,
        task,
        optimum_id,
        true_m,
        optimum_m,
        top_k=3,
        seed=0,
        dreamer=CountingDreamSimulator(n_turns=1, seed=0),
        dreaming_enabled=False,
    )

    assert disabled["selected_candidate_id"] == mapscore["selected_candidate_id"]
    assert disabled["rho_last"] == mapscore["rho_last"]
    assert disabled["selected_match"] == mapscore["selected_match"]


def test_v3_task_offers_enter_dreaming_prompt():
    task_entry = _first_task_entry()
    task, requester, pool, *_ = _states(task_entry)
    _, candidate_exts, _ = build_public_dream_profiles(task_entry, requester, pool)
    candidate = next(iter(candidate_exts.values()))

    prompt = build_agent_persona(candidate, "candidate", task)

    assert "Task offers / collaboration upside visible to the candidate" in prompt
    for offer_name in task_entry["task"]["offers"]:
        assert offer_name in prompt
        break
