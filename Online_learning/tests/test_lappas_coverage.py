"""Tests for the Lappas-style Skill Coverage baseline."""

from datatypes import CapabilityEntry, NeedEntry, Task, TaskOffer, TaskRequirement, UserState

from run_20_tasks_evaluation import (
    ENC,
    evaluate_task,
    lappas_style_skill_coverage_score,
    run_mapscore_greedy,
    run_lappas_coverage,
)


def _cap(skill: str, level: float) -> CapabilityEntry:
    return CapabilityEntry(ENC(skill), mu=level, sigma=0.0, description=skill)


def _need(text: str, intensity: float) -> NeedEntry:
    return NeedEntry(ENC(text), intensity=intensity, description=text)


def _req(skill: str, level: float) -> TaskRequirement:
    return TaskRequirement(ENC(skill), level=level, constraint_type="soft", description=skill)


def test_lappas_coverage_ranks_by_discrete_skill_coverage():
    task = Task(
        task_id="coverage_sort",
        goal="Rank by covered required skills.",
        requirements=[_req("python", 0.8), _req("statistics", 0.6)],
    )
    weak = UserState("weak", capabilities=[_cap("python", 0.7), _cap("statistics", 0.6)])
    strong = UserState("strong", capabilities=[_cap("python", 0.8), _cap("statistics", 0.6)])

    _, selected, _, scores = run_lappas_coverage([weak, strong], task)

    assert selected == "strong"
    assert scores["strong"] > scores["weak"]
    assert lappas_style_skill_coverage_score(strong, task) == 1.0


def test_lappas_coverage_ignores_candidate_needs():
    task = Task(
        task_id="ignore_needs",
        goal="Needs should not affect skill coverage.",
        requirements=[_req("python", 0.7), _req("frontend", 0.5)],
    )
    base_caps = [_cap("python", 0.7), _cap("frontend", 0.5)]
    low_need = UserState("candidate_a", capabilities=base_caps, needs=[_need("mentoring", 0.1)])
    high_need = UserState("candidate_b", capabilities=base_caps, needs=[_need("mentoring", 1.0)])

    score_low = lappas_style_skill_coverage_score(low_need, task)
    score_high = lappas_style_skill_coverage_score(high_need, task)

    assert score_low == score_high


def test_mapscore_greedy_can_use_task_offers_and_candidate_needs():
    task = Task(
        task_id="mapscore_need",
        goal="Full MapScore should include S_need.",
        requirements=[_req("python", 0.7)],
        offers=[
            TaskOffer(
                ENC("publication strategy support"),
                strength=1.0,
                source="explicit",
                description="publication strategy support",
            ),
            TaskOffer(
                ENC("low visibility maintenance"),
                strength=0.05,
                source="explicit",
                description="low visibility maintenance",
            ),
        ],
    )
    requester = UserState("requester")
    same_cap_low_need = UserState(
        "low_need",
        capabilities=[_cap("python", 0.7)],
        needs=[_need("low visibility maintenance", 1.0)],
    )
    same_cap_high_need = UserState(
        "high_need",
        capabilities=[_cap("python", 0.7)],
        needs=[_need("publication strategy support", 1.0)],
    )

    _, selected, _, scored = run_mapscore_greedy(
        requester, [same_cap_low_need, same_cap_high_need], task
    )

    assert selected == "high_need"
    assert scored["matches"]["high_need"].s_need > scored["matches"]["low_need"].s_need


def test_evaluate_task_reports_static_diagnostics():
    task_entry = {
        "task": {
            "task_id": "diagnostic_task",
            "title": "Diagnostic Task",
            "description": "Task for diagnostic reporting.",
            "required_skills": {"python": 0.7},
            "offers": {"publication strategy support": 0.9},
            "metadata": {},
        },
        "proposer_profile": {
            "user_id": "requester",
            "role": "requester",
            "capabilities": {},
            "needs": {},
            "preferences": {},
            "constraints": {},
        },
        "candidates": [
            {
                "candidate_profile": {
                    "user_id": "candidate_a",
                    "role": "engineer",
                    "capabilities": {"python": 0.7},
                    "needs": {"unrelated benefit": 0.2},
                    "preferences": {"availability": "high", "current_load": 0.1},
                    "constraints": {"workload": 0.1},
                },
                "candidate_card": {
                    "candidate_id": "candidate_a",
                    "summary": "A",
                    "highlighted_strengths": ["python"],
                    "highlighted_risks": [],
                    "explanation": "",
                },
                "capability_priors": {"python": {"mu_init": 0.7, "sigma_init": 0.01}},
                "context_latents": {},
            },
            {
                "candidate_profile": {
                    "user_id": "candidate_b",
                    "role": "engineer",
                    "capabilities": {"python": 0.7},
                    "needs": {"publication strategy support": 1.0},
                    "preferences": {"availability": "high", "current_load": 0.1},
                    "constraints": {"workload": 0.1},
                },
                "candidate_card": {
                    "candidate_id": "candidate_b",
                    "summary": "B",
                    "highlighted_strengths": ["python"],
                    "highlighted_risks": [],
                    "explanation": "",
                },
                "capability_priors": {"python": {"mu_init": 0.7, "sigma_init": 0.01}},
                "context_latents": {},
            },
        ],
    }

    result = evaluate_task(task_entry)

    assert "mapscore_greedy" in result["conditions"]
    assert result["diagnostics"]["s_need_pool"]["range"] > 0.0
    assert "total_reward" in result["diagnostics"]["bilateral_validity"]["mapscore_greedy"]
