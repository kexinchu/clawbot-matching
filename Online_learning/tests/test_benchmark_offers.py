"""Regression tests for benchmark task offers entering MapScore."""

from config import MatchConfig
from datatypes import NeedEntry, UserState
from scoring import compute_s_need

from run_20_tasks_evaluation import ENC, build_task


def test_build_task_offers_make_s_need_nonzero():
    task = build_task({
        "task_id": "task_offer_regression",
        "title": "Offer Regression",
        "description": "Task with explicit offers.",
        "required_skills": {"python": 0.7},
        "offers": {"python mentoring": 0.9},
    })
    candidate = UserState(
        user_id="candidate_with_matching_need",
        needs=[NeedEntry(ENC("python mentoring"), intensity=0.8, description="python mentoring")],
    )

    score, details = compute_s_need(candidate, task, MatchConfig(embedding_dim=64))

    assert score > 0.0
    assert details
