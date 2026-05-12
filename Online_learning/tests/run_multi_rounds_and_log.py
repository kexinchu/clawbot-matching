"""Run OnlineLearning for multiple rounds and save JSON traces under repo logs/.

Usage:
    python Online_learning/tests/run_one_round_and_log.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
ONLINE_LEARNING_DIR = TESTS_DIR.parent
REPO_ROOT = ONLINE_LEARNING_DIR.parent
MAPPING_ALGO_DIR = REPO_ROOT / "mapping-algo"
LOGS_DIR = REPO_ROOT / "logs"

for path in (ONLINE_LEARNING_DIR, MAPPING_ALGO_DIR, REPO_ROOT, TESTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from WorldModel import WorldModel  # noqa: E402
from Online_learning import OnlineLearning  # noqa: E402
from config import MatchConfig  # noqa: E402
from datatypes import CapabilityEntry, NeedEntry, Task, TaskOffer, TaskRequirement, UserState  # noqa: E402
from encoder import SimpleEncoder  # noqa: E402


ENC = SimpleEncoder(dim=64)
CFG = MatchConfig(embedding_dim=64)
N_ROUNDS = 20


def make_user(uid, caps, needs=None, clearance=0):
    return UserState(
        user_id=uid,
        capabilities=[
            CapabilityEntry(ENC(desc), mu=mu, sigma=sigma, source=source, description=desc)
            for desc, mu, sigma, source in caps
        ],
        needs=[
            NeedEntry(ENC(desc), intensity=intensity, description=desc)
            for desc, intensity in (needs or [])
        ],
        clearance_level=clearance,
    )


def make_task(reqs, offers=None, data_clearance=0):
    return Task(
        task_id="t_test",
        goal="test task",
        requirements=[
            TaskRequirement(ENC(desc), level=level, constraint_type=ctype, description=desc)
            for desc, level, ctype in reqs
        ],
        offers=[
            TaskOffer(ENC(desc), strength=strength, source=source, description=desc)
            for desc, strength, source in (offers or [])
        ],
        data_clearance=data_clearance,
    )


def build_demo_inputs():
    requester = make_user(
        "alice",
        caps=[
            ("bayesian statistics", 0.3, 0.2, "explicit"),
            ("python programming", 0.8, 0.1, "explicit"),
        ],
        needs=[("bayesian statistics mentorship", 0.8)],
    )
    bob = make_user(
        "bob",
        caps=[
            ("bayesian statistics", 0.6, 0.1, "explicit"),
            ("python programming", 0.5, 0.2, "explicit"),
        ],
        needs=[("academic paper writing", 0.6)],
    )
    carol = make_user(
        "carol",
        caps=[
            ("bayesian statistics", 0.6, 0.1, "meta"),
            ("python programming", 0.5, 0.2, "meta"),
        ],
        needs=[("python programming", 0.3)],
    )
    dave = make_user(
        "dave",
        caps=[
            ("bayesian statistics", 0.8, 0.15, "explicit"),
            ("python programming", 0.75, 0.1, "explicit"),
        ],
        needs=[("research collaboration", 0.4)],
    )
    task = make_task(
        reqs=[
            ("bayesian statistics", 0.95, "soft"),
            ("python programming", 0.9, "soft"),
        ],
        offers=[("research collaboration", 0.8, "explicit")],
    )
    return requester, [bob, carol, dave], task


def snapshot_candidate_means(candidate_pool: list[UserState]) -> dict[str, dict[str, float]]:
    return {
        candidate.user_id: {
            cap.description: round(cap.mu, 4)
            for cap in candidate.capabilities
        }
        for candidate in candidate_pool
    }


def snapshot_candidate_sigmas(candidate_pool: list[UserState]) -> dict[str, dict[str, float]]:
    return {
        candidate.user_id: {
            cap.description: round(cap.sigma, 4)
            for cap in candidate.capabilities
        }
        for candidate in candidate_pool
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    requester, candidate_pool, task = build_demo_inputs()
    world_model = WorldModel(config=CFG, theta_c=0.4, theta_n=-0.1)
    engine = OnlineLearning(world_model)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "num_rounds": N_ROUNDS,
        "requester_id": requester.user_id,
        "candidate_pool": [candidate.user_id for candidate in candidate_pool],
        "task_id": task.task_id,
        "task_goal": task.goal,
    }
    means_trace = {"metadata": metadata, "rounds": []}
    sigmas_trace = {"metadata": metadata, "rounds": []}
    match_scores_trace = {"metadata": metadata, "rounds": []}
    reward_trace = {"metadata": metadata, "rounds": []}

    for _ in range(N_ROUNDS):
        report = engine.run_one_round(
            requester=requester,
            candidate=None,
            task=task,
            candidate_pool=candidate_pool,
        )
        round_idx = report["round"]

        means_trace["rounds"].append({
            "round": round_idx,
            "selected_candidate_id": report["selected_candidate_id"],
            "candidate_means": snapshot_candidate_means(candidate_pool),
        })
        sigmas_trace["rounds"].append({
            "round": round_idx,
            "selected_candidate_id": report["selected_candidate_id"],
            "candidate_sigmas": snapshot_candidate_sigmas(candidate_pool),
        })
        match_scores_trace["rounds"].append({
            "round": round_idx,
            "selected_candidate_id": report["selected_candidate_id"],
            "selection_counts": report.get("selection_counts", {}),
            "candidate_ucb": report.get("candidate_ucb", {}),
            "layer3_1": report["layer3_1"],
            "layer3_2": report["layer3_2"],
            "match_score": report["match_score"],
        })
        reward_trace["rounds"].append({
            "round": round_idx,
            "selected_candidate_id": report["selected_candidate_id"],
            "reward": report["reward"],
        })

    means_path = LOGS_DIR / f"online_learning_candidate_means_{timestamp}.json"
    sigmas_path = LOGS_DIR / f"online_learning_candidate_sigmas_{timestamp}.json"
    match_scores_path = LOGS_DIR / f"online_learning_match_scores_{timestamp}.json"
    reward_path = LOGS_DIR / f"online_learning_reward_scores_{timestamp}.json"

    write_json(means_path, means_trace)
    write_json(sigmas_path, sigmas_trace)
    write_json(match_scores_path, match_scores_trace)
    write_json(reward_path, reward_trace)

    print(f"Wrote candidate means trace to {means_path}")
    print(f"Wrote candidate sigmas trace to {sigmas_path}")
    print(f"Wrote match scores trace to {match_scores_path}")
    print(f"Wrote reward scores trace to {reward_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
