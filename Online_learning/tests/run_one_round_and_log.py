"""Run OnlineLearning for one round and save a log under repo logs/.

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
            ("bayesian statistics", 0.9, 0.1, "explicit"),
            ("python programming", 0.5, 0.2, "explicit"),
        ],
        needs=[("academic paper writing", 0.6)],
    )
    carol = make_user(
        "carol",
        caps=[
            ("bayesian statistics", 0.6, 0.4, "meta"),
            ("python programming", 0.7, 0.35, "meta"),
        ],
        needs=[("python programming", 0.3)],
    )
    dave = make_user(
        "dave",
        caps=[
            ("bayesian statistics", 0.85, 0.15, "explicit"),
            ("python programming", 0.75, 0.1, "explicit"),
        ],
        needs=[("research collaboration", 0.4)],
    )
    task = make_task(
        reqs=[
            ("bayesian statistics", 0.8, "soft"),
            ("python programming", 0.6, "soft"),
        ],
        offers=[("research collaboration", 0.8, "explicit")],
    )
    return requester, bob, [bob, carol, dave], task


def write_log(log_path: Path, report: dict, requester_id: str, pool_ids: list[str]) -> None:
    lines = [
        "OnlineLearning one-round run",
        f"timestamp: {datetime.now().isoformat()}",
        f"requester: {requester_id}",
        f"candidate_pool: {', '.join(pool_ids)}",
        f"selected_candidate_id: {report['selected_candidate_id']}",
        "",
        "Layer 3.1 ranking:",
        f"  ranked candidates: {report['layer3_1']['candidate_ids']}",
        f"  num ranked: {report['layer3_1']['num_ranked_candidates']}",
        "",
        "Layer 3.2 dreaming:",
        f"  refined candidates: {report['layer3_2']['candidate_ids']}",
        f"  recommendations: {report['layer3_2']['recommendations']}",
        "",
        "Match score:",
        f"  {json.dumps(report['match_score'], indent=2)}",
        "",
        "Feedback:",
        f"  {json.dumps(report['feedback'], indent=2)}",
        "",
        "Reward:",
        f"  {json.dumps(report['reward'], indent=2)}",
        "",
        "Path 1 Bayesian update:",
        f"  {json.dumps(report['path1_bayesian'], indent=2)}",
        "",
        "Path 2 weights:",
        f"  {json.dumps(report['path2_weights'], indent=2)}",
        "",
        "Path 3 UCB:",
        f"  {json.dumps(report['path3_ucb'], indent=2)}",
        "",
        "Full report JSON:",
        json.dumps(report, indent=2),
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    requester, default_candidate, candidate_pool, task = build_demo_inputs()
    world_model = WorldModel(config=CFG, theta_c=0.4, theta_n=-0.1)
    engine = OnlineLearning(world_model)

    report = engine.run_one_round(
        requester=requester,
        candidate=default_candidate,
        task=task,
        candidate_pool=candidate_pool,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_DIR / f"online_learning_one_round_{timestamp}.log"
    write_log(
        log_path=log_path,
        report=report,
        requester_id=requester.user_id,
        pool_ids=[candidate.user_id for candidate in candidate_pool],
    )

    print(f"Wrote one-round OnlineLearning log to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
