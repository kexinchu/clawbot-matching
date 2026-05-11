from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ONLINE_LEARNING_DIR = REPO_ROOT / "Online_learning"
MAPPING_ALGO_DIR = REPO_ROOT / "mapping-algo"

for path in (str(REPO_ROOT), str(ONLINE_LEARNING_DIR), str(MAPPING_ALGO_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import MatchConfig 
from datatypes import CapabilityEntry, NeedEntry, Task, TaskOffer, TaskRequirement, UserState 
from encoder import SimpleEncoder 
from Online_learning import OnlineLearning  
from WorldModel import WorldModel 


ENC = SimpleEncoder(dim=64)


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
        task_id="task_layer3",
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


class TestOnlineLearningLayer3(unittest.TestCase):
    def setUp(self):
        cfg = MatchConfig(embedding_dim=64)
        self.world_model = WorldModel(config=cfg, theta_c=0.4, theta_n=-0.1)
        self.engine = OnlineLearning(self.world_model, layer3_top_k=3, layer3_top_n=3)
        self.requester = make_user(
            "alice",
            caps=[
                ("bayesian statistics", 0.3, 0.2, "explicit"),
                ("python programming", 0.8, 0.1, "explicit"),
            ],
            needs=[("bayesian statistics mentorship", 0.8)],
        )
        self.bob = make_user(
            "bob",
            caps=[
                ("bayesian statistics", 0.9, 0.1, "explicit"),
                ("python programming", 0.5, 0.2, "explicit"),
            ],
            needs=[("academic paper writing", 0.6)],
        )
        self.carol = make_user(
            "carol",
            caps=[
                ("bayesian statistics", 0.6, 0.4, "meta"),
                ("python programming", 0.7, 0.35, "meta"),
            ],
            needs=[("python programming", 0.3)],
        )
        self.task = make_task(
            reqs=[
                ("bayesian statistics", 0.8, "soft"),
                ("python programming", 0.6, "soft"),
            ],
            offers=[("research collaboration", 0.8, "explicit")],
        )

    def test_run_one_round_includes_layer3_sections_for_single_candidate(self):
        report = self.engine.run_one_round(self.requester, self.bob, self.task)

        self.assertEqual(report["selected_candidate_id"], self.bob.user_id)
        self.assertEqual(report["layer3_1"]["candidate_ids"], [self.bob.user_id])
        self.assertEqual(report["layer3_1"]["num_ranked_candidates"], 1)
        self.assertEqual(report["layer3_2"]["candidate_ids"], [self.bob.user_id])
        self.assertEqual(report["layer3_2"]["num_refined_candidates"], 1)

    def test_run_one_round_uses_candidate_pool_for_layer3(self):
        pool = [self.bob, self.carol]

        report = self.engine.run_one_round(
            self.requester,
            self.bob,
            self.task,
            candidate_pool=pool,
        )

        self.assertGreaterEqual(report["layer3_1"]["num_ranked_candidates"], 1)
        self.assertGreaterEqual(report["layer3_2"]["num_refined_candidates"], 1)
        self.assertIn(report["selected_candidate_id"], {self.bob.user_id, self.carol.user_id})
        self.assertIn(report["selected_candidate_id"], report["layer3_2"]["candidate_ids"])


if __name__ == "__main__":
    unittest.main()
