from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_MAPPING_ALGO_DIR = Path(__file__).resolve().parents[1] / "mapping-algo"
if str(_MAPPING_ALGO_DIR) not in sys.path:
    sys.path.append(str(_MAPPING_ALGO_DIR))

REPO_ROOT = Path(__file__).resolve().parents[1]
ONLINE_LEARNING_DIR = REPO_ROOT / "Online_learning"
FIXTURE_PATH = REPO_ROOT / "LLM_Dreaming" / "dreaming_fixture.json"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ONLINE_LEARNING_DIR))
from config import MatchConfig
from WorldModel import WorldModel

cfg = MatchConfig(embedding_dim=64)
world_model = WorldModel(theta_c=0.4, theta_n=-0.1, config=cfg)

import LLM_Dreaming.LLM_dreaming as dreaming

BASE_URL = "https://api.commonstack.ai/v1"


class TestDreamSimulatorLoading(unittest.TestCase):
    def test_uses_mock_mode_without_api_key(self):
        with mock.patch.object(dreaming, "API_KEY", ""):
            simulator = dreaming.DreamSimulator(base_url=BASE_URL)

        self.assertTrue(simulator.mock_mode)
        self.assertEqual(simulator.api_url, "https://api.commonstack.ai/v1/chat/completions")
        self.assertFalse(hasattr(simulator, "client"))

    def test_creates_http_client_when_api_key_is_present(self):
        created = {}

        class DummyClient:
            pass

        def fake_client(*, timeout):
            created["timeout"] = timeout
            return DummyClient()

        with mock.patch.object(dreaming, "API_KEY", "test-key"):
            fake_httpx = types.SimpleNamespace(Client=fake_client)
            with mock.patch.object(dreaming, "httpx", fake_httpx):
                simulator = dreaming.DreamSimulator(base_url=BASE_URL)

        self.assertFalse(simulator.mock_mode)
        self.assertIsInstance(simulator.client, DummyClient)
        self.assertEqual(created["timeout"], 60.0)


class TestProfileBuilding(unittest.TestCase):
    def test_generated_fixture_contains_requested_persona_counts(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(fixture["metadata"]["num_requester_personas"], 1)
        self.assertEqual(fixture["metadata"]["num_candidate_personas"], 5)
        self.assertEqual(len(fixture["requester_personas"]), 1)
        self.assertEqual(len(fixture["candidate_personas"]), 5)

    def test_create_test_candidates_builds_extended_and_soft_profiles(self):
        requester, task, candidates = dreaming.create_test_candidates()
        candidate = candidates[0]

        self.assertIsInstance(requester, dreaming.ExtendedProfile)
        self.assertIsInstance(requester.profile, dreaming.UserState)
        self.assertIsInstance(requester.soft, dreaming.SoftProfile)
        self.assertEqual(requester.user_id, "alice")
        self.assertEqual(task.goal, "Build Bayesian churn model, target NeurIPS")

        self.assertIsInstance(candidate, dreaming.ExtendedProfile)
        self.assertIsInstance(candidate.profile, dreaming.UserState)
        self.assertIsInstance(candidate.soft, dreaming.SoftProfile)
        self.assertEqual(candidate.user_id, "bob")
        self.assertEqual(candidate.soft.availability, "20h/week")
        self.assertIn("first-author NeurIPS paper", candidate.soft.priorities)

    def test_build_agent_persona_contains_role_specific_profile_details(self):
        requester, task, candidates = dreaming.create_test_candidates()
        candidate = candidates[0]

        requester_persona = dreaming.build_agent_persona(requester, "requester", task)
        candidate_persona = dreaming.build_agent_persona(candidate, "candidate", task)

        self.assertIn("posted the task", requester_persona)
        self.assertIn(requester.user_id, requester_persona)
        self.assertIn(requester.soft.collab_style, requester_persona)
        self.assertIn("considered for the task", candidate_persona)
        self.assertIn(candidate.user_id, candidate_persona)
        self.assertIn(candidate.soft.communication, candidate_persona)


class TestDreamConversation(unittest.TestCase):
    def test_simulate_conversation_returns_transcript_and_compatibility(self):
        requester, task, candidates = dreaming.create_test_candidates()
        candidate = candidates[0]

        with mock.patch.object(dreaming, "API_KEY", ""):
            simulator = dreaming.DreamSimulator(base_url=BASE_URL, n_turns=3)
            result = simulator.simulate_conversation(requester, candidate, task)

        self.assertEqual(result["candidate_id"], candidate.user_id)
        self.assertEqual(len(result["transcript"]), 6)
        self.assertEqual(result["transcript"][0]["role"], f"agent_{requester.user_id}")
        self.assertEqual(result["transcript"][1]["role"], f"agent_{candidate.user_id}")

        compatibility = result["compatibility"]
        self.assertGreaterEqual(compatibility["overall_compatibility"], 0.0)
        self.assertLessEqual(compatibility["overall_compatibility"], 1.0)
        self.assertIn(compatibility["recommendation"], {
            "strong_match",
            "good_match",
            "risky_match",
            "poor_match",
        })
        for key in ("time_energy", "priority_alignment", "collab_style", "personality_fit"):
            self.assertIn("score", compatibility[key])
            self.assertIn("reason", compatibility[key])


class TestLayer3PipelineAdapter(unittest.TestCase):
    def test_refines_mapping_pipeline_match_results(self):
        requester, task, candidates = dreaming.create_test_candidates()
        candidate = candidates[0]
        layer3_output = [
            dreaming.MappingMatchResult(
                candidate_id=candidate.user_id,
                match_score=0.82,
                s_cap=0.9,
                s_need=0.7,
                w_c=0.6,
                w_n=0.4,
                sigma_gate=1,
            )
        ]

        with mock.patch.object(dreaming, "API_KEY", ""):
            simulator = dreaming.DreamSimulator(base_url=BASE_URL, n_turns=2)
            planning = dreaming.PlanningLayer(
                world_model=world_model,
                dream_simulator=simulator,
                top_k=1,
                top_n=1,
            )
            refined = planning.run_from_layer3_output(
                requester=requester,
                task=task,
                candidate_profiles=candidates,
                layer3_output=layer3_output,
                verbose=False,
            )

        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0].candidate_id, candidate.user_id)
        self.assertEqual(refined[0].analytical_score, 0.82)
        self.assertGreaterEqual(refined[0].dream_score, 0.0)
        self.assertGreater(len(refined[0].transcript), 0)
