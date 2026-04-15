#!/usr/bin/env python3
"""Generate JSON test fixtures for the matching algorithm.

Produces:
  fixtures/scenario_academic.json  — full academic collaboration scenario
  fixtures/candidates_pool.json    — larger candidate pool for pipeline tests

Embeddings are generated via SBERTEncoder (BAAI/bge-base-en-v1.5, 768-dim).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add mapping-algo/ to sys.path so we can import encoder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder_sbert import SBERTEncoder

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

enc = SBERTEncoder("BAAI/bge-base-en-v1.5")


def _emb(text: str) -> list:
    return enc(text).tolist()


# ═══════════════════════════════════════════════════════════════════════
#  Scenario: Academic Collaboration (from §8 numerical example)
# ═══════════════════════════════════════════════════════════════════════

def generate_academic_scenario() -> dict:
    """Maya needs a clinical collaborator for a multimodal triage paper."""

    requester = {
        "user_id": "maya_001",
        "capabilities": [
            {"description": "multimodal ML", "embedding": _emb("multimodal ML"),
             "mu": 0.9, "sigma": 0.1, "source": "explicit"},
            {"description": "Python", "embedding": _emb("Python"),
             "mu": 0.8, "sigma": 0.1, "source": "explicit"},
        ],
        "needs": [
            {"description": "clinical expertise",
             "embedding": _emb("clinical expertise"), "intensity": 0.9},
            {"description": "paper writing help",
             "embedding": _emb("paper writing help"), "intensity": 0.7},
        ],
        "clearance_level": 2,
    }

    task = {
        "task_id": "task_001",
        "goal": "Need a clinical collaborator for multimodal patient triage paper",
        "requirements": [
            {"description": "clinical data analysis",
             "embedding": _emb("clinical data analysis"),
             "level": 0.8, "constraint_type": "hard"},
            {"description": "multimodal ML",
             "embedding": _emb("multimodal ML"),
             "level": 0.7, "constraint_type": "soft"},
            {"description": "paper writing",
             "embedding": _emb("paper writing"),
             "level": 0.6, "constraint_type": "soft"},
        ],
        "offers": [
            {"description": "NeurIPS co-authorship",
             "embedding": _emb("NeurIPS co-authorship"),
             "strength": 0.9, "source": "explicit"},
            {"description": "clinical dataset access",
             "embedding": _emb("clinical dataset access"),
             "strength": 0.8, "source": "explicit"},
        ],
        "data_clearance": 1,
    }

    candidates = [
        {
            "user_id": "bob_002",
            "capabilities": [
                {"description": "clinical research",
                 "embedding": _emb("clinical research"),
                 "mu": 0.85, "sigma": 0.2, "source": "explicit"},
                {"description": "medical writing",
                 "embedding": _emb("medical writing"),
                 "mu": 0.7, "sigma": 0.3, "source": "implicit"},
                {"description": "statistics",
                 "embedding": _emb("statistics"),
                 "mu": 0.6, "sigma": 0.4, "source": "meta"},
            ],
            "needs": [
                {"description": "learn deep learning",
                 "embedding": _emb("learn deep learning"), "intensity": 0.8},
                {"description": "NeurIPS publication",
                 "embedding": _emb("NeurIPS publication"), "intensity": 0.9},
            ],
            "clearance_level": 2,
        },
        {
            "user_id": "carol_003",
            "capabilities": [
                {"description": "biostatistics",
                 "embedding": _emb("biostatistics"),
                 "mu": 0.75, "sigma": 0.15, "source": "explicit"},
                {"description": "clinical trials",
                 "embedding": _emb("clinical trials"),
                 "mu": 0.8, "sigma": 0.2, "source": "explicit"},
                {"description": "academic writing",
                 "embedding": _emb("academic writing"),
                 "mu": 0.65, "sigma": 0.25, "source": "implicit"},
            ],
            "needs": [
                {"description": "machine learning skills",
                 "embedding": _emb("machine learning skills"), "intensity": 0.85},
                {"description": "top venue publication",
                 "embedding": _emb("top venue publication"), "intensity": 0.7},
            ],
            "clearance_level": 2,
        },
        {
            # Dave — should FAIL gate (clearance too low)
            "user_id": "dave_004",
            "capabilities": [
                {"description": "web development",
                 "embedding": _emb("web development"),
                 "mu": 0.9, "sigma": 0.1, "source": "explicit"},
                {"description": "JavaScript",
                 "embedding": _emb("JavaScript"),
                 "mu": 0.85, "sigma": 0.1, "source": "explicit"},
            ],
            "needs": [
                {"description": "backend experience",
                 "embedding": _emb("backend experience"), "intensity": 0.6},
            ],
            "clearance_level": 0,  # too low for task (needs 1)
        },
        {
            # Eve — should FAIL gate (no clinical capability for hard constraint)
            "user_id": "eve_005",
            "capabilities": [
                {"description": "natural language processing",
                 "embedding": _emb("natural language processing"),
                 "mu": 0.9, "sigma": 0.1, "source": "explicit"},
                {"description": "paper writing",
                 "embedding": _emb("paper writing"),
                 "mu": 0.8, "sigma": 0.15, "source": "explicit"},
            ],
            "needs": [
                {"description": "clinical data experience",
                 "embedding": _emb("clinical data experience"), "intensity": 0.9},
            ],
            "clearance_level": 2,
        },
    ]

    return {
        "scenario": "academic_collaboration",
        "description": "Maya needs a clinical collaborator for a multimodal patient triage paper",
        "embedding_dim": enc.embedding_dim,
        "requester": requester,
        "task": task,
        "candidates": candidates,
        "expected": {
            "gate_pass": ["bob_002", "carol_003"],
            "gate_fail": {
                "dave_004": "clearance",
                "eve_005": "hard constraint",
            },
            "top_1_candidate": "bob_002",
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  Larger candidate pool for pipeline / team building tests
# ═══════════════════════════════════════════════════════════════════════

def generate_candidate_pool() -> dict:
    """Generate 10 candidates with varying capabilities for pipeline tests."""

    rng = np.random.RandomState(123)

    skill_pool = [
        "clinical research", "medical writing", "statistics",
        "deep learning", "computer vision", "NLP",
        "data engineering", "Python", "R programming",
        "experimental design", "paper writing", "biostatistics",
        "clinical trials", "epidemiology", "genomics",
    ]

    need_pool = [
        "learn deep learning", "NeurIPS publication",
        "industry experience", "mentorship", "dataset access",
        "coding skills", "research methodology", "networking",
    ]

    candidates = []
    for i in range(10):
        uid = f"cand_{i:03d}"
        n_caps = rng.randint(2, 6)
        n_needs = rng.randint(1, 4)

        caps = []
        chosen_skills = rng.choice(skill_pool, size=n_caps, replace=False)
        for skill in chosen_skills:
            caps.append({
                "description": skill,
                "embedding": _emb(skill),
                "mu": round(float(rng.uniform(0.3, 0.95)), 2),
                "sigma": round(float(rng.uniform(0.05, 0.5)), 2),
                "source": rng.choice(["explicit", "implicit", "meta"]),
            })

        needs = []
        chosen_needs = rng.choice(need_pool, size=n_needs, replace=False)
        for need in chosen_needs:
            needs.append({
                "description": need,
                "embedding": _emb(need),
                "intensity": round(float(rng.uniform(0.3, 0.95)), 2),
            })

        candidates.append({
            "user_id": uid,
            "capabilities": caps,
            "needs": needs,
            "clearance_level": int(rng.choice([0, 1, 2, 3])),
        })

    return {
        "description": "10 randomly generated candidates for pipeline tests",
        "embedding_dim": enc.embedding_dim,
        "candidates": candidates,
    }


# ═══════════════════════════════════════════════════════════════════════

def main():
    # Academic scenario
    scenario = generate_academic_scenario()
    path = FIXTURES_DIR / "scenario_academic.json"
    with open(path, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"✓ {path}  ({len(scenario['candidates'])} candidates)")

    # Candidate pool
    pool = generate_candidate_pool()
    path = FIXTURES_DIR / "candidates_pool.json"
    with open(path, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"✓ {path}  ({len(pool['candidates'])} candidates)")


if __name__ == "__main__":
    main()
