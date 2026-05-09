"""L1 parser tests — rule-based JSON/dict → UserState/Task.

Covers:
  1. Basic parsing from dict
  2. Parsing from JSON file
  3. Default values (sigma, source, intensity, constraint_type)
  4. All source types (explicit / implicit / meta)
  5. Hard and soft constraint types preserved
  6. Inferred offer source preserved
  7. Edge cases: empty capabilities, empty needs, empty requirements
  8. Embedding shape correctness
  9. Clearance level propagation
  10. data_clearance propagation to Task
"""

import os
import json
import tempfile
import numpy as np
import pytest

from L1_parser import parse_user, parse_task
from config import MatchConfig

CFG = MatchConfig()
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ── 1. Basic parsing from dict ────────────────────────────────────────

def test_parse_user_basic():
    u = parse_user({
        "user_id": "alice",
        "capabilities": [
            {"description": "bayesian statistics", "mu": 0.8, "sigma": 0.2, "source": "explicit"}
        ],
        "needs": [
            {"description": "python mentorship", "intensity": 0.6}
        ]
    })
    assert u.user_id == "alice"
    assert len(u.capabilities) == 1
    assert len(u.needs) == 1
    assert u.capabilities[0].mu == 0.8
    assert u.capabilities[0].sigma == 0.2
    assert u.capabilities[0].source == "explicit"
    assert u.capabilities[0].description == "bayesian statistics"
    assert u.needs[0].intensity == 0.6
    assert u.needs[0].description == "python mentorship"


def test_parse_task_basic():
    t = parse_task({
        "task_id": "t1",
        "goal": "build model",
        "requirements": [
            {"description": "machine learning", "level": 0.7, "constraint_type": "soft"}
        ],
        "offers": [
            {"description": "co-authorship", "strength": 0.8, "source": "explicit"}
        ]
    })
    assert t.task_id == "t1"
    assert t.goal == "build model"
    assert len(t.requirements) == 1
    assert len(t.offers) == 1
    assert t.requirements[0].level == 0.7
    assert t.requirements[0].constraint_type == "soft"
    assert t.offers[0].strength == 0.8
    assert t.offers[0].source == "explicit"


# ── 2. Parsing from JSON file ─────────────────────────────────────────

def test_parse_user_from_file():
    alice = parse_user(os.path.join(DATA_DIR, "alice.json"))
    assert alice.user_id == "alice"
    assert len(alice.capabilities) == 3
    assert len(alice.needs) == 3
    descs = {c.description for c in alice.capabilities}
    assert "bayesian statistics" in descs
    assert "python programming" in descs
    assert "academic paper writing" in descs


def test_parse_task_from_file():
    task = parse_task(os.path.join(DATA_DIR, "task_001.json"))
    assert task.task_id == "task_001"
    assert len(task.requirements) == 3
    assert len(task.offers) == 3


def test_parse_user_from_tempfile():
    data = {
        "user_id": "temp_user",
        "capabilities": [{"description": "deep learning", "mu": 0.9}],
        "needs": []
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        u = parse_user(path)
        assert u.user_id == "temp_user"
        assert u.capabilities[0].mu == 0.9
    finally:
        os.unlink(path)


# ── 3. Default values ─────────────────────────────────────────────────

def test_default_sigma_is_sigma_init():
    """Missing sigma defaults to MatchConfig.sigma_init (1.0)."""
    u = parse_user({
        "user_id": "u",
        "capabilities": [{"description": "python", "mu": 0.5}],
        "needs": []
    })
    assert u.capabilities[0].sigma == CFG.sigma_init


def test_default_source_is_explicit():
    """Missing source defaults to 'explicit'."""
    u = parse_user({
        "user_id": "u",
        "capabilities": [{"description": "python", "mu": 0.5}],
        "needs": []
    })
    assert u.capabilities[0].source == "explicit"


def test_default_need_intensity():
    """Missing intensity defaults to 0.5."""
    u = parse_user({
        "user_id": "u",
        "capabilities": [],
        "needs": [{"description": "mentorship"}]
    })
    assert u.needs[0].intensity == 0.5


def test_default_requirement_level():
    """Missing level defaults to 0.5."""
    t = parse_task({
        "task_id": "t",
        "goal": "g",
        "requirements": [{"description": "coding"}],
        "offers": []
    })
    assert t.requirements[0].level == 0.5


def test_default_constraint_type_is_soft():
    """Missing constraint_type defaults to 'soft'."""
    t = parse_task({
        "task_id": "t", "goal": "g",
        "requirements": [{"description": "coding", "level": 0.6}],
        "offers": []
    })
    assert t.requirements[0].constraint_type == "soft"


def test_default_offer_source_is_explicit():
    """Missing offer source defaults to 'explicit'."""
    t = parse_task({
        "task_id": "t", "goal": "g",
        "requirements": [],
        "offers": [{"description": "collaboration", "strength": 0.7}]
    })
    assert t.offers[0].source == "explicit"


def test_default_offer_strength():
    """Missing strength defaults to 0.5."""
    t = parse_task({
        "task_id": "t", "goal": "g",
        "requirements": [],
        "offers": [{"description": "collaboration"}]
    })
    assert t.offers[0].strength == 0.5


def test_default_task_id():
    t = parse_task({"goal": "g", "requirements": [], "offers": []})
    assert t.task_id == "unknown"


# ── 4. All source types ───────────────────────────────────────────────

@pytest.mark.parametrize("source", ["explicit", "implicit", "meta"])
def test_all_capability_sources_preserved(source):
    u = parse_user({
        "user_id": "u",
        "capabilities": [{"description": "nlp", "mu": 0.6, "source": source}],
        "needs": []
    })
    assert u.capabilities[0].source == source


@pytest.mark.parametrize("source", ["explicit", "inferred"])
def test_all_offer_sources_preserved(source):
    t = parse_task({
        "task_id": "t", "goal": "g",
        "requirements": [],
        "offers": [{"description": "collab", "strength": 0.5, "source": source}]
    })
    assert t.offers[0].source == source


# ── 5. Hard / soft constraints ────────────────────────────────────────

def test_hard_constraint_preserved():
    t = parse_task({
        "task_id": "t", "goal": "g",
        "requirements": [
            {"description": "security clearance", "level": 0.9, "constraint_type": "hard"}
        ],
        "offers": []
    })
    assert t.requirements[0].constraint_type == "hard"


def test_mixed_constraints():
    t = parse_task({
        "task_id": "t", "goal": "g",
        "requirements": [
            {"description": "hard skill", "level": 0.9, "constraint_type": "hard"},
            {"description": "soft skill", "level": 0.6, "constraint_type": "soft"},
        ],
        "offers": []
    })
    ctypes = [r.constraint_type for r in t.requirements]
    assert ctypes == ["hard", "soft"]


# ── 6. Edge cases ─────────────────────────────────────────────────────

def test_empty_capabilities_no_crash():
    u = parse_user({"user_id": "u", "capabilities": [], "needs": []})
    assert u.capabilities == []
    assert u.needs == []


def test_empty_requirements_no_crash():
    t = parse_task({"task_id": "t", "goal": "g", "requirements": [], "offers": []})
    assert t.requirements == []
    assert t.offers == []


def test_missing_keys_no_crash():
    """A minimal dict with only user_id should not crash."""
    u = parse_user({"user_id": "minimal"})
    assert u.user_id == "minimal"
    assert u.capabilities == []
    assert u.needs == []


def test_multiple_capabilities_order_preserved():
    descs = ["alpha", "beta", "gamma"]
    u = parse_user({
        "user_id": "u",
        "capabilities": [{"description": d, "mu": 0.5} for d in descs],
        "needs": []
    })
    assert [c.description for c in u.capabilities] == descs


# ── 7. Embedding shape ────────────────────────────────────────────────

def test_embedding_shape_capabilities():
    u = parse_user({
        "user_id": "u",
        "capabilities": [{"description": "machine learning", "mu": 0.7}],
        "needs": []
    })
    emb = u.capabilities[0].embedding
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert emb.shape[0] == 64     # SimpleEncoder dim=64


def test_embedding_shape_needs():
    u = parse_user({
        "user_id": "u",
        "capabilities": [],
        "needs": [{"description": "mentorship", "intensity": 0.5}]
    })
    assert u.needs[0].embedding.shape == (64,)


def test_embedding_shape_requirements():
    t = parse_task({
        "task_id": "t", "goal": "g",
        "requirements": [{"description": "python", "level": 0.6}],
        "offers": []
    })
    assert t.requirements[0].embedding.shape == (64,)


def test_same_text_same_embedding():
    """SimpleEncoder is deterministic — same description → same embedding."""
    u1 = parse_user({"user_id": "a", "capabilities": [{"description": "deep learning", "mu": 0.5}], "needs": []})
    u2 = parse_user({"user_id": "b", "capabilities": [{"description": "deep learning", "mu": 0.8}], "needs": []})
    assert np.allclose(u1.capabilities[0].embedding, u2.capabilities[0].embedding)


def test_different_text_different_embedding():
    u1 = parse_user({"user_id": "a", "capabilities": [{"description": "deep learning", "mu": 0.5}], "needs": []})
    u2 = parse_user({"user_id": "b", "capabilities": [{"description": "quantum computing", "mu": 0.5}], "needs": []})
    assert not np.allclose(u1.capabilities[0].embedding, u2.capabilities[0].embedding)


# ── 8. Clearance levels ───────────────────────────────────────────────

def test_clearance_level_default_zero():
    u = parse_user({"user_id": "u", "capabilities": [], "needs": []})
    assert u.clearance_level == 0


def test_clearance_level_propagated():
    u = parse_user({"user_id": "u", "clearance_level": 3, "capabilities": [], "needs": []})
    assert u.clearance_level == 3


def test_data_clearance_default_zero():
    t = parse_task({"task_id": "t", "goal": "g", "requirements": [], "offers": []})
    assert t.data_clearance == 0


def test_data_clearance_propagated():
    t = parse_task({"task_id": "t", "goal": "g", "data_clearance": 2, "requirements": [], "offers": []})
    assert t.data_clearance == 2
