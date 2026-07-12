"""Helpers for loading / validating batch matching benchmarks."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_batch_testset(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def shared_candidate_profile(batch: dict, candidate_id: str) -> dict:
    for entry in batch["shared_candidates"]:
        profile = entry["candidate_profile"]
        if profile["user_id"] == candidate_id:
            return profile
    raise KeyError(candidate_id)


def verify_shared_profiles(batch: dict) -> list[str]:
    """Return list of invariant violation messages (empty => ok)."""
    errors: list[str] = []
    shared = {
        e["candidate_profile"]["user_id"]: e["candidate_profile"]
        for e in batch["shared_candidates"]
    }
    for task_entry in batch["tasks"]:
        for pair in task_entry["pair_entries"]:
            cid = pair["candidate_id"]
            if cid not in shared:
                errors.append(f"{cid} missing from shared_candidates")
                continue
            # Capability / need / availability / workload / timezone must match.
            a = pair["candidate_profile"]
            b = shared[cid]
            for key in ("capabilities", "needs", "preferences", "constraints"):
                if a.get(key) != b.get(key):
                    errors.append(
                        f"{batch['batch_id']}/{task_entry['task']['task_id']}/{cid}: "
                        f"shared profile field '{key}' mismatch"
                    )
    return errors


def pair_entry(batch: dict, task_id: str, candidate_id: str) -> dict:
    for task_entry in batch["tasks"]:
        if task_entry["task"]["task_id"] != task_id:
            continue
        for pair in task_entry["pair_entries"]:
            if pair["candidate_id"] == candidate_id:
                return pair
    raise KeyError((task_id, candidate_id))


def task_entry(batch: dict, task_id: str) -> dict:
    for te in batch["tasks"]:
        if te["task"]["task_id"] == task_id:
            return te
    raise KeyError(task_id)


def as_individual_task_entry(batch: dict, task_id: str, candidate_id: str) -> dict:
    """Adapt a batch pair into the individual testset shape for BilateralSimulator."""
    te = task_entry(batch, task_id)
    pe = pair_entry(batch, task_id, candidate_id)
    return {
        "task": deepcopy(te["task"]),
        "proposer_profile": deepcopy(te["proposer_profile"]),
        "candidates": [
            {
                "candidate_profile": deepcopy(pe["candidate_profile"]),
                "candidate_card": deepcopy(pe["candidate_card"]),
                "requester_personas": deepcopy(pe.get("requester_personas", [])),
                "candidate_personas": deepcopy(pe.get("candidate_personas", [])),
                "context_latents": deepcopy(pe.get("context_latents", {})),
            }
        ],
    }


def handcrafted_competition_batch(
    *,
    candidate_capacity: int = 1,
) -> dict:
    """Minimal batch where two requesters both prefer the same star candidate."""
    star = {
        "user_id": "cand_star",
        "role": "engineer",
        "capabilities": {"python": 0.95, "statistics": 0.9, "backend": 0.85},
        "needs": {"python": 0.9, "evaluation": 0.2},
        "preferences": {
            "availability": "high",
            "timezone": "UTC",
            "current_load": 0.2,
            "interests": "python statistics",
        },
        "constraints": {"workload": 0.2, "schedule_flexibility": "high"},
        "history_summary": "star candidate",
    }
    backup = {
        "user_id": "cand_backup",
        "role": "engineer",
        "capabilities": {"python": 0.7, "statistics": 0.65, "backend": 0.6},
        "needs": {"backend": 0.8},
        "preferences": {
            "availability": "medium",
            "timezone": "UTC",
            "current_load": 0.4,
            "interests": "backend",
        },
        "constraints": {"workload": 0.4, "schedule_flexibility": "medium"},
        "history_summary": "backup candidate",
    }
    weak = {
        "user_id": "cand_weak",
        "role": "designer",
        "capabilities": {"frontend": 0.9, "product_design": 0.8},
        "needs": {"frontend": 0.7},
        "preferences": {
            "availability": "low",
            "timezone": "CET",
            "current_load": 0.8,
            "interests": "frontend",
        },
        "constraints": {"workload": 0.8, "schedule_flexibility": "low"},
        "history_summary": "weak for these tasks",
    }

    def _task(tid: str, title: str, required: dict, offers: dict) -> dict:
        return {
            "task_id": tid,
            "title": title,
            "description": title,
            "required_skills": required,
            "offers": offers,
            "metadata": {
                "urgency": "medium",
                "budget": "competitive",
                "visibility": "high",
                "duration_weeks": 4,
                "domain": "matching",
            },
        }

    def _proposer(uid: str) -> dict:
        return {
            "user_id": uid,
            "role": "requester",
            "capabilities": {"python": 0.5},
            "needs": {"python": 0.8},
            "preferences": {"timezone": "UTC"},
            "constraints": {},
            "history_summary": "requester",
        }

    def _pair(cand: dict, feasible: bool) -> dict:
        return {
            "candidate_id": cand["user_id"],
            "candidate_profile": deepcopy(cand),
            "candidate_card": {
                "candidate_id": cand["user_id"],
                "summary": cand["user_id"],
                "highlighted_strengths": list(cand["capabilities"])[:2],
                "highlighted_risks": [],
                "explanation": "",
            },
            "requester_personas": [],
            "candidate_personas": [],
            "context_latents": {
                "history": {"prior_collaboration": False, "prior_skipped": False, "dispute_rate": 0.0},
                "latent_requester_preferences": {"structured": 0.5, "fastpaced": 0.5},
                "latent_candidate_preferences": {"creative": 0.5, "structured": 0.5},
                "latent_interpersonal_affinity": 0.0,
                "latent_risk_tolerance": 0.5,
                "latent_opportunity_bias": 0.0,
            },
            "public_feasible": feasible,
        }

    t1 = _task(
        "task_a",
        "Task A",
        {"python": 0.8, "statistics": 0.7},
        {"python": 0.9, "evaluation": 0.2, "frontend": 0.1},
    )
    t2 = _task(
        "task_b",
        "Task B",
        {"python": 0.75, "statistics": 0.7},
        {"python": 0.85, "backend": 0.3, "frontend": 0.1},
    )

    return {
        "batch_id": "handcrafted_competition",
        "candidate_capacity": candidate_capacity,
        "shared_candidates": [
            {"candidate_profile": deepcopy(star)},
            {"candidate_profile": deepcopy(backup)},
            {"candidate_profile": deepcopy(weak)},
        ],
        "tasks": [
            {
                "task": t1,
                "proposer_profile": _proposer("req_a"),
                "pair_entries": [
                    _pair(star, True),
                    _pair(backup, True),
                    _pair(weak, False),
                ],
            },
            {
                "task": t2,
                "proposer_profile": _proposer("req_b"),
                "pair_entries": [
                    _pair(star, True),
                    _pair(backup, True),
                    _pair(weak, False),
                ],
            },
        ],
    }


def summarize_batch_structure(testset: dict) -> dict[str, Any]:
    meta = testset.get("metadata", {})
    batches = testset.get("batches", [])
    return {
        "num_batches": len(batches),
        "tasks_per_batch": meta.get("tasks_per_batch"),
        "candidates_per_batch": meta.get("candidates_per_batch"),
        "candidate_capacity": meta.get("candidate_capacity"),
        "random_seed": meta.get("random_seed"),
    }
