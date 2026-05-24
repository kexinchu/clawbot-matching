"""Generate a theta-contrast testset: complement-dominant vs motivation-dominant.

Complement-dominant tasks: proposer has *empty* capabilities → large Gap_j.
Motivation-dominant tasks: strong explicit offers + candidate needs aligned.

Output: simulator/20_Tasks_ThetaContrast.json (10 + 10 tasks).
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
TIERED = _REPO / "simulator" / "20_Tasks_Testset_tiered.json"
OUT = _REPO / "simulator" / "20_Tasks_ThetaContrast.json"


def _make_complement_variant(entry: dict, idx: int) -> dict:
    """Proposer starts with no capabilities → complement_strength dominates."""
    out = deepcopy(entry)
    out["task"]["task_id"] = f"theta_comp_{idx:02d}"
    out["task"]["title"] = f"[complement] {out['task'].get('title', '')}"
    out["proposer_profile"]["user_id"] = f"theta_comp_{idx:02d}_proposer"
    out["proposer_profile"]["capabilities"] = {}
    # Boost requirement levels so gaps are large
    for skill in out["task"]["required_skills"]:
        out["task"]["required_skills"][skill] = min(
            1.0, float(out["task"]["required_skills"][skill]) + 0.15,
        )
    return out


def _make_motivation_variant(entry: dict, idx: int) -> dict:
    """Proposer covers all reqs; offers + candidate needs drive S_need."""
    out = deepcopy(entry)
    out["task"]["task_id"] = f"theta_mot_{idx:02d}"
    out["task"]["title"] = f"[motivation] {out['task'].get('title', '')}"
    out["proposer_profile"]["user_id"] = f"theta_mot_{idx:02d}_proposer"
    # Proposer already has every required skill at high level → Gap ≈ 0
    out["proposer_profile"]["capabilities"] = {
        skill: min(1.0, float(level) + 0.25)
        for skill, level in out["task"]["required_skills"].items()
    }
    # Strong offers (explicit) matching typical candidate needs
    offer_skills = list(out["task"]["required_skills"].keys())[:3]
    out["task"]["offer_skills"] = {
        s: 0.95 for s in offer_skills
    }
    for cand in out["candidates"]:
        prof = cand["candidate_profile"]
        for s in offer_skills:
            prof.setdefault("needs", {})[s] = max(
                float(prof.get("needs", {}).get(s, 0.5)), 0.85,
            )
    return out


def generate(seed: int = 42) -> dict:
    data = json.loads(TIERED.read_text())
    base_tasks = data["tasks"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(base_tasks), size=20, replace=False)

    comp = [_make_complement_variant(base_tasks[i], j + 1)
            for j, i in enumerate(indices[:10])]
    mot = [_make_motivation_variant(base_tasks[i], j + 1)
           for j, i in enumerate(indices[10:20])]

    return {
        "metadata": {
            "generated_by": "Experiments/interpretability/generate_theta_contrast_testset.py",
            "source": str(TIERED),
            "num_tasks": 20,
            "complement_dominant": 10,
            "motivation_dominant": 10,
            "random_seed": seed,
        },
        "tasks": comp + mot,
    }


def main() -> int:
    payload = generate()
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT} ({len(payload['tasks'])} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
