"""Layer 1: Rule-based parser — JSON/dict → UserState / Task.

Schema (user profile):
{
  "user_id": "alice",
  "clearance_level": 0,           # optional, default 0
  "capabilities": [
    {"description": "bayesian statistics", "mu": 0.8, "sigma": 0.2, "source": "explicit"},
    ...
  ],
  "needs": [
    {"description": "co-authorship opportunity", "intensity": 0.7},
    ...
  ]
}

Schema (task):
{
  "task_id": "task_001",
  "goal": "Build Bayesian churn model for NeurIPS",
  "data_clearance": 0,            # optional, default 0
  "requirements": [
    {"description": "bayesian statistics", "level": 0.8, "constraint_type": "soft"},
    ...
  ],
  "offers": [
    {"description": "research collaboration", "strength": 0.8, "source": "explicit"},
    ...
  ]
}

Defaults:
  capability.sigma       → MatchConfig.sigma_init (1.0)
  capability.source      → "explicit"
  need.intensity         → 0.5
  soft_profile           → None
  requirement.level      → 0.5
  requirement.constraint_type → "soft"
  offer.strength         → 0.5
  offer.source           → "explicit"
"""

import sys
import os
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mapping-algo')
if _p not in sys.path:
    sys.path.append(_p)

import json
from pathlib import Path
from typing import Union

from datatypes import (
    UserState, CapabilityEntry, NeedEntry,
    Task, TaskRequirement, TaskOffer,
)
from config import MatchConfig
from encoder import SimpleEncoder

_cfg = MatchConfig()
_enc = SimpleEncoder(dim=64)


def parse_user(data: Union[dict, str, Path]) -> UserState:
    """Parse a user profile dict or JSON file path → UserState."""
    if isinstance(data, (str, Path)):
        with open(data, encoding="utf-8") as f:
            data = json.load(f)

    capabilities = []
    for c in data.get("capabilities", []):
        desc = c["description"]
        capabilities.append(CapabilityEntry(
            embedding=_enc(desc),
            mu=float(c.get("mu", 0.5)),
            sigma=float(c.get("sigma", _cfg.sigma_init)),
            source=c.get("source", "explicit"),
            description=desc,
        ))

    needs = []
    for n in data.get("needs", []):
        desc = n["description"]
        needs.append(NeedEntry(
            embedding=_enc(desc),
            intensity=float(n.get("intensity", 0.5)),
            description=desc,
        ))

    return UserState(
        user_id=data["user_id"],
        capabilities=capabilities,
        needs=needs,
        clearance_level=int(data.get("clearance_level", 0)),
        soft_profile=data.get("soft_profile"),
    )


def parse_task(data: Union[dict, str, Path]) -> Task:
    """Parse a task dict or JSON file path → Task."""
    if isinstance(data, (str, Path)):
        with open(data, encoding="utf-8") as f:
            data = json.load(f)

    requirements = []
    for r in data.get("requirements", []):
        desc = r["description"]
        requirements.append(TaskRequirement(
            embedding=_enc(desc),
            level=float(r.get("level", 0.5)),
            constraint_type=r.get("constraint_type", "soft"),
            description=desc,
        ))

    offers = []
    for o in data.get("offers", []):
        desc = o["description"]
        offers.append(TaskOffer(
            embedding=_enc(desc),
            strength=float(o.get("strength", 0.5)),
            source=o.get("source", "explicit"),
            description=desc,
        ))

    return Task(
        task_id=data.get("task_id", "unknown"),
        goal=data.get("goal", ""),
        requirements=requirements,
        offers=offers,
        data_clearance=int(data.get("data_clearance", 0)),
    )
