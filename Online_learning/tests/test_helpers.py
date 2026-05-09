"""Shared path setup and fixtures for Online_learning tests."""

import sys
import os

_ol = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Online_learning/
_ma = os.path.join(_ol, '..', 'mapping-algo')
for _p in [_ol, _ma]:
    if _p not in sys.path:
        sys.path.append(_p)

import pytest
import numpy as np
from datatypes import UserState, CapabilityEntry, NeedEntry, Task, TaskRequirement, TaskOffer
from config import MatchConfig
from encoder import SimpleEncoder

ENC = SimpleEncoder(dim=64)
CFG = MatchConfig(embedding_dim=64)


def make_user(uid, caps, needs=None, clearance=0):
    """caps: list of (desc, mu, sigma, source). needs: list of (desc, intensity)."""
    return UserState(
        user_id=uid,
        capabilities=[
            CapabilityEntry(ENC(d), mu=mu, sigma=sig, source=src, description=d)
            for d, mu, sig, src in caps
        ],
        needs=[
            NeedEntry(ENC(d), intensity=inten, description=d)
            for d, inten in (needs or [])
        ],
        clearance_level=clearance,
    )


def make_task(reqs, offers=None, data_clearance=0):
    """reqs: list of (desc, level, ctype). offers: list of (desc, strength, source)."""
    return Task(
        task_id="t_test",
        goal="test task",
        requirements=[
            TaskRequirement(ENC(d), level=lvl, constraint_type=ct, description=d)
            for d, lvl, ct in reqs
        ],
        offers=[
            TaskOffer(ENC(d), strength=s, source=src, description=d)
            for d, s, src in (offers or [])
        ],
        data_clearance=data_clearance,
    )
