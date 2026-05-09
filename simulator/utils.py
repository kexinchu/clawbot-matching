"""
simulator/utils.py
==================
Shared utility functions: logging helpers, softmax, sampling, text utils.
"""
from __future__ import annotations

import json
import math
import random
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Softmax & sampling
# ---------------------------------------------------------------------------


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically-stable softmax with optional temperature."""
    if not scores:
        return []
    scores_arr = [s / temperature for s in scores]
    max_s = max(scores_arr)
    exps = [math.exp(s - max_s) for s in scores_arr]
    total = sum(exps)
    return [e / total for e in exps]


def sample_action(probs: dict[str, float]) -> str:
    """Sample an action from {accept, skip, reject} probability dict."""
    keys = list(probs.keys())
    vals = list(probs.values())
    cumsum = 0.0
    r = random.random()
    for k, v in zip(keys, vals):
        cumsum += v
        if r <= cumsum:
            return k
    return keys[-1]


def sample_action_from_probs(probs: dict[str, float]) -> Action:
    """
    Sample an Action from a {accept, skip, reject} probability distribution.

    Used when decision_mode == "probabilistic": the final action is drawn
    from the aggregated persona probability distribution rather than determined
    by a hard utility threshold.
    """
    from simulator.types import Action  # avoid circular import at module level
    sampled = sample_action(probs)
    return Action(sampled)


def weighted_sum(values: list[float], weights: list[float]) -> float:
    """Weighted sum of two same-length lists."""
    return sum(v * w for v, w in zip(values, weights))


# ---------------------------------------------------------------------------
# JSON logging
# ---------------------------------------------------------------------------


class TraceLogger:
    """Accumulates structured data during a simulator run for later export."""

    def __init__(self, verbose: bool = True):
        self.entries: list[dict[str, Any]] = []
        self.verbose = verbose

    def log(self, phase: str, data: dict[str, Any]) -> None:
        entry = {"phase": phase, "data": data}
        self.entries.append(entry)
        if self.verbose:
            indent = "  "
            print(f"{indent}[{phase}] {json.dumps(data, ensure_ascii=False, indent=2)}")

    def to_json(self) -> str:
        return json.dumps(self.entries, ensure_ascii=False, indent=2)

    def summary(self) -> dict[str, Any]:
        """Return a flat dict with one representative value per phase."""
        out = {}
        for e in self.entries:
            phase = e["phase"]
            data = e["data"]
            # Pick the most informative leaf values
            if isinstance(data, dict):
                out[phase] = {k: v for k, v in data.items()
                              if isinstance(v, (str, int, float, bool))}
            else:
                out[phase] = data
        return out


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def skill_coverage(
    required: dict[str, float],
    available: dict[str, float],
) -> float:
    """Fraction of required skills that are met by available capabilities."""
    if not required:
        return 1.0
    covered = sum(
        1 for skill, min_level in required.items()
        if available.get(skill, 0.0) >= min_level
    )
    return covered / len(required)


def need_fulfillment(
    needs: dict[str, float],
    task_metadata: dict[str, Any],
) -> float:
    """Heuristic need-fulfillment score based on task metadata alignment."""
    if not needs:
        return 0.5
    score = 0.0
    for need, intensity in needs.items():
        # Simple keyword-based heuristic
        text = str(task_metadata.get("description", "")).lower()
        if need.lower() in text:
            score += intensity
        else:
            score += intensity * 0.2  # partial credit
    return clamp(score / len(needs))


def risk_score_from_concerns(concerns: list[str]) -> float:
    """Turn a list of concern strings into a [0,1] risk score."""
    risk_keywords = {
        "risk": 0.1, "safety": 0.1, "conflict": 0.15,
        "overload": 0.15, "unclear": 0.1, "concern": 0.1,
        "incompatible": 0.2, "misalignment": 0.1, "uncertainty": 0.1,
        "overlap": 0.05, "redundancy": 0.05,
    }
    score = 0.0
    for c in concerns:
        for kw, inc in risk_keywords.items():
            if kw in c.lower():
                score += inc
    return clamp(score)
