"""Alignment metrics for the weight interpretability experiment.

Implemented in pure numpy (scipy is not installed in this environment).

Metrics:
  top1_hit                — argmax(predicted) == argmax(ground_truth) ?
  kendall_tau_b           — Kendall's tau-b on two ranked sequences
  spearman_rho            — Spearman rank correlation
  attribution_share       — Σ top contributions divided by total
  entropy                 — Shannon entropy of a distribution (in nats)
  normalised_entropy      — entropy / log(n), so 0 = perfectly peaked, 1 = uniform
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def top1_hit(predicted: Sequence[float], ground_truth: Sequence[float]) -> int:
    """Return 1 if the argmax indices agree, else 0."""
    if len(predicted) == 0 or len(ground_truth) == 0:
        return 0
    return int(int(np.argmax(predicted)) == int(np.argmax(ground_truth)))


def _ranks_avg_ties(x: np.ndarray) -> np.ndarray:
    """Return ranks of x with ties averaged (used by Spearman)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average tied ranks
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j)
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks + 1.0  # 1-based


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's τ-b on two sequences. Returns 0.0 when undefined."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    n = len(a)
    if n < 2 or len(b) != n:
        return 0.0
    concord = 0
    discord = 0
    tie_a = 0
    tie_b = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                tie_a += 1
                continue
            if db == 0:
                tie_b += 1
                continue
            if (da > 0 and db > 0) or (da < 0 and db < 0):
                concord += 1
            else:
                discord += 1
    n_pairs = n * (n - 1) / 2.0
    denom = np.sqrt((n_pairs - tie_a) * (n_pairs - tie_b))
    if denom <= 0:
        return 0.0
    return float((concord - discord) / denom)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (ties averaged)."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) < 2 or len(b) != len(a):
        return 0.0
    ra = _ranks_avg_ties(a)
    rb = _ranks_avg_ties(b)
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean()
    denom = np.sqrt((ra_c ** 2).sum() * (rb_c ** 2).sum())
    if denom <= 0:
        return 0.0
    return float((ra_c * rb_c).sum() / denom)


def attribution_share(contributions: Sequence[float], k: int = 1) -> float:
    """Fraction of total mass owned by the top-k entries.

    Useful for asking "of the total MapScore, what fraction did the
    single most-needed requirement contribute?"
    """
    arr = np.asarray(contributions, dtype=float)
    total = float(np.sum(np.abs(arr)))
    if total <= 0:
        return 0.0
    top = np.sort(np.abs(arr))[::-1][:k]
    return float(np.sum(top) / total)


def entropy(p: Sequence[float]) -> float:
    """Shannon entropy in nats; expects a probability distribution."""
    arr = np.asarray(p, dtype=float)
    if arr.size == 0:
        return 0.0
    s = arr.sum()
    if s <= 0:
        return 0.0
    q = arr / s
    nz = q[q > 0]
    return float(-(nz * np.log(nz)).sum())


def normalised_entropy(p: Sequence[float]) -> float:
    """Entropy divided by log(n). 0 = perfectly peaked, 1 = uniform."""
    arr = np.asarray(p, dtype=float)
    n = arr.size
    if n <= 1:
        return 0.0
    h = entropy(arr)
    return float(h / np.log(n))
