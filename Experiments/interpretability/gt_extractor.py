"""Ground-truth extractors for the weight interpretability experiment.

For every weight type (w_j / attention α / θ / Bayesian μ,σ) we need a
"what the persona most needs" reference signal that we can compare the
system's internal weight against. This module computes those references
directly from the 20-task tiered testset and the algorithm primitives.

All functions are read-only and side-effect free — they do not mutate
the testset or the UserState objects passed in.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_MAPPING_ALGO = _REPO_ROOT / "mapping-algo"
for _p in (str(_MAPPING_ALGO), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datatypes import (  # noqa: E402
    CapabilityEntry,
    NeedEntry,
    Task,
    TaskOffer,
    TaskRequirement,
    UserState,
)
from utils import attention_weighted_value, cosine_sim, softmax  # noqa: E402


# ---------------------------------------------------------------------------
# Per-task "most-needed" requirements
# ---------------------------------------------------------------------------

def critical_req_by_q(task: Task) -> Optional[TaskRequirement]:
    """Return the soft requirement with the largest level q_j.

    This is the trivial "task author says this is most important" signal.
    By construction the per-requirement weight w_j = q_j / Σq always picks
    this same requirement, so it is the upper bound for w_j alignment.
    """
    soft = [r for r in task.requirements if r.constraint_type == "soft"]
    if not soft:
        return None
    return max(soft, key=lambda r: r.level)


def critical_req_by_gap_weighted(
    task: Task,
    requester: UserState,
    temperature: float,
) -> Tuple[Optional[TaskRequirement], List[float]]:
    """Return the soft req with largest q_j · Gap_j and the full per-req scores.

    Gap_j = max(0, q_j - p̃_u^j) where p̃_u^j is the attention soft-match
    of req j against requester capabilities. This represents the "real"
    importance of a requirement from the requester's perspective: how
    much they actually still need it covered.
    """
    soft = [r for r in task.requirements if r.constraint_type == "soft"]
    if not soft:
        return None, []

    if requester.capabilities:
        u_embs = np.stack([c.embedding for c in requester.capabilities])
        u_mus = np.array([c.mu for c in requester.capabilities])
    else:
        u_embs = np.empty((0, soft[0].embedding.shape[0]))
        u_mus = np.array([])

    scores: List[float] = []
    for req in soft:
        p_tilde_u = attention_weighted_value(
            req.embedding, u_embs, u_mus, temperature
        )
        gap = max(0.0, req.level - p_tilde_u)
        scores.append(gap * req.level)

    best_idx = int(np.argmax(scores)) if any(s > 0 for s in scores) else -1
    if best_idx < 0:
        return None, scores
    return soft[best_idx], scores


# ---------------------------------------------------------------------------
# Critical capability / offer for attention checks
# ---------------------------------------------------------------------------

def critical_cap_for_req(
    req: TaskRequirement,
    candidate: UserState,
) -> Tuple[Optional[int], Optional[CapabilityEntry], float]:
    """For a given requirement, find the candidate capability that the
    attention SHOULD focus on.

    Definition: argmax cosine_sim(req.embedding, cap.embedding) over
    candidate.capabilities. We additionally consider exact-description
    matches as the canonical ground-truth (skills are named consistently
    across the testset).

    Returns (idx, cap, sim). idx = -1 / cap = None / sim = 0.0 if v has
    no capabilities.
    """
    caps = candidate.capabilities
    if not caps:
        return None, None, 0.0

    # Prefer exact description match — testset uses canonical skill names.
    for i, cap in enumerate(caps):
        if cap.description and cap.description == req.description:
            return i, cap, 1.0

    sims = np.array(
        [cosine_sim(req.embedding, cap.embedding) for cap in caps]
    )
    best_idx = int(np.argmax(sims))
    return best_idx, caps[best_idx], float(sims[best_idx])


def critical_offer_for_need(
    need: NeedEntry,
    task: Task,
) -> Tuple[Optional[int], Optional[TaskOffer], float]:
    """Mirror of critical_cap_for_req for the need→offer attention path."""
    offers = task.offers
    if not offers:
        return None, None, 0.0

    for i, off in enumerate(offers):
        if off.description and off.description == need.description:
            return i, off, 1.0

    sims = np.array(
        [cosine_sim(need.embedding, off.embedding) for off in offers]
    )
    best_idx = int(np.argmax(sims))
    return best_idx, offers[best_idx], float(sims[best_idx])


# ---------------------------------------------------------------------------
# Explicit attention weight tables (used by metrics + heatmap plot)
# ---------------------------------------------------------------------------

def req_to_cap_attention(
    req: TaskRequirement,
    candidate: UserState,
    temperature: float,
) -> np.ndarray:
    """Return the attention distribution α over candidate.capabilities for
    a single requirement. Mirrors the softmax inside
    `attention_weighted_value` (utils.py) but exposes the weights so we
    can inspect them.
    """
    if not candidate.capabilities:
        return np.zeros(0)
    sims = np.array(
        [cosine_sim(req.embedding, cap.embedding)
         for cap in candidate.capabilities]
    )
    return softmax(sims / temperature)


def need_to_offer_attention(
    need: NeedEntry,
    task: Task,
    temperature: float,
) -> np.ndarray:
    """Same shape for need→offer attention."""
    if not task.offers:
        return np.zeros(0)
    sims = np.array(
        [cosine_sim(need.embedding, off.embedding) for off in task.offers]
    )
    return softmax(sims / temperature)


# ---------------------------------------------------------------------------
# Bayesian μ,σ ground truth + criticality labels
# ---------------------------------------------------------------------------

def mu_true_lookup(candidate_profile: dict) -> Dict[str, float]:
    """Return {skill_description: true_mu} from the testset profile.

    `candidate_profile["capabilities"]` is the raw dict from the JSON.
    """
    caps = candidate_profile.get("capabilities", {}) or {}
    return {str(k): float(v) for k, v in caps.items()}


def label_critical_caps(
    task: Task,
    candidate: UserState,
    tau_update: float,
) -> Dict[str, bool]:
    """For a learning UserState, label each capability as critical / not.

    A capability is "critical" iff it is the argmax-sim cap for at least
    one task requirement AND that sim is >= tau_update. This mirrors
    `BayesianUpdater.update` (Parameter_update.py:30-60) exactly: per
    requirement, only the single argmax-sim cap can be updated, and only
    when its sim crosses the tau_update threshold. Using the same rule
    here ensures that "labelled critical" = "actually receives Bayesian
    updates", which is what the convergence plots are about.
    """
    caps = candidate.capabilities
    labels: Dict[str, bool] = {
        cap.description or f"cap_{i}": False
        for i, cap in enumerate(caps)
    }
    if not caps:
        return labels
    cap_embs = np.stack([c.embedding for c in caps])
    for req in task.requirements:
        sims = np.array([cosine_sim(req.embedding, e) for e in cap_embs])
        if sims.size == 0:
            continue
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) >= tau_update:
            key = caps[best_idx].description or f"cap_{best_idx}"
            labels[key] = True
    return labels


# ---------------------------------------------------------------------------
# Task-level "complement vs motivation" ratio (θ ground truth proxy)
# ---------------------------------------------------------------------------

def complement_vs_motivation_ratio(
    task: Task,
    requester: UserState,
    candidate_pool: List[UserState],
    temperature: float,
) -> Dict[str, float]:
    """Compute a per-task scalar characterising whether the task is
    'complement-dominant' (rewards lifting w_c) or 'motivation-dominant'
    (rewards lifting w_n).

    - complement_strength = Σ_j q_j · Gap_j (u perspective)
      — total complementarity demand the requester actually has.
    - motivation_strength = mean over candidates of
        Σ_l n_l · max_match(need_l, task.offers)
      — total need-offer signal across the candidate pool.

    Returns a dict with both numbers + the ratio
        ratio = complement_strength /
                (complement_strength + motivation_strength + ε)
    in [0, 1]. ratio > 0.5 means SGD on this task ought to push w_c up.
    """
    soft = [r for r in task.requirements if r.constraint_type == "soft"]
    eps = 1e-8

    if requester.capabilities:
        u_embs = np.stack([c.embedding for c in requester.capabilities])
        u_mus = np.array([c.mu for c in requester.capabilities])
    else:
        u_embs = np.empty(
            (0, soft[0].embedding.shape[0]) if soft else (0, 1)
        )
        u_mus = np.array([])

    complement_strength = 0.0
    for req in soft:
        p_tilde_u = attention_weighted_value(
            req.embedding, u_embs, u_mus, temperature
        )
        gap = max(0.0, req.level - p_tilde_u)
        complement_strength += req.level * gap

    motivation_vals: List[float] = []
    for cand in candidate_pool:
        if not cand.needs or not task.offers:
            motivation_vals.append(0.0)
            continue
        offer_embs = np.stack([o.embedding for o in task.offers])
        offer_strs = np.array([o.strength for o in task.offers])
        m = 0.0
        for need in cand.needs:
            o_tilde = attention_weighted_value(
                need.embedding, offer_embs, offer_strs, temperature
            )
            m += need.intensity * o_tilde
        motivation_vals.append(m)

    motivation_strength = float(np.mean(motivation_vals)) if motivation_vals else 0.0
    denom = complement_strength + motivation_strength + eps
    ratio = complement_strength / denom

    return {
        "complement_strength": float(complement_strength),
        "motivation_strength": float(motivation_strength),
        "ratio": float(ratio),
        "label": "complement_dominant" if ratio >= 0.5 else "motivation_dominant",
    }
