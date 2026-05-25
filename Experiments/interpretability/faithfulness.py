"""Faithfulness-centered explanation utilities for 1-1 MapScore matching.

This module re-uses the production scoring primitives in `mapping-algo`
(via the same ExperimentContext / WorldModel that `run_interpretability`
uses) and exposes a small set of read-only helpers that answer concrete
explanation questions:

  * `explain_match_decomposition`     — why was M(u,v,T) what it was?
  * `perturb_*`                       — what happens if we strip a single
                                        capability / offer / strength?
  * `rank_after_perturbation`         — re-rank the whole pool under one
                                        perturbation spec.
  * `deletion_faithfulness_test`      — does the factor that the
                                        explanation calls 'most important'
                                        cause the biggest *score* drop —
                                        and, when the pool is supplied,
                                        the biggest *rank* change?
  * `shuffled_deletion_baseline`      — does the true top factor beat
                                        random "shuffled explanation"
                                        factors at causing drops / flips?
  * `perturbation_oracle_baseline`    — does the explanation's top
                                        factor coincide with the
                                        *actually most-impactful*
                                        single perturbation (the
                                        SHAP/LIME-style brute-force
                                        ceiling)?
  * `pairwise_contrastive_explanation`— why is winner ranked above loser?
  * `counterfactual_rank_flip`        — what is the *minimum* edit that
                                        would let loser overtake winner?

All functions are side-effect free — they deepcopy any UserState/Task
they modify so that the caller's objects are never mutated. All scoring
calls use `use_ucb=False` (no exploration bonus) so the explanation is
about the *current best estimate* of M.

The module deliberately knows nothing about LLMs, dreaming, or weight
learning; it only inspects what `compute_match_score` actually computes.
That is the point — a *faithful* explanation has to be derived from the
same numbers the production scorer uses, not from a paraphrase of them.
"""

from __future__ import annotations

import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make sure mapping-algo / Online_learning are importable when this file is
# loaded standalone (e.g. by test harnesses).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_MAPPING_ALGO = _REPO_ROOT / "mapping-algo"
_ONLINE = _REPO_ROOT / "Online_learning"
for _p in (str(_MAPPING_ALGO), str(_ONLINE), str(_HERE), str(_REPO_ROOT)):
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
from scoring import compute_s_cap, compute_s_need  # noqa: E402
from utils import attention_weighted_value, cosine_sim, softmax  # noqa: E402
from WorldModel import WorldModel  # noqa: E402


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_M(
    requester: UserState,
    candidate: UserState,
    task: Task,
    world_model: WorldModel,
) -> float:
    """Always use no-UCB MapScore for explanation purposes."""
    r = world_model.compute_match(
        requester, candidate, task, use_ucb=False, round_t=1,
    )
    return float(r.match_score)


def _best_capability_for_requirement(
    req: TaskRequirement,
    candidate: UserState,
) -> Tuple[Optional[int], Optional[CapabilityEntry], float, float]:
    """Return (index, cap, attention_weight, cos_sim) of the candidate
    capability that the soft-attention puts most mass on for this req.

    The "best" capability is `argmax_i softmax(sim_i / τ)` — equivalently
    `argmax_i sim_i`, since softmax preserves order. Exact-description
    matches take priority because the testsets use canonical skill names.
    Returns (None, None, 0.0, 0.0) for candidates with no capabilities.
    """
    caps = candidate.capabilities
    if not caps:
        return None, None, 0.0, 0.0

    for i, cap in enumerate(caps):
        if cap.description and req.description and cap.description == req.description:
            return i, cap, 1.0, 1.0

    sims = np.array([cosine_sim(req.embedding, c.embedding) for c in caps])
    best_idx = int(np.argmax(sims))
    return best_idx, caps[best_idx], float(sims[best_idx]), float(sims[best_idx])


def _best_offer_for_need(
    need: NeedEntry,
    task: Task,
) -> Tuple[Optional[int], Optional[TaskOffer], float]:
    """Same as above but for need→offer attention."""
    offers = task.offers
    if not offers:
        return None, None, 0.0

    for i, off in enumerate(offers):
        if off.description and need.description and off.description == need.description:
            return i, off, 1.0

    sims = np.array([cosine_sim(need.embedding, o.embedding) for o in offers])
    best_idx = int(np.argmax(sims))
    return best_idx, offers[best_idx], float(sims[best_idx])


def _safe_div(num: float, denom: float, fallback: float = 0.0) -> float:
    """Division that returns `fallback` when the denominator is ~0."""
    if abs(denom) < 1e-12:
        return float(fallback)
    return float(num / denom)


# ---------------------------------------------------------------------------
# 2.1 — decomposition
# ---------------------------------------------------------------------------

def explain_match_decomposition(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
) -> Dict[str, Any]:
    """Decompose M(u, v, T) = σ · (w_c · S_cap + w_n · S_need) into the
    *per-requirement* and *per-need* contributions that actually drive
    the score.

    The math mirrors `scoring.compute_s_cap` / `compute_s_need` exactly:
      requirement j contributes (coverage_j · w_j) to the S_cap numerator
        and  (gap_j · w_j) to its denominator;
      need l contributes (satisfied_l · n_l) to the S_need numerator
        and  (n_l²) to its denominator;
      after S_cap and S_need are formed the soft-max weights w_c, w_n
        scale them into the final M.

    The returned dict's `top_positive_factors` is a single ranked list of
    requirement + need factors sorted by their absolute contribution to
    M, so downstream code (e.g. the deletion faithfulness test) can pick
    "the explanation's most important reason" without re-doing the math.
    """
    cfg = world_model.config
    eps = cfg.epsilon

    # ── overall score (no UCB) ───────────────────────────────────────
    full = world_model.compute_match(u, v, task, use_ucb=False, round_t=1)
    M = float(full.match_score)
    sigma_gate = int(full.sigma_gate)
    s_cap = float(full.s_cap)
    s_need = float(full.s_need)
    w_c = float(full.w_c)
    w_n = float(full.w_n)

    # u / v capability matrices used inside compute_s_cap
    if u.capabilities:
        u_embs = np.stack([c.embedding for c in u.capabilities])
        u_mus = np.array([c.mu for c in u.capabilities])
    else:
        u_embs = np.empty((0, cfg.embedding_dim))
        u_mus = np.array([])
    if v.capabilities:
        v_embs = np.stack([c.embedding for c in v.capabilities])
        v_mus = np.array([c.mu for c in v.capabilities])
    else:
        v_embs = np.empty((0, cfg.embedding_dim))
        v_mus = np.array([])

    soft_reqs = [r for r in task.requirements if r.constraint_type == "soft"]

    # weights w_j = q_j / Σq  (mirrors compute_s_cap)
    if soft_reqs:
        q_vals = np.array([r.level for r in soft_reqs])
        w_j_arr = q_vals / (q_vals.sum() + eps)
    else:
        w_j_arr = np.array([])

    # ── per-requirement contributions ────────────────────────────────
    req_num_total = 0.0
    req_den_total = 0.0
    req_contribs_intermediate: List[Dict[str, Any]] = []
    for idx, req in enumerate(soft_reqs):
        p_tilde_u = attention_weighted_value(
            req.embedding, u_embs, u_mus, cfg.temperature,
        )
        gap = max(0.0, req.level - p_tilde_u)
        p_tilde_v = attention_weighted_value(
            req.embedding, v_embs, v_mus, cfg.temperature,
        )
        coverage = min(p_tilde_v, gap)
        w_j = float(w_j_arr[idx])
        s_cap_num = float(coverage * w_j)
        s_cap_den = float(gap * w_j)
        req_num_total += s_cap_num
        req_den_total += s_cap_den

        best_idx, best_cap, _, _ = _best_capability_for_requirement(req, v)
        best_cap_name = best_cap.description if best_cap is not None else None
        best_cap_mu = float(best_cap.mu) if best_cap is not None else None

        req_contribs_intermediate.append({
            "factor_type": "requirement",
            "name": req.description,
            "q_j": float(req.level),
            "w_j": w_j,
            "p_tilde_u": float(p_tilde_u),
            "gap": float(gap),
            "p_tilde_v": float(p_tilde_v),
            "coverage": float(coverage),
            "s_cap_numerator_contribution": s_cap_num,
            "s_cap_denominator_contribution": s_cap_den,
            "best_capability": best_cap_name,
            "best_capability_mu": best_cap_mu,
        })

    requirement_contributions: List[Dict[str, Any]] = []
    for c in req_contribs_intermediate:
        # contribution to S_cap is normalised by the full S_cap denominator
        # so Σ contribution_to_S_cap = S_cap (when denom > 0).
        contrib_to_s_cap = _safe_div(
            c["s_cap_numerator_contribution"], req_den_total, fallback=0.0,
        )
        contrib_to_M = float(sigma_gate) * w_c * contrib_to_s_cap
        c["contribution_to_S_cap"] = float(contrib_to_s_cap)
        c["contribution_to_M"] = float(contrib_to_M)
        requirement_contributions.append(c)

    # ── per-need contributions ───────────────────────────────────────
    # Replicates compute_s_need: offers with source='inferred' are
    # discounted by alpha_infer before contributing to the matched value.
    offer_list = []
    for offer in task.offers:
        weight = offer.strength * (
            cfg.alpha_infer if offer.source == "inferred" else 1.0
        )
        offer_list.append((offer.embedding, weight))
    if offer_list:
        offer_embs = np.stack([o[0] for o in offer_list])
        offer_strs = np.array([o[1] for o in offer_list])
    else:
        offer_embs = np.empty((0, cfg.embedding_dim))
        offer_strs = np.array([])

    need_num_total = 0.0
    need_den_total = 0.0
    need_contribs_intermediate: List[Dict[str, Any]] = []
    for need in v.needs:
        o_tilde = attention_weighted_value(
            need.embedding, offer_embs, offer_strs, cfg.temperature,
        )
        satisfied = min(o_tilde, need.intensity)
        s_need_num = float(satisfied * need.intensity)
        s_need_den = float(need.intensity ** 2)
        need_num_total += s_need_num
        need_den_total += s_need_den

        best_idx, best_off, _ = _best_offer_for_need(need, task)
        best_off_name = best_off.description if best_off is not None else None
        best_off_strength = float(best_off.strength) if best_off is not None else None

        need_contribs_intermediate.append({
            "factor_type": "need",
            "name": need.description,
            "intensity": float(need.intensity),
            "offer_matched": float(o_tilde),
            "satisfied": float(satisfied),
            "s_need_numerator_contribution": s_need_num,
            "s_need_denominator_contribution": s_need_den,
            "best_offer": best_off_name,
            "best_offer_strength": best_off_strength,
        })

    need_contributions: List[Dict[str, Any]] = []
    for c in need_contribs_intermediate:
        contrib_to_s_need = _safe_div(
            c["s_need_numerator_contribution"], need_den_total, fallback=0.0,
        )
        contrib_to_M = float(sigma_gate) * w_n * contrib_to_s_need
        c["contribution_to_S_need"] = float(contrib_to_s_need)
        c["contribution_to_M"] = float(contrib_to_M)
        need_contributions.append(c)

    # ── top positive factors (single ranked list) ────────────────────
    all_factors: List[Dict[str, Any]] = []
    for c in requirement_contributions:
        all_factors.append({
            "factor_type": "requirement",
            "name": c["name"],
            "contribution_to_M": c["contribution_to_M"],
            "linked_capability_or_offer": c.get("best_capability"),
        })
    for c in need_contributions:
        all_factors.append({
            "factor_type": "need",
            "name": c["name"],
            "contribution_to_M": c["contribution_to_M"],
            "linked_capability_or_offer": c.get("best_offer"),
        })
    top_positive_factors = sorted(
        [f for f in all_factors if f["contribution_to_M"] > 0],
        key=lambda f: f["contribution_to_M"],
        reverse=True,
    )

    return {
        "candidate_id": v.user_id,
        "M": M,
        "sigma_gate": sigma_gate,
        "S_cap": s_cap,
        "S_need": s_need,
        "w_c": w_c,
        "w_n": w_n,
        "requirement_contributions": requirement_contributions,
        "need_contributions": need_contributions,
        "top_positive_factors": top_positive_factors,
    }


# ---------------------------------------------------------------------------
# 2.2 — remove capability
# ---------------------------------------------------------------------------

def perturb_candidate_remove_capability(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
    cap_description: str,
) -> Dict[str, Any]:
    """Deepcopy v, drop the capability with matching description, then
    recompute M. Returns (M_before, M_after, drop). `valid=False` when
    the capability is not on v.
    """
    M_before = _score_M(u, v, task, world_model)
    v_copy = deepcopy(v)
    new_caps = [c for c in v_copy.capabilities if c.description != cap_description]
    if len(new_caps) == len(v_copy.capabilities):
        return {
            "removed_capability": cap_description,
            "M_before": M_before,
            "M_after": M_before,
            "drop": 0.0,
            "valid": False,
        }
    v_copy.capabilities = new_caps
    M_after = _score_M(u, v_copy, task, world_model)
    return {
        "removed_capability": cap_description,
        "M_before": M_before,
        "M_after": M_after,
        "drop": float(M_before - M_after),
        "valid": True,
    }


# ---------------------------------------------------------------------------
# 2.3 — remove offer
# ---------------------------------------------------------------------------

def perturb_task_remove_offer(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
    offer_description: str,
) -> Dict[str, Any]:
    """Deepcopy task, drop the offer with matching description, then
    recompute M. `valid=False` when no offer matches.
    """
    M_before = _score_M(u, v, task, world_model)
    task_copy = deepcopy(task)
    new_offers = [o for o in task_copy.offers if o.description != offer_description]
    if len(new_offers) == len(task_copy.offers):
        return {
            "removed_offer": offer_description,
            "M_before": M_before,
            "M_after": M_before,
            "drop": 0.0,
            "valid": False,
        }
    task_copy.offers = new_offers
    M_after = _score_M(u, v, task_copy, world_model)
    return {
        "removed_offer": offer_description,
        "M_before": M_before,
        "M_after": M_after,
        "drop": float(M_before - M_after),
        "valid": True,
    }


# ---------------------------------------------------------------------------
# 2.4 — reduce capability μ
# ---------------------------------------------------------------------------

def perturb_candidate_reduce_capability_mu(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
    cap_description: str,
    new_mu: float = 0.0,
) -> Dict[str, Any]:
    """Deepcopy v, lower one capability's μ to `new_mu`, recompute M.

    Softer than `perturb_candidate_remove_capability`: it leaves the
    capability slot in place (so attention weights don't reshape) and
    just dials its proficiency down. That makes the deletion test
    closer to a calibrated counterfactual.
    """
    M_before = _score_M(u, v, task, world_model)
    v_copy = deepcopy(v)
    target_idx: Optional[int] = None
    old_mu: Optional[float] = None
    for i, c in enumerate(v_copy.capabilities):
        if c.description == cap_description:
            target_idx = i
            old_mu = float(c.mu)
            break
    if target_idx is None:
        return {
            "capability": cap_description,
            "old_mu": None,
            "new_mu": float(new_mu),
            "M_before": M_before,
            "M_after": M_before,
            "drop": 0.0,
            "valid": False,
        }
    v_copy.capabilities[target_idx].mu = float(new_mu)
    M_after = _score_M(u, v_copy, task, world_model)
    return {
        "capability": cap_description,
        "old_mu": old_mu,
        "new_mu": float(new_mu),
        "M_before": M_before,
        "M_after": M_after,
        "drop": float(M_before - M_after),
        "valid": True,
    }


# ---------------------------------------------------------------------------
# 2.5 — reduce offer strength
# ---------------------------------------------------------------------------

def perturb_task_reduce_offer_strength(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
    offer_description: str,
    new_strength: float = 0.0,
) -> Dict[str, Any]:
    """Deepcopy the task, lower one offer's strength to `new_strength`,
    recompute M. Mirror of `perturb_candidate_reduce_capability_mu`.
    """
    M_before = _score_M(u, v, task, world_model)
    task_copy = deepcopy(task)
    target_idx: Optional[int] = None
    old_strength: Optional[float] = None
    for i, o in enumerate(task_copy.offers):
        if o.description == offer_description:
            target_idx = i
            old_strength = float(o.strength)
            break
    if target_idx is None:
        return {
            "offer": offer_description,
            "old_strength": None,
            "new_strength": float(new_strength),
            "M_before": M_before,
            "M_after": M_before,
            "drop": 0.0,
            "valid": False,
        }
    task_copy.offers[target_idx].strength = float(new_strength)
    M_after = _score_M(u, v, task_copy, world_model)
    return {
        "offer": offer_description,
        "old_strength": old_strength,
        "new_strength": float(new_strength),
        "M_before": M_before,
        "M_after": M_after,
        "drop": float(M_before - M_after),
        "valid": True,
    }


# ---------------------------------------------------------------------------
# 2.6a — rank-level perturbation
# ---------------------------------------------------------------------------

def _factor_to_perturbation_spec(
    v: UserState,
    factor: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Translate a `top_positive_factors` entry into the canonical
    perturbation spec consumed by `rank_after_perturbation`.

    Mirrors the policy from `_perturb_factor`: requirement → reduce the
    linked capability's μ to 0 (fallback: name-match remove); need →
    reduce the linked offer's strength to 0 (fallback: name-match
    remove). Returns None when neither route can fire.
    """
    factor_type = factor.get("factor_type")
    linked = factor.get("linked_capability_or_offer")
    name = factor.get("name")
    if factor_type == "requirement":
        target = linked or name
        if not target:
            return None
        return {
            "type": "reduce_capability_mu",
            "candidate_id": v.user_id,
            "capability": target,
            "new_mu": 0.0,
        }
    if factor_type == "need":
        target = linked or name
        if not target:
            return None
        return {
            "type": "reduce_offer_strength",
            "offer": target,
            "new_strength": 0.0,
        }
    return None


def rank_after_perturbation(
    u: UserState,
    task: Task,
    candidate_pool: List[UserState],
    world_model: WorldModel,
    perturbation_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-rank `candidate_pool` after applying a single perturbation.

    `perturbation_spec` types currently supported:

      * ``{"type": "reduce_capability_mu", "candidate_id", "capability",
            "new_mu"}`` — lowers one capability's μ on one specific
        candidate. Only that candidate's score moves.
      * ``{"type": "reduce_offer_strength", "offer", "new_strength"}``
        — lowers one offer's strength on the task itself. All
        candidates' S_need scores can move.

    The returned dict contains both the original ranking (so callers
    can diff without re-scoring themselves) and the post-perturbation
    ranking, plus rank-flip indicators. `valid=False` is returned for
    specs that don't match anything; in that case `new_ranking` echoes
    `original_ranking`.
    """
    # ── score everyone before perturbation ───────────────────────────
    original_scores: List[Tuple[str, float]] = []
    for cand in candidate_pool:
        m = _score_M(u, cand, task, world_model)
        original_scores.append((cand.user_id, m))
    original_sorted = sorted(original_scores, key=lambda x: x[1], reverse=True)
    original_ranking = [
        {"candidate_id": cid, "M": float(m), "rank": i + 1}
        for i, (cid, m) in enumerate(original_sorted)
    ]
    original_winner_id = original_sorted[0][0] if original_sorted else None

    ptype = perturbation_spec.get("type")
    valid = False
    new_pool = candidate_pool
    new_task = task

    if ptype == "reduce_capability_mu":
        target_cid = perturbation_spec.get("candidate_id")
        target_cap = perturbation_spec.get("capability")
        new_mu = float(perturbation_spec.get("new_mu", 0.0))
        new_pool = []
        for cand in candidate_pool:
            if cand.user_id == target_cid:
                cand_copy = deepcopy(cand)
                hit = False
                for i, c in enumerate(cand_copy.capabilities):
                    if c.description == target_cap:
                        cand_copy.capabilities[i].mu = new_mu
                        hit = True
                        break
                if hit:
                    valid = True
                new_pool.append(cand_copy)
            else:
                new_pool.append(cand)

    elif ptype == "reduce_offer_strength":
        target_off = perturbation_spec.get("offer")
        new_strength = float(perturbation_spec.get("new_strength", 0.0))
        task_copy = deepcopy(task)
        for i, o in enumerate(task_copy.offers):
            if o.description == target_off:
                task_copy.offers[i].strength = new_strength
                valid = True
                break
        if valid:
            new_task = task_copy

    if not valid:
        # Echo the original ranking — nothing changed.
        return {
            "valid": False,
            "original_ranking": original_ranking,
            "new_ranking": deepcopy(original_ranking),
            "original_winner_id": original_winner_id,
            "new_winner_id": original_winner_id,
            "winner_rank_after": 1 if original_winner_id else None,
            "rank_changed": False,
            "winner_dropped_from_top1": False,
            "perturbation_spec": perturbation_spec,
        }

    new_scores: List[Tuple[str, float]] = []
    for cand in new_pool:
        m = _score_M(u, cand, new_task, world_model)
        new_scores.append((cand.user_id, m))
    new_sorted = sorted(new_scores, key=lambda x: x[1], reverse=True)
    new_ranking = [
        {"candidate_id": cid, "M": float(m), "rank": i + 1}
        for i, (cid, m) in enumerate(new_sorted)
    ]
    new_winner_id = new_sorted[0][0] if new_sorted else None
    winner_rank_after = next(
        (entry["rank"] for entry in new_ranking
         if entry["candidate_id"] == original_winner_id),
        None,
    )

    original_ids = [e["candidate_id"] for e in original_ranking]
    new_ids = [e["candidate_id"] for e in new_ranking]
    rank_changed = original_ids != new_ids

    return {
        "valid": True,
        "original_ranking": original_ranking,
        "new_ranking": new_ranking,
        "original_winner_id": original_winner_id,
        "new_winner_id": new_winner_id,
        "winner_rank_after": winner_rank_after,
        "rank_changed": bool(rank_changed),
        "winner_dropped_from_top1": bool(
            new_winner_id is not None
            and original_winner_id is not None
            and new_winner_id != original_winner_id
        ),
        "perturbation_spec": perturbation_spec,
    }


# ---------------------------------------------------------------------------
# 2.6 — deletion faithfulness test
# ---------------------------------------------------------------------------

def _perturb_factor(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
    factor: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the canonical 'weaken this factor' edit for a single
    requirement or need entry from `top_positive_factors`.

    For a requirement, the canonical edit is to dial its linked
    capability's μ down to 0 (preferred, smoother than deletion).
    For a need, the canonical edit is to dial its linked offer's
    strength down to 0. If no linked target exists, we fall back to
    removing the entity with the same name as the factor (best effort —
    callers should treat `valid=False` as "no test possible").
    """
    factor_type = factor["factor_type"]
    linked = factor.get("linked_capability_or_offer")
    if factor_type == "requirement":
        if linked:
            return perturb_candidate_reduce_capability_mu(
                u, v, task, world_model, linked, new_mu=0.0,
            )
        return perturb_candidate_remove_capability(
            u, v, task, world_model, factor["name"],
        )
    elif factor_type == "need":
        if linked:
            return perturb_task_reduce_offer_strength(
                u, v, task, world_model, linked, new_strength=0.0,
            )
        return perturb_task_remove_offer(
            u, v, task, world_model, factor["name"],
        )
    return {"valid": False, "drop": 0.0}


def _rank_effect_for_factor(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
    candidate_pool: Optional[List[UserState]],
    factor: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the rank-level effect of weakening `factor`. Returns a
    compact summary (winner_rank_after / dropped_from_top1 / new winner).
    `valid=False` when no pool is provided or the perturbation spec
    cannot be built / applied.
    """
    if candidate_pool is None:
        return {"valid": False, "reason": "no_candidate_pool"}
    spec = _factor_to_perturbation_spec(v, factor)
    if spec is None:
        return {"valid": False, "reason": "no_perturbation_spec"}
    rp = rank_after_perturbation(
        u, task, candidate_pool, world_model, spec,
    )
    if not rp.get("valid"):
        return {
            "valid": False,
            "reason": "perturbation_not_applied",
            "perturbation_spec": spec,
        }
    return {
        "valid": True,
        "winner_rank_after": rp["winner_rank_after"],
        "winner_dropped_from_top1": rp["winner_dropped_from_top1"],
        "new_winner_id": rp["new_winner_id"],
        "rank_changed": rp["rank_changed"],
        "perturbation_spec": spec,
    }


def deletion_faithfulness_test(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
    candidate_pool: Optional[List[UserState]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Validate that the factor the explanation calls 'most important'
    actually moves M the most when weakened.

    We knock down three factors: the top, a deterministically sampled
    random factor, and the bottom (smallest positive contribution).
    A faithful explanation should produce:
        drop(top) ≥ drop(random)   and   drop(top) ≥ drop(bottom)
    on average. The `*_beats_*` booleans and ratios let the caller
    aggregate this across the whole testset.

    When ``candidate_pool`` is provided, we additionally record
    rank-level effects ("did weakening this factor knock the winner
    out of Top-1?"). When omitted, the rank-level fields are returned
    with ``valid=False`` so the schema stays stable.

    `valid=False` is returned when the candidate has fewer than 1
    positive factor (the decomposition is degenerate).
    """
    decomp = explain_match_decomposition(u, v, task, world_model)
    factors = list(decomp.get("top_positive_factors", []))

    if not factors:
        return {
            "valid": False,
            "candidate_id": v.user_id,
            "reason": "no_positive_factors",
        }

    rng = random.Random(seed)

    top_factor = factors[0]
    bottom_factor = factors[-1]
    if len(factors) >= 3:
        # exclude top and bottom from random pool when possible
        middle = factors[1:-1]
        random_factor = rng.choice(middle)
    elif len(factors) == 2:
        random_factor = factors[1]  # = bottom
    else:
        random_factor = factors[0]  # = top

    top_pert = _perturb_factor(u, v, task, world_model, top_factor)
    random_pert = _perturb_factor(u, v, task, world_model, random_factor)
    bottom_pert = _perturb_factor(u, v, task, world_model, bottom_factor)

    top_drop = float(top_pert.get("drop", 0.0)) if top_pert.get("valid") else 0.0
    random_drop = float(random_pert.get("drop", 0.0)) if random_pert.get("valid") else 0.0
    bottom_drop = float(bottom_pert.get("drop", 0.0)) if bottom_pert.get("valid") else 0.0

    top_rank = _rank_effect_for_factor(
        u, v, task, world_model, candidate_pool, top_factor,
    )
    random_rank = _rank_effect_for_factor(
        u, v, task, world_model, candidate_pool, random_factor,
    )
    bottom_rank = _rank_effect_for_factor(
        u, v, task, world_model, candidate_pool, bottom_factor,
    )

    def _flip(rank_effect: Dict[str, Any]) -> bool:
        return bool(
            rank_effect.get("valid")
            and rank_effect.get("winner_dropped_from_top1")
        )

    eps = 1e-12

    def _ratio(a: float, b: float) -> Optional[float]:
        if abs(b) < eps:
            return None
        return float(a / b)

    return {
        "valid": True,
        "candidate_id": v.user_id,
        "top_factor": top_factor,
        "random_factor": random_factor,
        "bottom_factor": bottom_factor,
        "top_factor_drop": top_drop,
        "random_factor_drop": random_drop,
        "bottom_factor_drop": bottom_drop,
        "top_beats_random": bool(top_drop >= random_drop),
        "top_beats_bottom": bool(top_drop >= bottom_drop),
        "drop_ratio_top_vs_random": _ratio(top_drop, random_drop),
        "drop_ratio_top_vs_bottom": _ratio(top_drop, bottom_drop),
        "top_perturbation": top_pert,
        "random_perturbation": random_pert,
        "bottom_perturbation": bottom_pert,
        "top_rank_effect": top_rank,
        "random_rank_effect": random_rank,
        "bottom_rank_effect": bottom_rank,
        "top_causes_rank_flip": _flip(top_rank),
        "random_causes_rank_flip": _flip(random_rank),
        "bottom_causes_rank_flip": _flip(bottom_rank),
    }


# ---------------------------------------------------------------------------
# 2.6b — shuffled-explanation deletion baseline
# ---------------------------------------------------------------------------

def shuffled_deletion_baseline(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
    candidate_pool: Optional[List[UserState]] = None,
    seed: int = 42,
    n_trials: int = 20,
) -> Dict[str, Any]:
    """Test whether the explanation's ranking is informative by shuffling.

    The faithfulness test only shows that the *top* factor causes a
    big drop. A skeptic could argue that *any* factor in our list
    causes a big drop and we just happen to label one "top". This
    baseline knocks down a *uniformly sampled* factor n_trials times
    and reports how the true top compares to that null distribution.

    Returns a dict with:
      * ``mean_shuffled_top_drop``, ``std_shuffled_top_drop`` — score
        deltas from the random picks;
      * ``frac_shuffled_rank_flip`` — fraction of random picks that
        kicked the winner out of Top-1 (only meaningful when
        ``candidate_pool`` was given);
      * ``true_top_drop`` — the actual top-factor drop;
      * ``true_top_drop_percentile`` — fraction of shuffled trials with
        drop ≤ true_top_drop. 1.0 means the true top beat every random
        pick.

    `valid=False` is returned when the decomposition has no positive
    factors (nothing to shuffle).
    """
    decomp = explain_match_decomposition(u, v, task, world_model)
    factors = list(decomp.get("top_positive_factors", []))
    if not factors:
        return {"valid": False, "reason": "no_positive_factors"}

    top_factor = factors[0]
    top_pert = _perturb_factor(u, v, task, world_model, top_factor)
    true_top_drop = (
        float(top_pert.get("drop", 0.0)) if top_pert.get("valid") else 0.0
    )
    if candidate_pool is not None:
        true_top_rank = _rank_effect_for_factor(
            u, v, task, world_model, candidate_pool, top_factor,
        )
        true_top_flip = bool(
            true_top_rank.get("valid")
            and true_top_rank.get("winner_dropped_from_top1")
        )
    else:
        true_top_flip = False

    rng = random.Random(seed + 1)
    shuffled_drops: List[float] = []
    shuffled_flips: List[bool] = []
    for _ in range(int(n_trials)):
        f = rng.choice(factors)
        pert = _perturb_factor(u, v, task, world_model, f)
        d = float(pert.get("drop", 0.0)) if pert.get("valid") else 0.0
        shuffled_drops.append(d)
        if candidate_pool is not None:
            rank_eff = _rank_effect_for_factor(
                u, v, task, world_model, candidate_pool, f,
            )
            shuffled_flips.append(bool(
                rank_eff.get("valid")
                and rank_eff.get("winner_dropped_from_top1")
            ))
        else:
            shuffled_flips.append(False)

    drops_arr = np.asarray(shuffled_drops, dtype=float)
    mean_shuf = float(drops_arr.mean()) if drops_arr.size > 0 else 0.0
    std_shuf = float(drops_arr.std()) if drops_arr.size > 0 else 0.0
    frac_flip = (
        float(sum(1 for f in shuffled_flips if f) / len(shuffled_flips))
        if shuffled_flips else 0.0
    )
    if drops_arr.size > 0:
        percentile = float(np.mean(drops_arr <= true_top_drop + 1e-12))
    else:
        percentile = 0.0

    return {
        "valid": True,
        "n_trials": int(n_trials),
        "candidate_id": v.user_id,
        "true_top_factor": top_factor,
        "true_top_drop": true_top_drop,
        "true_top_caused_rank_flip": true_top_flip,
        "mean_shuffled_top_drop": mean_shuf,
        "std_shuffled_top_drop": std_shuf,
        "frac_shuffled_rank_flip": frac_flip,
        "true_top_beats_shuffled_mean": bool(true_top_drop >= mean_shuf),
        "true_top_drop_percentile": percentile,
    }


# ---------------------------------------------------------------------------
# 2.6c — perturbation oracle (SHAP/LIME-style brute-force ceiling)
# ---------------------------------------------------------------------------

def perturbation_oracle_baseline(
    u: UserState,
    v: UserState,
    task: Task,
    world_model: WorldModel,
    candidate_pool: Optional[List[UserState]] = None,
) -> Dict[str, Any]:
    """Brute-force enumerate every single-factor perturbation we can apply
    to (v, task) and compute the resulting score drop + rank effect.

    This is the SHAP/LIME-style upper bound: instead of trusting the
    decomposition's `contribution_to_M` ordering, we just *measure* what
    each individual edit does. The oracle's "top perturbation" is the
    factor that — empirically — moves M the most. We then ask:

      * Does our explanation's top factor *match* the oracle's top?
      * If not, how far down the perturbation-impact ranking is the
        explanation's pick?
      * How does the explanation's drop compare to the oracle's drop?

    A faithful explanation should land near rank 1 on the oracle list.
    Disagreements are *exactly* the cases where attribution-share
    (what the explanation reports) and perturbation-sensitivity (what
    really matters causally) come apart — the off-topic-need failure
    mode is the canonical example.

    Notes:
      * We enumerate v.capabilities (reduce μ to 0) and task.offers
        (reduce strength to 0). These are the same atomic edits the
        deletion / counterfactual tests use, so the comparison is
        apples-to-apples.
      * Rank effects are computed when `candidate_pool` is given; that
        also lets us report whether the explanation's pick caused a flip
        the oracle missed (or vice versa).
    """
    decomp = explain_match_decomposition(u, v, task, world_model)
    expl_factors = list(decomp.get("top_positive_factors", []))
    if not expl_factors:
        return {"valid": False, "reason": "no_positive_factors"}
    expl_top_factor = expl_factors[0]
    expl_top_pert = _perturb_factor(u, v, task, world_model, expl_top_factor)
    expl_top_drop = (
        float(expl_top_pert.get("drop", 0.0))
        if expl_top_pert.get("valid") else 0.0
    )

    M_before = _score_M(u, v, task, world_model)

    # ── enumerate atomic edits ───────────────────────────────────────
    candidates: List[Dict[str, Any]] = []

    # 1) every capability on v: drive μ to 0
    for cap in v.capabilities:
        if not cap.description:
            continue
        pert = perturb_candidate_reduce_capability_mu(
            u, v, task, world_model, cap.description, new_mu=0.0,
        )
        if not pert.get("valid"):
            continue
        rank_eff: Dict[str, Any] = {"valid": False}
        if candidate_pool is not None:
            spec = {
                "type": "reduce_capability_mu",
                "candidate_id": v.user_id,
                "capability": cap.description,
                "new_mu": 0.0,
            }
            rp = rank_after_perturbation(
                u, task, candidate_pool, world_model, spec,
            )
            rank_eff = {
                "valid": rp.get("valid", False),
                "winner_rank_after": rp.get("winner_rank_after"),
                "winner_dropped_from_top1": rp.get("winner_dropped_from_top1"),
                "new_winner_id": rp.get("new_winner_id"),
            }
        candidates.append({
            "factor_type": "capability",
            "name": cap.description,
            "linked_capability_or_offer": cap.description,
            "drop": float(pert["drop"]),
            "causes_rank_flip": bool(
                rank_eff.get("valid")
                and rank_eff.get("winner_dropped_from_top1")
            ),
            "rank_effect": rank_eff,
        })

    # 2) every offer on task: drive strength to 0
    for off in task.offers:
        if not off.description:
            continue
        pert = perturb_task_reduce_offer_strength(
            u, v, task, world_model, off.description, new_strength=0.0,
        )
        if not pert.get("valid"):
            continue
        rank_eff = {"valid": False}
        if candidate_pool is not None:
            spec = {
                "type": "reduce_offer_strength",
                "offer": off.description,
                "new_strength": 0.0,
            }
            rp = rank_after_perturbation(
                u, task, candidate_pool, world_model, spec,
            )
            rank_eff = {
                "valid": rp.get("valid", False),
                "winner_rank_after": rp.get("winner_rank_after"),
                "winner_dropped_from_top1": rp.get("winner_dropped_from_top1"),
                "new_winner_id": rp.get("new_winner_id"),
            }
        candidates.append({
            "factor_type": "offer",
            "name": off.description,
            "linked_capability_or_offer": off.description,
            "drop": float(pert["drop"]),
            "causes_rank_flip": bool(
                rank_eff.get("valid")
                and rank_eff.get("winner_dropped_from_top1")
            ),
            "rank_effect": rank_eff,
        })

    if not candidates:
        return {"valid": False, "reason": "no_atomic_edits"}

    # ── rank by empirical drop ───────────────────────────────────────
    candidates.sort(key=lambda d: d["drop"], reverse=True)
    oracle_top = candidates[0]

    # Where does the explanation's pick sit in the oracle ranking?
    # Match by linked target — that's what `_perturb_factor` actually
    # perturbs, so we want to compare *the same edit* across the two
    # rankings, not just the same skill name.
    expl_linked = expl_top_factor.get("linked_capability_or_offer")
    expl_type = expl_top_factor.get("factor_type")
    expl_target_kind = "capability" if expl_type == "requirement" else "offer"
    expl_rank: Optional[int] = None
    for i, c in enumerate(candidates, start=1):
        if (c["factor_type"] == expl_target_kind
                and c["linked_capability_or_offer"] == expl_linked):
            expl_rank = i
            break

    matches_oracle = bool(
        oracle_top["factor_type"] == expl_target_kind
        and oracle_top["linked_capability_or_offer"] == expl_linked
    )

    oracle_top_drop = float(oracle_top["drop"])
    ratio = (
        float(expl_top_drop / oracle_top_drop)
        if oracle_top_drop > 1e-12 else
        (1.0 if expl_top_drop <= 1e-12 else 0.0)
    )

    return {
        "valid": True,
        "candidate_id": v.user_id,
        "M_before": float(M_before),
        "explanation_top_factor": {
            "factor_type": expl_top_factor.get("factor_type"),
            "name": expl_top_factor.get("name"),
            "linked_capability_or_offer": expl_linked,
            "contribution_to_M": float(
                expl_top_factor.get("contribution_to_M", 0.0)
            ),
        },
        "explanation_top_drop": expl_top_drop,
        "oracle_top_factor": {
            "factor_type": oracle_top["factor_type"],
            "name": oracle_top["name"],
            "linked_capability_or_offer": oracle_top["linked_capability_or_offer"],
            "drop": oracle_top_drop,
            "causes_rank_flip": bool(oracle_top["causes_rank_flip"]),
        },
        "oracle_top_drop": oracle_top_drop,
        "explanation_vs_oracle_ratio": ratio,
        "explanation_matches_oracle": matches_oracle,
        "explanation_rank_among_all_perturbations": expl_rank,
        "num_perturbations": len(candidates),
        "all_perturbations": candidates,
    }


# ---------------------------------------------------------------------------
# 2.7 — pairwise contrastive
# ---------------------------------------------------------------------------

def pairwise_contrastive_explanation(
    u: UserState,
    winner: UserState,
    loser: UserState,
    task: Task,
    world_model: WorldModel,
) -> Dict[str, Any]:
    """Why is `winner` ranked above `loser`?

    Decomposes both candidates against the same task and lines up
    contributions by (factor_type, name). The signed delta
    `winner.contribution - loser.contribution` measures where the gap
    actually came from. We keep the top 5 positive deltas (factors that
    helped winner) and also surface any factors where the loser was
    actually ahead (`loser_advantages`) — those are the places a
    counterfactual edit would need to overcome.
    """
    w_decomp = explain_match_decomposition(u, winner, task, world_model)
    l_decomp = explain_match_decomposition(u, loser, task, world_model)

    def _flatten(decomp: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for c in decomp["requirement_contributions"]:
            out[("requirement", c["name"])] = c
        for c in decomp["need_contributions"]:
            out[("need", c["name"])] = c
        return out

    w_map = _flatten(w_decomp)
    l_map = _flatten(l_decomp)

    deltas: List[Dict[str, Any]] = []
    for key in set(w_map.keys()) | set(l_map.keys()):
        w_c = w_map.get(key, {})
        l_c = l_map.get(key, {})
        w_contrib = float(w_c.get("contribution_to_M", 0.0))
        l_contrib = float(l_c.get("contribution_to_M", 0.0))
        delta = w_contrib - l_contrib
        if abs(delta) < 1e-12:
            continue
        if key[0] == "requirement":
            w_linked = w_c.get("best_capability") if w_c else None
            l_linked = l_c.get("best_capability") if l_c else None
        else:
            w_linked = w_c.get("best_offer") if w_c else None
            l_linked = l_c.get("best_offer") if l_c else None
        deltas.append({
            "factor_type": key[0],
            "name": key[1],
            "winner_contribution": w_contrib,
            "loser_contribution": l_contrib,
            "delta": float(delta),
            "winner_linked": w_linked,
            "loser_linked": l_linked,
        })

    positives = sorted(
        [d for d in deltas if d["delta"] > 0],
        key=lambda d: d["delta"],
        reverse=True,
    )
    negatives = sorted(
        [d for d in deltas if d["delta"] < 0],
        key=lambda d: d["delta"],
    )

    return {
        "winner_id": winner.user_id,
        "loser_id": loser.user_id,
        "winner_M": float(w_decomp["M"]),
        "loser_M": float(l_decomp["M"]),
        "score_gap": float(w_decomp["M"] - l_decomp["M"]),
        "top_contrastive_reasons": positives[:5],
        "loser_advantages": [
            {**d, "delta": float(-d["delta"])} for d in negatives[:5]
        ],
    }


# ---------------------------------------------------------------------------
# 2.8 — counterfactual rank flip
# ---------------------------------------------------------------------------

def counterfactual_rank_flip(
    u: UserState,
    winner: UserState,
    loser: UserState,
    task: Task,
    world_model: WorldModel,
    step: float = 0.05,
) -> Dict[str, Any]:
    """Search for the smallest single-factor edit that flips loser >
    winner. Two edit types are tried:

      1. Raise one of loser's capability μ values up to 1.0.
      2. Raise one of the task's offer strengths up to 1.0.

    Edit 1 only affects loser, so we compare against the original winner
    score. Edit 2 changes the task itself so it can move winner too —
    we re-score both and take the cheapest edit that still inverts the
    ranking. Returns `valid=False` when no single-factor edit suffices.
    """
    original_winner_M = _score_M(u, winner, task, world_model)
    original_loser_M = _score_M(u, loser, task, world_model)
    original_gap = float(original_winner_M - original_loser_M)

    best: Optional[Dict[str, Any]] = None

    def _maybe_update(candidate: Dict[str, Any]) -> None:
        nonlocal best
        if best is None or candidate["edit_size"] < best["edit_size"]:
            best = candidate

    # ── edit type 1: raise a loser capability μ ──────────────────────
    for cap_idx, cap in enumerate(loser.capabilities):
        old_mu = float(cap.mu)
        if old_mu >= 1.0 - 1e-9:
            continue
        target_mu = old_mu + step
        while target_mu <= 1.0 + 1e-9:
            mu_val = min(target_mu, 1.0)
            loser_copy = deepcopy(loser)
            loser_copy.capabilities[cap_idx].mu = float(mu_val)
            new_loser_M = _score_M(u, loser_copy, task, world_model)
            if new_loser_M > original_winner_M:
                _maybe_update({
                    "edit_type": "increase_capability_mu",
                    "target": cap.description or f"cap_{cap_idx}",
                    "from": old_mu,
                    "to": float(mu_val),
                    "edit_size": float(mu_val - old_mu),
                    "new_loser_M": float(new_loser_M),
                    "new_winner_M": float(original_winner_M),
                })
                break  # smallest mu that works for this cap
            if mu_val >= 1.0 - 1e-9:
                break
            target_mu += step

    # ── edit type 2: raise an offer strength ─────────────────────────
    for off_idx, off in enumerate(task.offers):
        old_strength = float(off.strength)
        if old_strength >= 1.0 - 1e-9:
            continue
        target_s = old_strength + step
        while target_s <= 1.0 + 1e-9:
            s_val = min(target_s, 1.0)
            task_copy = deepcopy(task)
            task_copy.offers[off_idx].strength = float(s_val)
            new_winner_M = _score_M(u, winner, task_copy, world_model)
            new_loser_M = _score_M(u, loser, task_copy, world_model)
            if new_loser_M > new_winner_M:
                _maybe_update({
                    "edit_type": "increase_offer_strength",
                    "target": off.description or f"offer_{off_idx}",
                    "from": old_strength,
                    "to": float(s_val),
                    "edit_size": float(s_val - old_strength),
                    "new_loser_M": float(new_loser_M),
                    "new_winner_M": float(new_winner_M),
                })
                break
            if s_val >= 1.0 - 1e-9:
                break
            target_s += step

    if best is None:
        return {
            "valid": False,
            "winner_id": winner.user_id,
            "loser_id": loser.user_id,
            "original_winner_M": float(original_winner_M),
            "original_loser_M": float(original_loser_M),
            "original_gap": original_gap,
            "edit_type": None,
            "target": None,
            "from": None,
            "to": None,
            "edit_size": None,
            "new_loser_M": None,
            "new_winner_M": None,
            "rank_flipped": False,
        }

    return {
        "valid": True,
        "winner_id": winner.user_id,
        "loser_id": loser.user_id,
        "original_winner_M": float(original_winner_M),
        "original_loser_M": float(original_loser_M),
        "original_gap": original_gap,
        "edit_type": best["edit_type"],
        "target": best["target"],
        "from": float(best["from"]),
        "to": float(best["to"]),
        "edit_size": float(best["edit_size"]),
        "new_loser_M": float(best["new_loser_M"]),
        "new_winner_M": float(best["new_winner_M"]),
        "rank_flipped": True,
    }
