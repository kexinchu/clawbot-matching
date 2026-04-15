"""Scoring functions — Layer 2 world model (§4A.3)

Core components:
  σ(u,v,T)         — binary gate (§3.4)
  S_cap(v,u,T)     — capability coverage (§3.1)
  S_need(v,T)      — need satisfaction (§3.2)
  M(u,v,T)         — MapScore = σ · (w_c·S_cap + w_n·S_need)
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import MatchConfig
from datatypes import (
    GapDetail,
    MatchResult,
    NeedDetail,
    Task,
    UserState,
)
from utils import attention_weighted_value, cosine_sim, softmax


# ── Gate σ(u, v, T) ──────────────────────────────────────────────────

def compute_gate(
    u: UserState,
    v: UserState,
    task: Task,
    cfg: MatchConfig,
    security_check_fn: Optional[Callable] = None,
) -> Tuple[int, str]:
    """Binary gate (§3.4).  Returns (0|1, failure_reason).

    Checks:
      1. External security check (optional callback)
      2. Data clearance: v.clearance >= task.data_clearance
      3. Hard constraints: for each hard req j,
         max-match proficiency p̂_v^j >= q_j
    """
    # External security check
    if security_check_fn is not None:
        passed, reason = security_check_fn(u, v, task)
        if not passed:
            return 0, reason

    # Data clearance
    if v.clearance_level < task.data_clearance:
        return 0, (
            f"clearance: v={v.clearance_level} < task={task.data_clearance}"
        )

    # Hard constraint check — MAX matching (not attention!)
    hard_reqs = [r for r in task.requirements if r.constraint_type == "hard"]
    for req in hard_reqs:
        best_mu = 0.0
        found = False
        for cap in v.capabilities:
            sim = cosine_sim(req.embedding, cap.embedding)
            if sim > cfg.tau_hard:
                found = True
                best_mu = max(best_mu, cap.mu)
        if not found:
            return 0, (
                f"hard: no capability matches '{req.description}' "
                f"(no sim > {cfg.tau_hard})"
            )
        if best_mu < req.level:
            return 0, (
                f"hard: '{req.description}' need {req.level:.2f}, "
                f"best {best_mu:.2f}"
            )

    return 1, ""


# ── S_cap(v, u, T) ───────────────────────────────────────────────────

def compute_s_cap(
    v: UserState,
    u: UserState,
    task: Task,
    cfg: MatchConfig,
    v_mu_override: Optional[np.ndarray] = None,
) -> Tuple[float, List[GapDetail]]:
    """Capability coverage score (§3.1) — 'how much can v help u?'

    v_mu_override: when using UCB, pass min(1, μ + β·σ) to replace raw μ.
    Returns (score ∈ [0,1], per-requirement details).
    """
    soft_reqs = [r for r in task.requirements if r.constraint_type == "soft"]
    if not soft_reqs:
        return 1.0, []

    # u's capability matrix
    if u.capabilities:
        u_embs = np.stack([c.embedding for c in u.capabilities])
        u_mus = np.array([c.mu for c in u.capabilities])
    else:
        u_embs = np.empty((0, cfg.embedding_dim))
        u_mus = np.array([])

    # v's capability matrix (with optional UCB override)
    if v.capabilities:
        v_embs = np.stack([c.embedding for c in v.capabilities])
        v_mus = (
            v_mu_override
            if v_mu_override is not None
            else np.array([c.mu for c in v.capabilities])
        )
    else:
        v_embs = np.empty((0, cfg.embedding_dim))
        v_mus = np.array([])

    # Requirement weights: w_j = q_j / Σq
    q_vals = np.array([r.level for r in soft_reqs])
    w_j = q_vals / (q_vals.sum() + cfg.epsilon)

    numerator = 0.0
    denominator = 0.0
    details: List[GapDetail] = []

    for idx, req in enumerate(soft_reqs):
        p_tilde_u = attention_weighted_value(
            req.embedding, u_embs, u_mus, cfg.temperature
        )
        gap = max(0.0, req.level - p_tilde_u)
        p_tilde_v = attention_weighted_value(
            req.embedding, v_embs, v_mus, cfg.temperature
        )
        coverage = min(p_tilde_v, gap)

        numerator += coverage * w_j[idx]
        denominator += gap * w_j[idx]

        pct = (
            f"{coverage / (gap + cfg.epsilon) * 100:.0f}%"
            if gap > cfg.epsilon
            else "N/A (u already met)"
        )
        details.append(GapDetail(
            req_description=req.description,
            q_j=req.level,
            p_tilde_u=round(p_tilde_u, 4),
            gap=round(gap, 4),
            p_tilde_v=round(p_tilde_v, 4),
            coverage=round(coverage, 4),
            coverage_pct=pct,
        ))

    if denominator < cfg.epsilon:
        return 1.0, details

    return numerator / (denominator + cfg.epsilon), details


# ── S_need(v, T) ─────────────────────────────────────────────────────

def compute_s_need(
    v: UserState,
    task: Task,
    cfg: MatchConfig,
) -> Tuple[float, List[NeedDetail]]:
    """Need satisfaction score (§3.2) — 'how attractive is T to v?'

    Returns (score ∈ [0,1], per-need details).
    """
    if not v.needs:
        return 0.5, []

    # Merge offers (inferred ones get discounted)
    offer_list = []
    for offer in task.offers:
        s = offer.strength * (
            cfg.alpha_infer if offer.source == "inferred" else 1.0
        )
        offer_list.append((offer.embedding, s))
    if not offer_list:
        return 0.0, []

    offer_embs = np.stack([o[0] for o in offer_list])
    offer_strs = np.array([o[1] for o in offer_list])

    numerator = 0.0
    denominator = 0.0
    details: List[NeedDetail] = []

    for need in v.needs:
        o_tilde = attention_weighted_value(
            need.embedding, offer_embs, offer_strs, cfg.temperature
        )
        satisfied = min(o_tilde, need.intensity)
        numerator += satisfied * need.intensity
        denominator += need.intensity ** 2

        pct = f"{satisfied / (need.intensity + cfg.epsilon) * 100:.0f}%"
        details.append(NeedDetail(
            need_description=need.description,
            need_intensity=round(need.intensity, 4),
            offer_matched=round(o_tilde, 4),
            satisfied=round(satisfied, 4),
            satisfaction_pct=pct,
        ))

    return numerator / (denominator + cfg.epsilon), details


# ── MapScore M(u, v, T) ──────────────────────────────────────────────

def compute_match_score(
    u: UserState,
    v: UserState,
    task: Task,
    theta: np.ndarray,
    cfg: MatchConfig,
    use_ucb: bool = False,
    round_t: int = 1,
    security_check_fn: Optional[Callable] = None,
) -> MatchResult:
    """Layer 2 core — MapScore.

    M = σ · (w_c · S_cap + w_n · S_need)
    θ ∈ R², w = softmax(θ), w_c + w_n = 1.

    When use_ucb=True, S_cap uses optimistic μ̃ = min(1, μ + β·σ).
    S_need is NOT UCB-enhanced.
    """
    w = softmax(theta)

    # Gate
    sigma, fail_reason = compute_gate(u, v, task, cfg, security_check_fn)
    if sigma == 0:
        return MatchResult(
            candidate_id=v.user_id, match_score=0.0,
            s_cap=0.0, s_need=0.0, w_c=float(w[0]), w_n=float(w[1]),
            sigma_gate=0, gate_fail_reason=fail_reason,
        )

    # UCB enhancement (S_cap only)
    v_mu_ucb = None
    ucb_bonus = 0.0
    if use_ucb and v.capabilities:
        beta_t = math.sqrt(cfg.ucb_beta_scale * math.log(max(round_t, 2)))
        orig = np.array([c.mu for c in v.capabilities])
        sigs = np.array([c.sigma for c in v.capabilities])
        v_mu_ucb = np.minimum(1.0, orig + beta_t * sigs)
        ucb_bonus = float(np.sum(v_mu_ucb - orig))

    s_cap, gap_details = compute_s_cap(v, u, task, cfg, v_mu_ucb)
    s_need, need_details = compute_s_need(v, task, cfg)

    score = float(sigma * (w[0] * s_cap + w[1] * s_need))

    return MatchResult(
        candidate_id=v.user_id, match_score=score,
        s_cap=s_cap, s_need=s_need,
        w_c=float(w[0]), w_n=float(w[1]),
        sigma_gate=sigma,
        gap_details=gap_details, need_details=need_details,
        ucb_bonus=ucb_bonus,
    )
