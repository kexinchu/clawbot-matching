"""Matching pipelines — Layer 3.1 coarse filtering (§4A.4)

  match_one_to_one : 1-1 ranking (sort by M̃, take top-K)
  match_one_to_n   : 1-N team building (greedy submodular maximisation)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import MatchConfig
from scoring import compute_gate, compute_match_score
from datatypes import MatchResult, Task, TeamResult, UserState
from utils import attention_weighted_value, softmax


# ── 1-1 matching pipeline ────────────────────────────────────────────

def match_one_to_one(
    u: UserState,
    task: Task,
    pool: List[UserState],
    theta: np.ndarray,
    cfg: MatchConfig,
    top_k: int = 10,
    use_ucb: bool = True,
    round_t: int = 1,
    security_check_fn: Optional[Callable] = None,
) -> List[MatchResult]:
    """1-1 matching: rank candidates by M̃, return top-K.

    Pipeline:
      Phase 1: gate filtering
      Phase 2: MapScore (with optional UCB)
      Phase 3: sort descending, return top-K
    """
    results: List[MatchResult] = []
    for v in pool:
        r = compute_match_score(
            u, v, task, theta, cfg,
            use_ucb=use_ucb, round_t=round_t,
            security_check_fn=security_check_fn,
        )
        if r.sigma_gate == 1:
            results.append(r)

    results.sort(key=lambda r: r.match_score, reverse=True)
    return results[:top_k]


# ── 1-N team building (greedy submodular maximisation) ────────────────

def match_one_to_n(
    u: UserState,
    task: Task,
    pool: List[UserState],
    theta: np.ndarray,
    cfg: MatchConfig,
    use_ucb: bool = True,
    round_t: int = 1,
    security_check_fn: Optional[Callable] = None,
) -> TeamResult:
    """1-N team building — greedy submodular maximisation (§4.3).

    Objective R(S) uses **max model**:
      R(S) = Σ_j min(max_{v∈S} p̃_v^j, Gap_j) × w_j / Σ_j Gap_j × w_j

    Tracks max_coverage[j] = max_{v∈S} p̃_v^j per requirement.
    """
    soft_reqs = [r for r in task.requirements if r.constraint_type == "soft"]
    m = len(soft_reqs)
    if m == 0:
        return TeamResult(
            team_members=[], collective_coverage=1.0,
            residual_gaps={}, termination_reason="no_soft_reqs",
        )

    beta_t = (
        math.sqrt(cfg.ucb_beta_scale * math.log(max(round_t, 2)))
        if use_ucb else 0.0
    )

    # ── Phase 1: pre-processing ──
    eligible: List[Tuple[UserState, np.ndarray]] = []
    for v in pool:
        sg, _ = compute_gate(u, v, task, cfg, security_check_fn)
        if sg == 0 or not v.capabilities:
            continue
        v_embs = np.stack([c.embedding for c in v.capabilities])
        v_mus = np.array([c.mu for c in v.capabilities])
        if use_ucb:
            v_sigs = np.array([c.sigma for c in v.capabilities])
            v_mus = np.minimum(1.0, v_mus + beta_t * v_sigs)
        ptildes = np.array([
            attention_weighted_value(
                req.embedding, v_embs, v_mus, cfg.temperature,
            )
            for req in soft_reqs
        ])
        eligible.append((v, ptildes))

    if not eligible:
        return TeamResult(
            team_members=[], collective_coverage=0.0,
            residual_gaps={r.description: 0.0 for r in soft_reqs},
            termination_reason="no_eligible_candidate",
        )

    # ── Phase 2: initialise gaps ──
    u_embs = (
        np.stack([c.embedding for c in u.capabilities])
        if u.capabilities
        else np.empty((0, cfg.embedding_dim))
    )
    u_mus = (
        np.array([c.mu for c in u.capabilities])
        if u.capabilities
        else np.array([])
    )

    q_vals = np.array([r.level for r in soft_reqs])
    w_j = q_vals / (q_vals.sum() + cfg.epsilon)

    gaps = np.array([
        max(0.0, req.level - attention_weighted_value(
            req.embedding, u_embs, u_mus, cfg.temperature,
        ))
        for req in soft_reqs
    ])
    denom = float(np.dot(gaps, w_j) + cfg.epsilon)

    # ── Phase 3: greedy iteration ──
    team: List[UserState] = []
    team_ids: List[str] = []
    remaining = list(range(len(eligible)))
    max_cov = np.zeros(m)
    sel_order: List[Tuple[str, float]] = []
    term_reason = "max_size"

    while len(team_ids) < cfg.n_max:
        covered = np.minimum(max_cov, gaps)
        if np.dot(gaps - covered, w_j) < cfg.gap_epsilon:
            term_reason = "gap_covered"
            break

        best_i, best_gain = -1, 0.0
        for i in remaining:
            _, ptildes = eligible[i]
            new_max = np.maximum(max_cov, ptildes)
            new_cov = np.minimum(new_max, gaps)
            old_cov = np.minimum(max_cov, gaps)
            gain = float(np.dot(new_cov - old_cov, w_j)) / denom
            if gain > best_gain:
                best_gain, best_i = gain, i

        if best_i < 0:
            term_reason = "no_positive_gain"
            break

        sel_v, sel_pt = eligible[best_i]
        team_ids.append(sel_v.user_id)
        team.append(sel_v)
        remaining.remove(best_i)
        sel_order.append((sel_v.user_id, best_gain))
        max_cov = np.maximum(max_cov, sel_pt)

    # ── Phase 4: output ──
    final_cov = np.minimum(max_cov, gaps)
    coll = float(np.dot(final_cov, w_j)) / denom
    resid: Dict[str, float] = {
        soft_reqs[j].description: float(gaps[j] - final_cov[j])
        for j in range(m)
    }

    per_member = [
        compute_match_score(
            u, vp, task, theta, cfg, use_ucb, round_t, security_check_fn,
        )
        for vp in team
    ]

    return TeamResult(
        team_members=team_ids,
        collective_coverage=round(coll, 4),
        residual_gaps=resid,
        selection_order=sel_order,
        per_member=per_member,
        termination_reason=term_reason,
    )
