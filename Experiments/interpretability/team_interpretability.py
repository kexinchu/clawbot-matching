"""1-N team interpretability — Shapley attribution of collective coverage.

For each task we:
  1. Build requester + task + candidate pool (same as run_interpretability).
  2. Define v(S) = collective coverage R(S) from the MAX submodular objective.
  3. Compute exact Shapley values φ_i for each pool member (n ≤ 8 feasible).
  4. Check that greedy-selected team members have higher φ than non-selected
     specialists with overlapping (redundant) skills.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_MAPPING = _REPO_ROOT / "mapping-algo"
_ONLINE = _REPO_ROOT / "Online_learning"
for _p in (str(_HERE), str(_MAPPING), str(_ONLINE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datatypes import UserState, Task  # noqa: E402
from pipeline import match_one_to_n  # noqa: E402
from utils import attention_weighted_value  # noqa: E402

from experiment_config import ExperimentContext, build_encoder  # noqa: E402


def _team_coverage_value(
    u: UserState,
    task: Task,
    subset: Tuple[UserState, ...],
    ctx: ExperimentContext,
    use_ucb: bool = False,
    round_t: int = 1,
) -> float:
    """Characteristic function v(S): collective coverage of subset S."""
    soft_reqs = [r for r in task.requirements if r.constraint_type == "soft"]
    if not soft_reqs or not subset:
        return 0.0

    cfg = ctx.cfg
    beta_t = (
        math.sqrt(cfg.ucb_beta_scale * math.log(max(round_t, 2)))
        if use_ucb else 0.0
    )

    u_embs = (
        np.stack([c.embedding for c in u.capabilities])
        if u.capabilities else np.empty((0, cfg.embedding_dim))
    )
    u_mus = np.array([c.mu for c in u.capabilities]) if u.capabilities else np.array([])

    q_vals = np.array([r.level for r in soft_reqs])
    w_j = q_vals / (q_vals.sum() + cfg.epsilon)
    gaps = np.array([
        max(0.0, req.level - attention_weighted_value(
            req.embedding, u_embs, u_mus, cfg.temperature,
        ))
        for req in soft_reqs
    ])
    denom = float(np.dot(gaps, w_j) + cfg.epsilon)
    if denom <= 0:
        return 1.0

    max_cov = np.zeros(len(soft_reqs))
    for v in subset:
        if not v.capabilities:
            continue
        v_embs = np.stack([c.embedding for c in v.capabilities])
        v_mus = np.array([c.mu for c in v.capabilities])
        if use_ucb:
            v_sigs = np.array([c.sigma for c in v.capabilities])
            v_mus = np.minimum(1.0, v_mus + beta_t * v_sigs)
        ptildes = np.array([
            attention_weighted_value(req.embedding, v_embs, v_mus, cfg.temperature)
            for req in soft_reqs
        ])
        max_cov = np.maximum(max_cov, ptildes)

    final_cov = np.minimum(max_cov, gaps)
    return float(np.dot(final_cov, w_j)) / denom


def exact_shapley(
    u: UserState,
    task: Task,
    pool: List[UserState],
    ctx: ExperimentContext,
) -> Dict[str, float]:
    """Exact Shapley values for v(S) over ``pool`` (feasible for n ≤ 10)."""
    n = len(pool)
    if n == 0:
        return {}
    # Precompute v(S) for all subsets via bitmask
    v_cache: Dict[int, float] = {}

    def v_of_mask(mask: int) -> float:
        if mask not in v_cache:
            members = tuple(pool[i] for i in range(n) if mask & (1 << i))
            v_cache[mask] = _team_coverage_value(u, task, members, ctx)
        return v_cache[mask]

    shapley: Dict[str, float] = {c.user_id: 0.0 for c in pool}
    factorial_n = math.factorial(n)
    for i, player in enumerate(pool):
        phi = 0.0
        others_mask_full = (1 << n) - 1 - (1 << i)
        for r in range(n):
            for coalition in itertools.combinations(
                [j for j in range(n) if j != i], r,
            ):
                s_mask = sum(1 << j for j in coalition)
                s_size = r
                weight = (
                    math.factorial(s_size)
                    * math.factorial(n - s_size - 1)
                    / factorial_n
                )
                v_with = v_of_mask(s_mask | (1 << i))
                v_without = v_of_mask(s_mask)
                phi += weight * (v_with - v_without)
        shapley[player.user_id] = float(phi)
    return shapley


def evaluate_team_task(
    task_entry: dict,
    pool_size: int,
    ctx: ExperimentContext,
    theta: Optional[np.ndarray] = None,
) -> Dict:
    """Run 1-N greedy team build + Shapley analysis for one task."""
    from run_interpretability import (  # noqa: WPS433 — shared builders
        build_task,
        build_requester,
        build_learning_candidate,
        EVAL_THETA_C,
        EVAL_THETA_N,
    )

    task_dict = task_entry["task"]
    proposer = task_entry["proposer_profile"]
    candidate_entries = task_entry["candidates"][:pool_size]
    pool = [build_learning_candidate(c) for c in candidate_entries]

    task = build_task(task_dict)
    requester = build_requester(proposer)
    theta_arr = theta if theta is not None else np.array(
        [EVAL_THETA_C, EVAL_THETA_N], dtype=float,
    )

    team = match_one_to_n(
        requester, task, pool, theta_arr, ctx.cfg,
        use_ucb=False, round_t=1,
    )
    shapley = exact_shapley(requester, task, pool, ctx)

    selected = set(team.team_members)
    unselected = [c.user_id for c in pool if c.user_id not in selected]

    sel_phi = [shapley[cid] for cid in team.team_members if cid in shapley]
    unsel_phi = [shapley[cid] for cid in unselected if cid in shapley]

    # Complementarity check: selected members should beat pool median φ
    all_phi = list(shapley.values())
    median_phi = float(np.median(all_phi)) if all_phi else 0.0
    mean_sel = float(np.mean(sel_phi)) if sel_phi else 0.0
    mean_unsel = float(np.mean(unsel_phi)) if unsel_phi else 0.0

    return {
        "task_id": task_dict["task_id"],
        "title": task_dict.get("title", ""),
        "team_members": team.team_members,
        "collective_coverage": team.collective_coverage,
        "termination_reason": team.termination_reason,
        "selection_order": team.selection_order,
        "shapley": shapley,
        "mean_shapley_selected": mean_sel,
        "mean_shapley_unselected": mean_unsel,
        "selected_beats_median": bool(mean_sel >= median_phi),
        "selected_beats_unselected_mean": bool(mean_sel >= mean_unsel),
    }


def run_team_interpretability(
    testset_path: Path,
    pool_size: int = 5,
    num_tasks: int = 0,
    encoder: str = "simple",
    out_path: Optional[Path] = None,
) -> Dict:
    ctx = build_encoder(encoder)
    # Patch run_interpretability globals so shared builders use this encoder
    import run_interpretability as ri  # noqa: WPS433
    ri.ENC = ctx.enc
    ri.CFG = ctx.cfg
    ri.EMBED_DIM = ctx.embed_dim

    data = json.loads(testset_path.read_text())
    tasks = data["tasks"]
    if num_tasks > 0:
        tasks = tasks[:num_tasks]

    results = [evaluate_team_task(t, pool_size, ctx) for t in tasks]
    n_sel_beats = sum(1 for r in results if r["selected_beats_median"])
    n_sel_gt_unsel = sum(1 for r in results if r["selected_beats_unselected_mean"])

    summary = {
        "n_tasks": len(results),
        "frac_selected_beats_median_shapley": n_sel_beats / max(len(results), 1),
        "frac_selected_beats_unselected_mean": n_sel_gt_unsel / max(len(results), 1),
        "mean_collective_coverage": float(np.mean(
            [r["collective_coverage"] for r in results],
        )),
    }

    payload = {
        "metadata": {
            "testset": str(testset_path),
            "pool_size": pool_size,
            "encoder": ctx.encoder_name,
        },
        "per_task": results,
        "summary": summary,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str))

    return payload


def main() -> int:
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument("--pool-size", type=int, default=5)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--encoder", default="simple")
    parser.add_argument("--testset", default="")
    args = parser.parse_args()
    testset = Path(args.testset) if args.testset else (
        _REPO_ROOT / "simulator" / "20_Tasks_Testset_tiered.json"
    )
    out = _REPO_ROOT / "logs" / "team_interpretability.json"
    payload = run_team_interpretability(
        testset, pool_size=args.pool_size,
        num_tasks=args.num_tasks, encoder=args.encoder, out_path=out,
    )
    s = payload["summary"]
    print(f"Team Shapley: {s['frac_selected_beats_median_shapley']:.2f} "
          f"selected ≥ median, coverage={s['mean_collective_coverage']:.3f}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
