"""Run the 1-1 faithfulness explanation experiment over the tiered testset.

For every task in `simulator/20_Tasks_Testset_tiered.json` we:

  1. Build a requester, task, and candidate pool (re-using the exact
     `build_task` / `build_requester` / `build_learning_candidate`
     helpers that `run_interpretability` uses, so the world the
     candidates live in is identical to the weight-alignment experiment).
  2. Score every candidate with no-UCB MapScore using a fixed θ
     (EVAL_THETA_C, EVAL_THETA_N) — these are the same θ values used in
     the production scoring evaluation, so the explanations describe
     the deployed model, not some artefact of online learning.
  3. Take the Top-1 candidate as the *winner* and Top-2 as the *loser*.
  4. For the winner:
       * a per-requirement / per-need decomposition of M (so we can see
         which factors actually drove the score), and
       * a deletion faithfulness test: knock down the top, a random,
         and the bottom contributing factor and check that the explanation's
         top reason caused the largest drop.
  5. For winner vs loser:
       * a pairwise contrastive explanation (where the gap came from),
       * a minimal-edit counterfactual that flips the rank.
  6. Aggregate everything into a JSON suitable for downstream analysis.

The script intentionally never calls an LLM and adds no dependencies
beyond what `run_interpretability.py` already imports. Production code
in `mapping-algo` and `Online_learning` is untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_ONLINE = _REPO_ROOT / "Online_learning"
_MAPPING_ALGO = _REPO_ROOT / "mapping-algo"
for _p in (str(_HERE), str(_ONLINE), str(_MAPPING_ALGO), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# load .env best-effort so SBERT etc. share configuration with the
# weight interpretability experiment, even though we don't call LLMs.
from env_loader import load_dotenv  # noqa: E402
load_dotenv()

from datatypes import Task, UserState  # noqa: E402
from WorldModel import WorldModel  # noqa: E402

from experiment_config import build_encoder  # noqa: E402
from faithfulness import (  # noqa: E402
    counterfactual_rank_flip,
    deletion_faithfulness_test,
    explain_match_decomposition,
    pairwise_contrastive_explanation,
    perturbation_oracle_baseline,
    shuffled_deletion_baseline,
)
import run_interpretability as ri  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TESTSET = _REPO_ROOT / "simulator" / "20_Tasks_Testset_tiered.json"
LOGS_DIR = _REPO_ROOT / "logs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> Optional[float]:
    """Mean that returns None for empty input or all-NaN input (so the
    summary JSON is round-trippable through `json.load`).
    """
    if not values:
        return None
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
        dtype=float,
    )
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _safe_frac(bools: List[bool]) -> Optional[float]:
    if not bools:
        return None
    return float(sum(1 for b in bools if b) / len(bools))


def _saturation_diagnostics(
    ranking: List[Dict[str, Any]],
    winner_id: str,
    loser_id: str,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Summarise how many candidates have `S_cap ≈ 1.0` and how much
    headroom separates the winner and loser. Drives the
    "is the testset saturated?" half of the comparison table.
    """
    if not ranking:
        return {
            "num_candidates_scap_eq_1": 0,
            "frac_candidates_scap_eq_1": 0.0,
            "winner_scap": None,
            "loser_scap": None,
            "winner_loser_scap_gap": None,
        }
    n_sat = sum(1 for r in ranking if r["S_cap"] >= 1.0 - eps)
    by_id = {r["candidate_id"]: r for r in ranking}
    w = by_id.get(winner_id)
    l = by_id.get(loser_id)
    winner_scap = float(w["S_cap"]) if w else None
    loser_scap = float(l["S_cap"]) if l else None
    gap = (
        float(winner_scap - loser_scap)
        if winner_scap is not None and loser_scap is not None
        else None
    )
    return {
        "num_candidates_scap_eq_1": int(n_sat),
        "frac_candidates_scap_eq_1": float(n_sat / len(ranking)),
        "winner_scap": winner_scap,
        "loser_scap": loser_scap,
        "winner_loser_scap_gap": gap,
    }


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy scalars / NaN / Infinity to JSON-safe types.

    `json.dumps` already understands list/dict, but we need to:
      * unwrap numpy ints / floats / bools to native types
      * collapse NaN / +Inf / -Inf into None so the resulting JSON parses
        back with `json.load` (Python defaults to dumping `NaN`, which
        is not valid JSON).
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------

def evaluate_task(
    task_entry: dict,
    pool_size: int,
    seed: int,
    world_model: WorldModel,
) -> Optional[Dict[str, Any]]:
    """Score every candidate, identify winner / loser, then build the
    explanation, deletion test, contrastive view, and counterfactual.

    Returns `None` (and lets the caller record it under `skipped`) when
    fewer than 2 candidates pass the gate — neither contrastive nor
    counterfactual is meaningful with only one candidate.
    """
    task_dict = task_entry["task"]
    proposer = task_entry["proposer_profile"]
    candidate_entries = task_entry["candidates"]

    if pool_size and pool_size > 0:
        candidate_entries = candidate_entries[:pool_size]

    task = ri.build_task(task_dict)
    requester = ri.build_requester(proposer)
    candidate_pool: List[UserState] = [
        ri.build_learning_candidate(c) for c in candidate_entries
    ]

    # ── score everyone (no UCB, identical θ as the production eval) ──
    ranking_raw: List[Dict[str, Any]] = []
    for cand in candidate_pool:
        r = world_model.compute_match(
            requester, cand, task, use_ucb=False, round_t=1,
        )
        ranking_raw.append({
            "candidate_id": cand.user_id,
            "M": float(r.match_score),
            "S_cap": float(r.s_cap),
            "S_need": float(r.s_need),
            "sigma_gate": int(r.sigma_gate),
        })

    ranking_sorted = sorted(ranking_raw, key=lambda x: x["M"], reverse=True)

    if len(ranking_sorted) < 2:
        return {
            "_skip_reason": "not_enough_candidates",
            "task_id": task_dict["task_id"],
        }

    winner_entry = ranking_sorted[0]
    loser_entry = ranking_sorted[1]

    cand_by_id = {c.user_id: c for c in candidate_pool}
    winner = cand_by_id[winner_entry["candidate_id"]]
    loser = cand_by_id[loser_entry["candidate_id"]]

    saturation = _saturation_diagnostics(
        ranking_sorted, winner.user_id, loser.user_id,
    )

    # ── explanations ─────────────────────────────────────────────────
    winner_decomp = explain_match_decomposition(
        requester, winner, task, world_model,
    )
    # candidate_pool is passed so we get rank-level effects too.
    deletion = deletion_faithfulness_test(
        requester, winner, task, world_model,
        candidate_pool=candidate_pool, seed=seed,
    )
    shuffled = shuffled_deletion_baseline(
        requester, winner, task, world_model,
        candidate_pool=candidate_pool, seed=seed, n_trials=20,
    )
    oracle = perturbation_oracle_baseline(
        requester, winner, task, world_model,
        candidate_pool=candidate_pool,
    )
    contrastive = pairwise_contrastive_explanation(
        requester, winner, loser, task, world_model,
    )
    counterfactual = counterfactual_rank_flip(
        requester, winner, loser, task, world_model, step=0.05,
    )

    # Pull testset-provided archetype metadata when available (contrast V2
    # writes per-candidate archetypes so we can later tally winner
    # distribution by intended/actual archetype). Safe-default missing.
    cand_archetypes = (
        task_entry.get("metadata", {}).get("candidate_archetypes")
        or {
            c["candidate_profile"]["user_id"]:
                (c.get("tier_meta") or {}).get("archetype")
            for c in candidate_entries
        }
    )
    intended_archetype = (
        task_entry.get("metadata", {}).get("intended_winner_archetype")
    )

    return {
        "task_id": task_dict["task_id"],
        "title": task_dict.get("title", ""),
        "winner_id": winner.user_id,
        "loser_id": loser.user_id,
        "winner_archetype": cand_archetypes.get(winner.user_id),
        "loser_archetype": cand_archetypes.get(loser.user_id),
        "intended_winner_archetype": intended_archetype,
        "candidate_archetypes": cand_archetypes,
        "ranking": ranking_sorted,
        "saturation_diagnostics": saturation,
        "winner_decomposition": winner_decomp,
        "deletion_test": deletion,
        "shuffled_baseline": shuffled,
        "perturbation_oracle_baseline": oracle,
        "contrastive_explanation": contrastive,
        "counterfactual": counterfactual,
    }


# ---------------------------------------------------------------------------
# Systematic error analysis
# ---------------------------------------------------------------------------

def _collect_error_cases(per_task: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag per-task results that violate any of the faithfulness
    invariants we expect to hold. Each violation is reported separately
    (so a single task can appear multiple times for different reasons),
    and tagged with a short rule-based ``diagnosis`` string suggesting
    *why* it likely failed.
    """
    cases: List[Dict[str, Any]] = []
    for t in per_task:
        d = t.get("deletion_test", {}) or {}
        sb = t.get("shuffled_baseline", {}) or {}
        ob = t.get("perturbation_oracle_baseline", {}) or {}
        cf = t.get("counterfactual", {}) or {}
        sat = t.get("saturation_diagnostics", {}) or {}
        violations: List[Tuple[str, str]] = []

        if d.get("valid"):
            if d.get("top_factor_drop", 0.0) <= d.get("random_factor_drop", 0.0):
                violations.append((
                    "top_drop_le_random",
                    _diagnose(t, "top<=random"),
                ))
            if d.get("top_factor_drop", 0.0) <= d.get("bottom_factor_drop", 0.0):
                violations.append((
                    "top_drop_le_bottom",
                    _diagnose(t, "top<=bottom"),
                ))
            if not d.get("top_causes_rank_flip"):
                violations.append((
                    "top_no_rank_flip",
                    _diagnose(t, "no_flip"),
                ))

        if not cf.get("valid"):
            violations.append((
                "counterfactual_invalid",
                _diagnose(t, "cf_invalid"),
            ))

        if sb.get("valid"):
            if sb.get("true_top_drop_percentile", 1.0) < 0.8:
                violations.append((
                    "shuffled_percentile_low",
                    _diagnose(t, "shuffled_low"),
                ))

        if ob.get("valid"):
            if ob.get("explanation_vs_oracle_ratio", 1.0) < 0.8:
                violations.append((
                    "oracle_ratio_low",
                    _diagnose(t, "oracle_low"),
                ))

        if not violations:
            continue
        for error_type, diagnosis in violations:
            cases.append({
                "task_id": t["task_id"],
                "error_type": error_type,
                "diagnosis": diagnosis,
                "winner_id": t.get("winner_id"),
                "loser_id": t.get("loser_id"),
                "saturation_diagnostics": sat,
                "top_factor": d.get("top_factor"),
                "random_factor": d.get("random_factor"),
                "bottom_factor": d.get("bottom_factor"),
                "top_factor_drop": d.get("top_factor_drop"),
                "random_factor_drop": d.get("random_factor_drop"),
                "bottom_factor_drop": d.get("bottom_factor_drop"),
                "top_causes_rank_flip": d.get("top_causes_rank_flip"),
                "counterfactual": cf,
                "shuffled_baseline": sb,
                "perturbation_oracle_baseline": ob,
            })
    return cases


def _diagnose(task_result: Dict[str, Any], hint: str) -> str:
    """Rule-based one-line diagnosis. Reads winner_loser_scap_gap,
    explanation top factor type, and oracle disagreement to guess the
    failure mode.
    """
    sat = task_result.get("saturation_diagnostics", {}) or {}
    gap = task_result.get("contrastive_explanation", {}).get("score_gap", 0.0)
    scap_sat = sat.get("frac_candidates_scap_eq_1") or 0.0
    wl_scap_gap = sat.get("winner_loser_scap_gap") or 0.0
    ob = task_result.get("perturbation_oracle_baseline", {}) or {}
    expl = task_result.get("winner_decomposition", {})
    top_factor = (expl.get("top_positive_factors") or [{}])[0]
    factor_type = top_factor.get("factor_type")
    expl_target = top_factor.get("linked_capability_or_offer")
    oracle_target = (
        (ob.get("oracle_top_factor") or {}).get("linked_capability_or_offer")
    )

    if hint == "cf_invalid":
        if scap_sat >= 0.6 and abs(wl_scap_gap) < 0.05:
            return ("S_cap saturated for everyone; loser cannot overtake "
                    "with single-factor edit")
        if gap < 0.01:
            return "winner-loser score gap too small for stable rank flip"
        return "counterfactual requires multi-factor edit"
    if hint == "no_flip":
        if gap < 0.01:
            return ("winner-loser score gap too small — perturbations move "
                    "score but not rank")
        return ("top factor moves score but not enough to dethrone winner; "
                "consider rank-aware counterfactual")
    if hint == "top<=random":
        if factor_type == "need":
            return ("need-side attribution share ≠ perturbation sensitivity "
                    "(attention redistribution may cancel the edit)")
        return ("random factor happens to share a perturbation target "
                "with another high-importance factor")
    if hint == "top<=bottom":
        return ("bottom factor's linked element also dominates another "
                "high-contribution factor (entangled targets)")
    if hint == "shuffled_low":
        return "explanation ordering not statistically distinguishable from random"
    if hint == "oracle_low":
        if expl_target != oracle_target:
            return (f"explanation top factor ({expl_target}) is not the most "
                    f"impactful edit (oracle: {oracle_target})")
        return "explanation matches oracle target but with smaller drop"
    return "uncategorised"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
    winner_M = [t["winner_decomposition"]["M"] for t in per_task]
    loser_M = [
        t["contrastive_explanation"]["loser_M"] for t in per_task
    ]
    gaps = [
        t["contrastive_explanation"]["score_gap"] for t in per_task
    ]

    valid_deletions = [
        t["deletion_test"] for t in per_task
        if t["deletion_test"].get("valid")
    ]
    top_drops = [d["top_factor_drop"] for d in valid_deletions]
    rand_drops = [d["random_factor_drop"] for d in valid_deletions]
    bot_drops = [d["bottom_factor_drop"] for d in valid_deletions]
    top_beats_rand = [bool(d["top_beats_random"]) for d in valid_deletions]
    top_beats_bot = [bool(d["top_beats_bottom"]) for d in valid_deletions]

    # Rank-flip rates per kind of factor (only count tests where the
    # rank-level perturbation actually applied).
    def _rank_flip_rate(kind_field: str) -> Optional[float]:
        applicable: List[bool] = []
        for d in valid_deletions:
            eff = d.get(f"{kind_field}_rank_effect", {}) or {}
            if eff.get("valid"):
                applicable.append(bool(d.get(f"{kind_field}_causes_rank_flip")))
        if not applicable:
            return None
        return float(sum(1 for f in applicable if f) / len(applicable))

    top_flip_rate = _rank_flip_rate("top")
    random_flip_rate = _rank_flip_rate("random")
    bottom_flip_rate = _rank_flip_rate("bottom")

    valid_cfs = [
        t["counterfactual"] for t in per_task
        if t["counterfactual"].get("valid")
    ]
    cf_edit_sizes = [c["edit_size"] for c in valid_cfs]
    cf_attempts = len(per_task)

    # Saturation diagnostics (only count tasks where it was computed).
    sat_fracs = [
        t["saturation_diagnostics"]["frac_candidates_scap_eq_1"]
        for t in per_task
        if t.get("saturation_diagnostics")
        and t["saturation_diagnostics"].get("frac_candidates_scap_eq_1") is not None
    ]
    wl_scap_gaps = [
        t["saturation_diagnostics"]["winner_loser_scap_gap"]
        for t in per_task
        if t.get("saturation_diagnostics")
        and t["saturation_diagnostics"].get("winner_loser_scap_gap") is not None
    ]

    # Shuffled baseline diagnostics.
    valid_shuffles = [
        t["shuffled_baseline"] for t in per_task
        if t["shuffled_baseline"].get("valid")
    ]
    shuf_means = [s["mean_shuffled_top_drop"] for s in valid_shuffles]
    shuf_percentiles = [s["true_top_drop_percentile"] for s in valid_shuffles]
    true_beats_shuf = [
        bool(s["true_top_beats_shuffled_mean"]) for s in valid_shuffles
    ]

    # Oracle baseline diagnostics.
    valid_oracles = [
        t["perturbation_oracle_baseline"] for t in per_task
        if t.get("perturbation_oracle_baseline", {}).get("valid")
    ]
    oracle_ratios = [o["explanation_vs_oracle_ratio"] for o in valid_oracles]
    oracle_match = [bool(o["explanation_matches_oracle"]) for o in valid_oracles]
    oracle_ranks = [
        o["explanation_rank_among_all_perturbations"]
        for o in valid_oracles
        if o.get("explanation_rank_among_all_perturbations") is not None
    ]
    oracle_top_drops = [o["oracle_top_drop"] for o in valid_oracles]

    # Archetype distribution (when contrast-V2-style metadata is present).
    intended_dist: Dict[str, int] = {}
    actual_dist: Dict[str, int] = {}
    winner_index_dist: Dict[str, int] = {}
    for t in per_task:
        intended = t.get("intended_winner_archetype")
        actual = t.get("winner_archetype")
        if intended:
            intended_dist[intended] = intended_dist.get(intended, 0) + 1
        if actual:
            actual_dist[actual] = actual_dist.get(actual, 0) + 1
        wid = (t.get("winner_id") or "").rsplit("_", 1)
        if len(wid) == 2:
            winner_index_dist[wid[-1]] = winner_index_dist.get(wid[-1], 0) + 1
    archetype_dist: Optional[Dict[str, Any]] = None
    if intended_dist or actual_dist:
        archetype_dist = {
            "intended": intended_dist,
            "actual": actual_dist,
            "winner_candidate_index": winner_index_dist,
        }

    return {
        "mean_winner_M": _safe_mean(winner_M),
        "mean_loser_M": _safe_mean(loser_M),
        "mean_score_gap": _safe_mean(gaps),

        "mean_top_factor_drop": _safe_mean(top_drops),
        "mean_random_factor_drop": _safe_mean(rand_drops),
        "mean_bottom_factor_drop": _safe_mean(bot_drops),

        "frac_top_beats_random": _safe_frac(top_beats_rand),
        "frac_top_beats_bottom": _safe_frac(top_beats_bot),

        "counterfactual_valid_rate": (
            float(len(valid_cfs) / cf_attempts) if cf_attempts > 0 else None
        ),
        "mean_counterfactual_edit_size": _safe_mean(cf_edit_sizes),

        "num_valid_deletion_tests": len(valid_deletions),
        "num_valid_counterfactuals": len(valid_cfs),

        "mean_frac_candidates_scap_eq_1": _safe_mean(sat_fracs),
        "mean_winner_loser_scap_gap": _safe_mean(wl_scap_gaps),

        "top_rank_flip_rate": top_flip_rate,
        "random_rank_flip_rate": random_flip_rate,
        "bottom_rank_flip_rate": bottom_flip_rate,

        "mean_shuffled_top_drop": _safe_mean(shuf_means),
        "mean_true_top_drop_percentile": _safe_mean(shuf_percentiles),
        "frac_true_top_beats_shuffled_mean": _safe_frac(true_beats_shuf),
        "num_valid_shuffled_baselines": len(valid_shuffles),

        "mean_oracle_top_drop": _safe_mean(oracle_top_drops),
        "mean_explanation_vs_oracle_ratio": _safe_mean(oracle_ratios),
        "frac_explanation_matches_oracle": _safe_frac(oracle_match),
        "mean_explanation_rank_among_all_perturbations": _safe_mean(oracle_ranks),
        "num_valid_oracle_baselines": len(valid_oracles),

        "archetype_distribution": archetype_dist,
    }


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------

def _resolve_seeds(args) -> List[int]:
    """--seeds wins, then --n-seeds (consecutive from --seed), else [--seed]."""
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.n_seeds and args.n_seeds > 1:
        return list(range(args.seed, args.seed + args.n_seeds))
    return [args.seed]


def _run_one_seed(
    tasks: List[dict],
    args,
    world_model: WorldModel,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_task: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for i, task_entry in enumerate(tasks, start=1):
        res = evaluate_task(
            task_entry,
            pool_size=args.pool_size,
            seed=seed,
            world_model=world_model,
        )
        if res is None:
            tid = task_entry.get("task", {}).get("task_id", f"task_{i}")
            skipped.append({"task_id": tid, "reason": "unknown"})
            print(f"  [{i:>2}/{len(tasks)}] SKIPPED (unknown reason)")
            continue
        if "_skip_reason" in res:
            skipped.append({
                "task_id": res["task_id"],
                "reason": res["_skip_reason"],
            })
            print(
                f"  [{i:>2}/{len(tasks)}] {res['task_id']}: "
                f"SKIPPED ({res['_skip_reason']})"
            )
            continue
        per_task.append(res)
        deletion = res["deletion_test"]
        cf = res["counterfactual"]
        ob = res["perturbation_oracle_baseline"]
        print(
            f"  [{i:>2}/{len(tasks)}] {res['task_id']}: "
            f"winner={res['winner_id']} loser={res['loser_id']} "
            f"M_w={res['winner_decomposition']['M']:.3f} "
            f"gap={res['contrastive_explanation']['score_gap']:+.3f} "
            f"top_drop={deletion.get('top_factor_drop', 0.0):.3f} "
            f"rand_drop={deletion.get('random_factor_drop', 0.0):.3f} "
            f"oracle_r={ob.get('explanation_vs_oracle_ratio') if ob.get('valid') else None} "
            f"cf_edit={cf.get('edit_size')}"
        )
    return per_task, skipped


def _aggregate_mean_std(per_seed_summary: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-key mean ± std over seed summaries. Skips non-numeric values."""
    if not per_seed_summary:
        return {}
    keys = next(iter(per_seed_summary.values())).keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = []
        for s in per_seed_summary.values():
            v = s.get(k)
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and (math.isnan(v) or math.isinf(v))
            ):
                vals.append(float(v))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": int(arr.size),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks", type=int, default=10,
                        help="Number of tasks to run (0 = all).")
    parser.add_argument("--pool-size", type=int, default=5,
                        help="Candidates per task (0 = all).")
    parser.add_argument("--encoder", choices=("simple", "sbert"),
                        default="simple",
                        help="Embedding backend.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma-separated seed list (overrides --seed).")
    parser.add_argument("--n-seeds", type=int, default=0,
                        help="If >1, use seed, seed+1, ... (overrides --seed).")
    parser.add_argument("--testset", type=str,
                        default=str(DEFAULT_TESTSET),
                        help="Path to testset JSON.")
    parser.add_argument("--tag", type=str, default="faithfulness",
                        help="Tag appended to output filename.")
    parser.add_argument("--out", type=str, default="",
                        help="Optional explicit output path.")
    args = parser.parse_args(argv)

    ctx = build_encoder(args.encoder)
    # Keep run_interpretability's builders in sync with this encoder /
    # MatchConfig — they read from module-level globals (ENC, CFG).
    ri.configure_experiment(ctx)

    testset_path = Path(args.testset)
    data = json.loads(testset_path.read_text())
    tasks = data["tasks"]
    if args.num_tasks and args.num_tasks > 0:
        tasks = tasks[: args.num_tasks]

    world_model = WorldModel(
        config=ctx.cfg,
        theta_c=ri.EVAL_THETA_C,
        theta_n=ri.EVAL_THETA_N,
    )

    seeds = _resolve_seeds(args)
    print(
        f"[faithfulness] testset={testset_path.name}  encoder={ctx.encoder_name}  "
        f"n_tasks={len(tasks)}  pool={args.pool_size}  seeds={seeds}  "
        f"tag={args.tag}"
    )

    per_seed_per_task: Dict[int, List[Dict[str, Any]]] = {}
    per_seed_skipped: Dict[int, List[Dict[str, Any]]] = {}
    per_seed_summary: Dict[int, Dict[str, Any]] = {}
    per_seed_errors: Dict[int, List[Dict[str, Any]]] = {}

    for seed in seeds:
        print()
        print(f"--- seed={seed} ---")
        pt, sk = _run_one_seed(tasks, args, world_model, seed)
        per_seed_per_task[seed] = pt
        per_seed_skipped[seed] = sk
        s_summary = aggregate(pt)
        s_errors = _collect_error_cases(pt)
        s_summary["num_error_cases"] = len(s_errors)
        s_summary["error_case_rate"] = (
            float(len(s_errors) / max(len(pt), 1))
        )
        error_type_counts: Dict[str, int] = {}
        for e in s_errors:
            et = e["error_type"]
            error_type_counts[et] = error_type_counts.get(et, 0) + 1
        s_summary["error_type_counts"] = error_type_counts
        per_seed_summary[seed] = s_summary
        per_seed_errors[seed] = s_errors

    # The primary (single-seed-compatible) summary uses the first seed's
    # results — keeps the old JSON shape intact for any consumer that
    # never went multi-seed.
    primary_seed = seeds[0]
    per_task = per_seed_per_task[primary_seed]
    skipped = per_seed_skipped[primary_seed]
    summary = per_seed_summary[primary_seed]
    error_cases = per_seed_errors[primary_seed]

    metadata: Dict[str, Any] = {
        "testset": str(testset_path),
        "num_tasks_requested": args.num_tasks,
        "num_tasks_run": len(per_task),
        "num_tasks_skipped": len(skipped),
        "pool_size": args.pool_size,
        "encoder": ctx.encoder_name,
        "seed": args.seed,
        "seeds": seeds,
        "theta": [ri.EVAL_THETA_C, ri.EVAL_THETA_N],
    }

    payload: Dict[str, Any] = {
        "metadata": metadata,
        "summary": summary,
        "per_task": per_task,
        "skipped": skipped,
        "error_cases": error_cases,
    }
    if len(seeds) > 1:
        payload["per_seed"] = {
            str(s): {
                "summary": per_seed_summary[s],
                "error_cases": per_seed_errors[s],
                "num_skipped": len(per_seed_skipped[s]),
            }
            for s in seeds
        }
        payload["summary_mean_std"] = _aggregate_mean_std(per_seed_summary)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = (
        Path(args.out) if args.out
        else LOGS_DIR / f"faithfulness_{args.tag}_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_sanitize(payload), indent=2))

    def _fmt(v: Any) -> str:
        if v is None:
            return "None"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    print()
    print("Faithfulness summary (primary seed = {} ):".format(primary_seed))
    print(f"  tasks_run = {len(per_task)}")
    print(f"  tasks_skipped = {len(skipped)}")
    print(f"  mean_winner_M = {_fmt(summary['mean_winner_M'])}")
    print(f"  mean_loser_M = {_fmt(summary['mean_loser_M'])}")
    print(f"  mean_score_gap = {_fmt(summary['mean_score_gap'])}")
    print(f"  mean_top_factor_drop = {_fmt(summary['mean_top_factor_drop'])}")
    print(f"  mean_random_factor_drop = {_fmt(summary['mean_random_factor_drop'])}")
    print(f"  mean_bottom_factor_drop = {_fmt(summary['mean_bottom_factor_drop'])}")
    print(f"  frac_top_beats_random = {_fmt(summary['frac_top_beats_random'])}")
    print(f"  frac_top_beats_bottom = {_fmt(summary['frac_top_beats_bottom'])}")
    print(f"  counterfactual_valid_rate = {_fmt(summary['counterfactual_valid_rate'])}")
    print(f"  mean_counterfactual_edit_size = "
          f"{_fmt(summary['mean_counterfactual_edit_size'])}")
    print()
    print("Additional diagnostics:")
    print(f"  mean_frac_candidates_scap_eq_1 = "
          f"{_fmt(summary['mean_frac_candidates_scap_eq_1'])}")
    print(f"  mean_winner_loser_scap_gap = "
          f"{_fmt(summary['mean_winner_loser_scap_gap'])}")
    print(f"  top_rank_flip_rate = {_fmt(summary['top_rank_flip_rate'])}")
    print(f"  random_rank_flip_rate = {_fmt(summary['random_rank_flip_rate'])}")
    print(f"  bottom_rank_flip_rate = {_fmt(summary['bottom_rank_flip_rate'])}")
    print(f"  mean_shuffled_top_drop = "
          f"{_fmt(summary['mean_shuffled_top_drop'])}")
    print(f"  mean_true_top_drop_percentile = "
          f"{_fmt(summary['mean_true_top_drop_percentile'])}")
    print(f"  frac_true_top_beats_shuffled_mean = "
          f"{_fmt(summary['frac_true_top_beats_shuffled_mean'])}")
    print()
    print("Oracle baseline:")
    print(f"  mean_explanation_vs_oracle_ratio = "
          f"{_fmt(summary['mean_explanation_vs_oracle_ratio'])}")
    print(f"  frac_explanation_matches_oracle = "
          f"{_fmt(summary['frac_explanation_matches_oracle'])}")
    print(f"  mean_explanation_rank_among_all_perturbations = "
          f"{_fmt(summary['mean_explanation_rank_among_all_perturbations'])}")
    print()
    print("Error analysis:")
    print(f"  num_error_cases = {summary['num_error_cases']}")
    print(f"  error_case_rate = {_fmt(summary['error_case_rate'])}")
    if summary.get("error_type_counts"):
        for k, v in summary["error_type_counts"].items():
            print(f"    {k}: {v}")

    if summary.get("archetype_distribution"):
        print()
        print("Archetype distribution:")
        ad = summary["archetype_distribution"]
        print(f"  intended winners: {ad.get('intended')}")
        print(f"  actual winners:   {ad.get('actual')}")
        print(f"  winner candidate index: {ad.get('winner_candidate_index')}")

    if len(seeds) > 1:
        print()
        print("Mean ± std across seeds:")
        for k in (
            "mean_score_gap",
            "mean_top_factor_drop",
            "mean_random_factor_drop",
            "frac_top_beats_random",
            "top_rank_flip_rate",
            "random_rank_flip_rate",
            "mean_true_top_drop_percentile",
            "mean_explanation_vs_oracle_ratio",
            "frac_explanation_matches_oracle",
            "counterfactual_valid_rate",
        ):
            cell = payload["summary_mean_std"].get(k)
            if not cell:
                continue
            print(f"  {k:<46s} {cell['mean']:.4f} ± {cell['std']:.4f}"
                  f"  [{cell['min']:.4f}, {cell['max']:.4f}]")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
