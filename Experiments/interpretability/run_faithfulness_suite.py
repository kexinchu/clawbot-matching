"""V2 suite — runs the full faithfulness benchmark matrix and writes a
top-level comparison JSON.

Conditions:

  1. simple + tiered (full 20)
  2. simple + contrast V1 (full 20)
  3. simple + contrast V2 (full 20)
  4. sbert  + contrast V2 (full 20)             — best-effort; recorded
                                                   as unavailable if
                                                   SBERT cannot load.
  5. simple + failure case (10 tasks)
  6. simple + contrast V2 multi-seed (5 seeds)

Each condition is driven by ``run_faithfulness.main`` so the per-task
JSON files in ``logs/`` are byte-identical to what the standalone
runner produces. This script only stitches a ``conditions`` dict
together and computes a few cross-condition booleans (e.g. is
contrast V2 less saturated than tiered? does SBERT preserve
faithfulness?).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
LOGS_DIR = _REPO / "logs"
SIM = _REPO / "simulator"
TIERED = SIM / "20_Tasks_Testset_tiered.json"
CONTRAST_V1 = SIM / "20_Tasks_FaithfulnessContrast.json"
CONTRAST_V2 = SIM / "20_Tasks_FaithfulnessContrastV2.json"
FAILURE = SIM / "20_Tasks_FaithfulnessFailureCase.json"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_faithfulness as rf  # noqa: E402
import generate_faithfulness_contrast_testset as gen_contrast_v1  # noqa: E402
import generate_faithfulness_contrast_v2 as gen_contrast_v2  # noqa: E402
import generate_faithfulness_failure_case as gen_failure  # noqa: E402


# ---------------------------------------------------------------------------
# Per-condition driver
# ---------------------------------------------------------------------------

def _ensure_testsets() -> None:
    if not CONTRAST_V1.is_file():
        print(f"[suite] missing {CONTRAST_V1.name} — generating")
        gen_contrast_v1.main()
    if not CONTRAST_V2.is_file():
        print(f"[suite] missing {CONTRAST_V2.name} — generating")
        gen_contrast_v2.main()
    if not FAILURE.is_file():
        print(f"[suite] missing {FAILURE.name} — generating")
        gen_failure.main()


def _run_condition(
    name: str,
    *,
    encoder: str,
    testset: Path,
    num_tasks: int,
    pool_size: int,
    seed: int,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Spawn one `run_faithfulness.main` invocation with the given config."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"suite_{name}"
    out_path = LOGS_DIR / f"faithfulness_{tag}_{ts}.json"
    argv: List[str] = [
        "--num-tasks", str(num_tasks),
        "--pool-size", str(pool_size),
        "--encoder", encoder,
        "--seed", str(seed),
        "--testset", str(testset),
        "--tag", tag,
        "--out", str(out_path),
    ]
    if seeds:
        argv.extend(["--seeds", ",".join(str(s) for s in seeds)])
    print(
        f"\n[suite] === {name} (encoder={encoder}, "
        f"testset={testset.name}, seeds={seeds or [seed]}) ==="
    )
    try:
        rc = rf.main(argv)
        if rc != 0:
            return {"available": False, "error": f"rc={rc}"}
        payload = json.loads(out_path.read_text())
        return {
            "available": True,
            "json_path": str(out_path),
            "summary": payload.get("summary", {}),
            "summary_mean_std": payload.get("summary_mean_std"),
            "metadata": payload.get("metadata", {}),
            "num_error_cases": len(payload.get("error_cases") or []),
        }
    except Exception as exc:  # noqa: BLE001 — keep suite alive
        tb = traceback.format_exc(limit=4)
        print(f"[suite] {name} failed: {exc}")
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
        }


# ---------------------------------------------------------------------------
# Cross-condition comparison
# ---------------------------------------------------------------------------

def _gt(a: Optional[float], b: Optional[float]) -> Optional[bool]:
    if a is None or b is None:
        return None
    return bool(a > b)


def _comparison(conditions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def _sum(name: str) -> Optional[Dict[str, Any]]:
        c = conditions.get(name) or {}
        return c.get("summary") if c.get("available") else None

    s_tier = _sum("simple_tiered") or {}
    s_v1 = _sum("simple_contrast_v1") or {}
    s_v2 = _sum("simple_contrast_v2") or {}
    sbert_v2 = _sum("sbert_contrast_v2")
    s_fail = _sum("simple_failure_case") or {}
    s_multi = _sum("simple_contrast_v2_multi_seed") or {}
    multi_meta = conditions.get("simple_contrast_v2_multi_seed", {}).get("summary_mean_std")

    archetype_counts = (s_v2.get("archetype_distribution") or {}).get("actual") or {}
    diverse = bool(archetype_counts and len(archetype_counts) >= 3)

    # Multi-seed stability: max std across the headline faithfulness
    # metrics. If any of them has std > 0.05 we consider the seed run
    # "unstable" (it doesn't mean the result is wrong, just noisy).
    stability_keys = (
        "mean_top_factor_drop",
        "mean_random_factor_drop",
        "top_rank_flip_rate",
        "mean_true_top_drop_percentile",
        "mean_explanation_vs_oracle_ratio",
    )
    stable_under_seeds: Optional[bool] = None
    if multi_meta:
        stds = [
            multi_meta[k]["std"]
            for k in stability_keys if k in multi_meta
        ]
        stable_under_seeds = bool(stds and max(stds) <= 0.05)

    # SBERT stability check: top_beats_random must stay above 0.7.
    faith_stable_sbert: Optional[bool] = None
    if sbert_v2:
        faith_stable_sbert = bool(
            (sbert_v2.get("frac_top_beats_random") or 0) >= 0.7
            and (sbert_v2.get("mean_top_factor_drop") or 0)
            > (sbert_v2.get("mean_random_factor_drop") or 0)
        )

    # Failure case sanity: top should NOT beat random consistently — if
    # it does, our diagnostic dataset isn't actually exposing the bug.
    failure_reproduced: Optional[bool] = None
    if s_fail:
        failure_reproduced = bool(
            (s_fail.get("frac_top_beats_random") or 0) < 0.5
            and (s_fail.get("mean_true_top_drop_percentile") or 0) < 0.8
        )

    # Oracle support: the explanation should land in the top-2 of the
    # oracle ranking on average, with high overall ratio.
    oracle_supports = None
    if s_v2:
        oracle_supports = bool(
            (s_v2.get("mean_explanation_vs_oracle_ratio") or 0) >= 0.8
            and (s_v2.get("mean_explanation_rank_among_all_perturbations")
                 or 99) <= 2.5
        )

    return {
        "tiered_has_scap_saturation": bool(
            (s_tier.get("mean_frac_candidates_scap_eq_1") or 0) >= 0.5
        ),
        "contrast_v1_reduces_saturation": _gt(
            s_tier.get("mean_frac_candidates_scap_eq_1"),
            s_v1.get("mean_frac_candidates_scap_eq_1"),
        ),
        "contrast_v2_reduces_saturation": _gt(
            s_tier.get("mean_frac_candidates_scap_eq_1"),
            s_v2.get("mean_frac_candidates_scap_eq_1"),
        ),
        "contrast_v2_has_archetype_diversity": diverse,
        "v2_increases_score_gap_vs_tiered": _gt(
            s_v2.get("mean_score_gap"),
            s_tier.get("mean_score_gap"),
        ),
        "sbert_available": bool(sbert_v2),
        "faithfulness_stable_under_sbert": faith_stable_sbert,
        "faithfulness_stable_across_seeds": stable_under_seeds,
        "failure_case_reproduced": failure_reproduced,
        "oracle_baseline_supports_explanation": oracle_supports,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks-full", type=int, default=20)
    parser.add_argument("--num-tasks-failure", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds-multi", type=int, default=5,
                        help="Seeds for the multi-seed condition.")
    parser.add_argument("--skip-sbert", action="store_true")
    args = parser.parse_args(argv)

    _ensure_testsets()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    conditions: Dict[str, Dict[str, Any]] = {}
    conditions["simple_tiered"] = _run_condition(
        "simple_tiered", encoder="simple", testset=TIERED,
        num_tasks=args.num_tasks_full, pool_size=args.pool_size, seed=args.seed,
    )
    conditions["simple_contrast_v1"] = _run_condition(
        "simple_contrast_v1", encoder="simple", testset=CONTRAST_V1,
        num_tasks=args.num_tasks_full, pool_size=args.pool_size, seed=args.seed,
    )
    conditions["simple_contrast_v2"] = _run_condition(
        "simple_contrast_v2", encoder="simple", testset=CONTRAST_V2,
        num_tasks=args.num_tasks_full, pool_size=args.pool_size, seed=args.seed,
    )
    if not args.skip_sbert:
        conditions["sbert_contrast_v2"] = _run_condition(
            "sbert_contrast_v2", encoder="sbert", testset=CONTRAST_V2,
            num_tasks=args.num_tasks_full, pool_size=args.pool_size, seed=args.seed,
        )
    conditions["simple_failure_case"] = _run_condition(
        "simple_failure_case", encoder="simple", testset=FAILURE,
        num_tasks=args.num_tasks_failure, pool_size=args.pool_size,
        seed=args.seed,
    )
    if args.n_seeds_multi and args.n_seeds_multi > 1:
        seeds = list(range(args.seed, args.seed + args.n_seeds_multi))
        conditions["simple_contrast_v2_multi_seed"] = _run_condition(
            "simple_contrast_v2_multi_seed",
            encoder="simple", testset=CONTRAST_V2,
            num_tasks=args.num_tasks_full, pool_size=args.pool_size,
            seed=args.seed, seeds=seeds,
        )

    comparison = _comparison(conditions)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = LOGS_DIR / f"faithfulness_suite_v2_{ts}.json"
    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_tasks_full": args.num_tasks_full,
            "num_tasks_failure": args.num_tasks_failure,
            "pool_size": args.pool_size,
            "seed": args.seed,
            "n_seeds_multi": args.n_seeds_multi,
            "skip_sbert": bool(args.skip_sbert),
        },
        "conditions": conditions,
        "comparison": comparison,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 78)
    print("Suite summary:")
    for name, c in conditions.items():
        if not c.get("available"):
            print(f"  {name}: UNAVAILABLE ({c.get('error', '?')})")
            continue
        s = c.get("summary", {}) or {}
        gap = s.get("mean_score_gap")
        sat = s.get("mean_frac_candidates_scap_eq_1")
        td = s.get("mean_top_factor_drop")
        ftr = s.get("frac_top_beats_random")
        tfr = s.get("top_rank_flip_rate")
        oratio = s.get("mean_explanation_vs_oracle_ratio")
        orank = s.get("mean_explanation_rank_among_all_perturbations")
        cf = s.get("counterfactual_valid_rate")
        ne = c.get("num_error_cases")
        print(
            f"  {name}: gap={gap}  sat={sat}  top_drop={td}  "
            f"top_beats_rand={ftr}  top_flip={tfr}  "
            f"oracle_r={oratio}  oracle_rank={orank}  cf={cf}  errs={ne}"
        )
    print("\ncomparison:", json.dumps(comparison, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
