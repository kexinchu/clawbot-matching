#!/usr/bin/env python3
"""Run batch Gale–Shapley stable matching: GS-SkillCoverage vs CoWeaver-DA.

Both methods share:
  - the same batch benchmark
  - the same feasible edge set
  - the same deferred-acceptance allocator
  - the same tie-breaking seed

Oracle Max-Weight Assignment is reported only as an upper bound.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OL = Path(__file__).resolve().parents[1]
MA = ROOT / "mapping-algo"
for p in (str(ROOT), str(OL), str(MA), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Experiments.stable_matching.batch_generator_utils import (  # noqa: E402
    load_batch_testset,
    verify_shared_profiles,
)
from Experiments.stable_matching.deferred_acceptance import deferred_acceptance  # noqa: E402
from Experiments.stable_matching.evaluation import (  # noqa: E402
    allocation_metrics,
    build_oracle_utilities,
    evaluate_matching_outcomes,
    mean_std,
    oracle_max_weight_assignment,
    outcome_derived_prefs,
    preference_diagnostics,
)
from Experiments.stable_matching.plot_batch_results import write_markdown_report  # noqa: E402
from Experiments.stable_matching.preferences import (  # noqa: E402
    METHOD_CW,
    METHOD_GS,
    build_preferences,
    collect_feasible_edges,
)
from simulator.generate_batch_matching_testset import build_batch_testset  # noqa: E402

ORACLE_METHOD = "Oracle-MaxWeight"
DEFAULT_SEEDS = (42, 123, 456)


def run_method_on_batch(
    batch: dict,
    method: str,
    *,
    feasible_edges: set[tuple[str, str]],
    tie_break_seed: int,
    outcome_seed: int,
    oracle_utils: dict[tuple[str, str], float] | None = None,
    oracle_task_prefs: dict | None = None,
    oracle_cand_prefs: dict | None = None,
) -> dict[str, Any]:
    capacity = int(batch.get("candidate_capacity", 1))
    prefs = build_preferences(
        method,
        batch,
        feasible_edges=feasible_edges,
        tie_break_seed=tie_break_seed,
    )
    da = deferred_acceptance(
        prefs.task_ids,
        prefs.candidate_ids,
        prefs.task_prefs,
        prefs.candidate_prefs,
        candidate_capacity=capacity,
        tie_break_seed=tie_break_seed,
    )
    alloc = allocation_metrics(
        batch,
        da.matching,
        prefs,
        candidate_capacity=capacity,
        oracle_task_prefs=oracle_task_prefs,
        oracle_cand_prefs=oracle_cand_prefs,
    )
    outcomes = evaluate_matching_outcomes(batch, da.matching, seed=outcome_seed)
    return {
        "method": method,
        "matching": da.matching,
        "candidate_loads": da.candidate_loads,
        "da_trace": da.trace,
        "da_stats": {
            "proposals": da.proposals,
            "accepts": da.accepts,
            "rejects": da.rejects,
            "replacements": da.replacements,
        },
        "allocation": alloc,
        "outcomes": {k: v for k, v in outcomes.items() if k != "pair_outcomes"},
        "pair_outcomes": outcomes.get("pair_outcomes", []),
        "preference_meta": {
            "task_prefs": prefs.task_prefs,
            "candidate_prefs": prefs.candidate_prefs,
            "task_scores": prefs.task_scores,
            "candidate_scores": prefs.candidate_scores,
            "diagnostics": prefs.diagnostics,
            "n_feasible_edges": len(prefs.feasible_edges),
        },
    }


def run_oracle_on_batch(
    batch: dict,
    *,
    utilities: dict[tuple[str, str], float],
    outcome_seed: int,
    oracle_task_prefs: dict,
    oracle_cand_prefs: dict,
) -> dict[str, Any]:
    capacity = int(batch.get("candidate_capacity", 1))
    task_ids = [t["task"]["task_id"] for t in batch["tasks"]]
    candidate_ids = [c["candidate_profile"]["user_id"] for c in batch["shared_candidates"]]
    matching, best_value = oracle_max_weight_assignment(
        task_ids,
        candidate_ids,
        utilities,
        candidate_capacity=capacity,
    )
    # Synthetic PreferenceBundle-like object is not needed; build minimal alloc stats.
    from Experiments.stable_matching.preferences import PreferenceBundle

    # Oracle does not use DA prefs; report stability under outcome-derived prefs.
    empty_prefs = PreferenceBundle(
        method=ORACLE_METHOD,
        task_ids=task_ids,
        candidate_ids=candidate_ids,
        task_prefs=oracle_task_prefs,
        candidate_prefs=oracle_cand_prefs,
        task_scores={},
        candidate_scores={},
        feasible_edges=set(utilities.keys()),
        diagnostics={"oracle_upper_bound": True},
    )
    alloc = allocation_metrics(
        batch,
        matching,
        empty_prefs,
        candidate_capacity=capacity,
        oracle_task_prefs=oracle_task_prefs,
        oracle_cand_prefs=oracle_cand_prefs,
    )
    outcomes = evaluate_matching_outcomes(batch, matching, seed=outcome_seed)
    return {
        "method": ORACLE_METHOD,
        "matching": matching,
        "oracle_objective": best_value,
        "is_oracle_upper_bound": True,
        "allocation": alloc,
        "outcomes": {k: v for k, v in outcomes.items() if k != "pair_outcomes"},
        "pair_outcomes": outcomes.get("pair_outcomes", []),
    }


def evaluate_batch(
    batch: dict,
    *,
    tie_break_seed: int = 0,
    outcome_seed: int = 42,
    include_oracle: bool = True,
) -> dict[str, Any]:
    errors = verify_shared_profiles(batch)
    if errors:
        raise AssertionError("Shared profile invariants failed:\n" + "\n".join(errors))

    feasible_edges = collect_feasible_edges(batch)
    utilities = build_oracle_utilities(batch, seed=outcome_seed)
    task_ids = [t["task"]["task_id"] for t in batch["tasks"]]
    candidate_ids = [c["candidate_profile"]["user_id"] for c in batch["shared_candidates"]]
    oracle_task_prefs, oracle_cand_prefs = outcome_derived_prefs(
        utilities, task_ids, candidate_ids
    )

    gs = run_method_on_batch(
        batch,
        METHOD_GS,
        feasible_edges=feasible_edges,
        tie_break_seed=tie_break_seed,
        outcome_seed=outcome_seed,
        oracle_utils=utilities,
        oracle_task_prefs=oracle_task_prefs,
        oracle_cand_prefs=oracle_cand_prefs,
    )
    cw = run_method_on_batch(
        batch,
        METHOD_CW,
        feasible_edges=feasible_edges,
        tie_break_seed=tie_break_seed,
        outcome_seed=outcome_seed,
        oracle_utils=utilities,
        oracle_task_prefs=oracle_task_prefs,
        oracle_cand_prefs=oracle_cand_prefs,
    )

    from Experiments.stable_matching.preferences import PreferenceBundle

    gs_prefs = PreferenceBundle(
        method=METHOD_GS,
        task_ids=task_ids,
        candidate_ids=candidate_ids,
        task_prefs=gs["preference_meta"]["task_prefs"],
        candidate_prefs=gs["preference_meta"]["candidate_prefs"],
        task_scores=gs["preference_meta"]["task_scores"],
        candidate_scores=gs["preference_meta"]["candidate_scores"],
        feasible_edges=feasible_edges,
        diagnostics=gs["preference_meta"]["diagnostics"],
    )
    cw_prefs = PreferenceBundle(
        method=METHOD_CW,
        task_ids=task_ids,
        candidate_ids=candidate_ids,
        task_prefs=cw["preference_meta"]["task_prefs"],
        candidate_prefs=cw["preference_meta"]["candidate_prefs"],
        task_scores=cw["preference_meta"]["task_scores"],
        candidate_scores=cw["preference_meta"]["candidate_scores"],
        feasible_edges=feasible_edges,
        diagnostics=cw["preference_meta"]["diagnostics"],
    )
    diag = preference_diagnostics(
        gs_prefs, cw_prefs, gs["matching"], cw["matching"], batch
    )

    result: dict[str, Any] = {
        "batch_id": batch["batch_id"],
        "n_feasible_edges": len(feasible_edges),
        "methods": {
            METHOD_GS: gs,
            METHOD_CW: cw,
        },
        "preference_diagnostics": diag,
    }
    if include_oracle:
        result["methods"][ORACLE_METHOD] = run_oracle_on_batch(
            batch,
            utilities=utilities,
            outcome_seed=outcome_seed,
            oracle_task_prefs=oracle_task_prefs,
            oracle_cand_prefs=oracle_cand_prefs,
        )
    return result


def _collect_metric(per_batch: list[dict], method: str, path: tuple[str, ...]) -> list[float]:
    vals = []
    for b in per_batch:
        node = b["methods"][method]
        for key in path:
            node = node[key]
        vals.append(float(node))
    return vals


def aggregate_results(
    per_batch: list[dict],
    *,
    seeds: list[int],
    meta: dict,
    tie_break_seed: int,
) -> dict[str, Any]:
    methods = [METHOD_GS, METHOD_CW, ORACLE_METHOD]
    outcome_keys = [
        "mean_mutual_accept_probability",
        "mean_completion_probability",
        "mean_requester_satisfaction",
        "mean_candidate_satisfaction",
        "mean_joint_reward",
        "mean_total_reward",
        "total_batch_reward",
        "min_matched_pair_total_reward",
    ]
    alloc_keys = [
        "matched_task_rate",
        "unmatched_task_rate",
        "candidate_utilization",
        "feasible_match_rate",
        "gate_violation_count",
        "mean_requester_rank",
        "mean_candidate_rank",
        "true_blocking_pair_rate",
    ]

    method_summary: dict[str, Any] = {}
    for method in methods:
        if method not in per_batch[0]["methods"]:
            continue
        block: dict[str, Any] = {}
        for key in outcome_keys:
            block[key] = mean_std(_collect_metric(per_batch, method, ("outcomes", key)))
        for key in alloc_keys:
            block[key] = mean_std(_collect_metric(per_batch, method, ("allocation", key)))
        # Stability rate
        stables = [
            1.0 if b["methods"][method]["allocation"]["assignment_stable_under_own_prefs"] else 0.0
            for b in per_batch
            if method in b["methods"]
        ]
        block["assignment_stable_rate"] = mean_std(stables)
        method_summary[method] = block

    diag_keys = [
        "mean_requester_ranking_kendall_tau",
        "n_tasks_assignment_differ",
        "gs_mean_matched_coverage",
        "cw_mean_matched_coverage",
        "gs_mean_matched_s_need",
        "cw_mean_matched_s_need",
    ]
    diag_summary = {}
    for key in diag_keys:
        vals = [float(b["preference_diagnostics"][key]) for b in per_batch]
        diag_summary[key] = mean_std(vals)

    return {
        "seeds": seeds,
        "num_batches": meta.get("num_batches"),
        "tasks_per_batch": meta.get("tasks_per_batch"),
        "candidates_per_batch": meta.get("candidates_per_batch"),
        "candidate_capacity": meta.get("candidate_capacity"),
        "tie_break_seed": tie_break_seed,
        "n_batch_evals": len(per_batch),
        "methods": method_summary,
        "preference_diagnostics": diag_summary,
    }


def run_seed(
    seed: int,
    *,
    testset_path: Path | None,
    num_batches: int,
    tasks_per_batch: int,
    candidates_per_batch: int,
    candidate_capacity: int,
    tie_break_seed: int,
    outcome_seed: int,
    max_batches: int | None,
    include_oracle: bool,
) -> tuple[dict, list[dict]]:
    if testset_path is not None and testset_path.exists():
        testset = load_batch_testset(testset_path)
        # If file seed differs, still use file content (caller selects path per seed).
    else:
        testset = build_batch_testset(
            seed=seed,
            num_batches=num_batches,
            tasks_per_batch=tasks_per_batch,
            candidates_per_batch=candidates_per_batch,
            candidate_capacity=candidate_capacity,
        )

    batches = testset["batches"]
    if max_batches is not None:
        batches = batches[:max_batches]

    per_batch = []
    for batch in batches:
        print(f"  [{seed}] evaluating {batch['batch_id']} ...", flush=True)
        per_batch.append(
            evaluate_batch(
                batch,
                tie_break_seed=tie_break_seed,
                outcome_seed=outcome_seed,
                include_oracle=include_oracle,
            )
        )
    return testset["metadata"], per_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testset",
        type=Path,
        default=None,
        help="Optional single testset JSON. If omitted, generate per --seeds.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated benchmark seeds (default: 42,123,456).",
    )
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--tasks-per-batch", type=int, default=5)
    parser.add_argument("--candidates-per-batch", type=int, default=7)
    parser.add_argument("--candidate-capacity", type=int, default=1)
    parser.add_argument("--tie-break-seed", type=int, default=0)
    parser.add_argument("--outcome-seed", type=int, default=42)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on batches per seed (smoke runs).",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip oracle max-weight upper bound (faster smoke).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional filename tag (default: timestamp).",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    t0 = time.time()

    all_per_batch: list[dict] = []
    seed_payloads: dict[str, Any] = {}
    meta_ref: dict = {}

    for seed in seeds:
        print(f"=== Seed {seed} ===", flush=True)
        testset_path = None
        if args.testset is not None:
            # Allow pattern: path with {seed} or a directory of seed files.
            p = Path(str(args.testset).format(seed=seed))
            if p.exists():
                testset_path = p
            elif args.testset.exists() and len(seeds) == 1:
                testset_path = args.testset

        # Prefer canonical seeded files if present.
        if testset_path is None:
            candidate = ROOT / "simulator" / f"Batch_Matching_Testset_v1_seed{seed}.json"
            if candidate.exists():
                testset_path = candidate

        meta, per_batch = run_seed(
            seed,
            testset_path=testset_path,
            num_batches=args.num_batches,
            tasks_per_batch=args.tasks_per_batch,
            candidates_per_batch=args.candidates_per_batch,
            candidate_capacity=args.candidate_capacity,
            tie_break_seed=args.tie_break_seed,
            outcome_seed=args.outcome_seed,
            max_batches=args.max_batches,
            include_oracle=not args.skip_oracle,
        )
        meta_ref = meta
        # Annotate seed on each batch result.
        for b in per_batch:
            b["benchmark_seed"] = seed
        all_per_batch.extend(per_batch)
        seed_payloads[str(seed)] = {
            "metadata": meta,
            "per_batch": per_batch,
        }

    summary = aggregate_results(
        all_per_batch,
        seeds=seeds,
        meta={
            "num_batches": args.max_batches or meta_ref.get("num_batches", args.num_batches),
            "tasks_per_batch": meta_ref.get("tasks_per_batch", args.tasks_per_batch),
            "candidates_per_batch": meta_ref.get(
                "candidates_per_batch", args.candidates_per_batch
            ),
            "candidate_capacity": meta_ref.get(
                "candidate_capacity", args.candidate_capacity
            ),
        },
        tie_break_seed=args.tie_break_seed,
    )
    summary["elapsed_seconds"] = round(time.time() - t0, 2)

    payload = {
        "metadata": {
            "runner": "Online_learning/tests/run_batch_stable_matching.py",
            "methods": [METHOD_GS, METHOD_CW, ORACLE_METHOD],
            "oracle_note": (
                "Oracle-MaxWeight uses hidden total_reward and is an upper bound only."
            ),
            "seeds": seeds,
            "tie_break_seed": args.tie_break_seed,
            "outcome_seed": args.outcome_seed,
            "max_batches": args.max_batches,
            "elapsed_seconds": summary["elapsed_seconds"],
        },
        "summary": summary,
        "per_seed": seed_payloads,
        "per_batch": all_per_batch,
    }

    json_path = args.output_dir / f"batch_stable_matching_{tag}.json"
    md_path = ROOT / "Experiments" / "stable_matching" / f"results_{tag}.md"
    # Also keep a copy under Experiments for the markdown summary requested.
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    commands = [
        "python simulator/generate_batch_matching_testset.py "
        "--seeds 42,123,456 --output simulator/Batch_Matching_Testset_v1.json",
        "python Online_learning/tests/run_batch_stable_matching.py "
        f"--seeds {','.join(map(str, seeds))} "
        f"--tie-break-seed {args.tie_break_seed} --tag {tag}",
        "pytest Online_learning/tests/test_stable_matching.py "
        "Online_learning/tests/test_batch_generator.py -q",
    ]
    write_markdown_report(summary, md_path, commands=commands)

    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")
    print("\nAggregate total_batch_reward:")
    for method, block in summary["methods"].items():
        m = block["total_batch_reward"]
        print(f"  {method}: {m['mean']:.4f} ± {m['std']:.4f}")


if __name__ == "__main__":
    main()
