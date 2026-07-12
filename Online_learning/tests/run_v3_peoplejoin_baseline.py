"""V3 PeopleJoin-Reactive baseline runner.

Compares:
  - lappas_coverage
  - mapscore_greedy
  - peoplejoin_reactive

Uses the same v3 tasks, candidate pools, public profiles, hidden simulator
outcome oracle, matching metrics, and reward evaluator as the v3 dreaming
ablation. PeopleJoin never uses MapScore / UCB / Dreaming / oracle latents for
selection; those appear only as post-hoc diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
ONLINE_LEARNING_DIR = TESTS_DIR.parent
REPO_ROOT = ONLINE_LEARNING_DIR.parent
MAPPING_ALGO_DIR = REPO_ROOT / "mapping-algo"
LOGS_DIR = REPO_ROOT / "logs"
EXPERIMENTS_DIR = REPO_ROOT / "Experiments"
DEFAULT_TESTSET = REPO_ROOT / "simulator" / "20_Tasks_Testset_v3_tiered.json"

for path in (TESTS_DIR, ONLINE_LEARNING_DIR, MAPPING_ALGO_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from WorldModel import WorldModel  # noqa: E402
from datatypes import UserState, Task  # noqa: E402

from run_20_tasks_evaluation import (  # noqa: E402
    CFG,
    EVAL_THETA_C,
    EVAL_THETA_N,
    build_learning_state_from_priors,
    build_requester_for_match,
    build_task,
    compute_ground_truth_M,
    run_lappas_coverage,
    run_mapscore_greedy,
)
from run_v3_dreaming_ablation import (  # noqa: E402
    OUTCOME_FIELDS,
    analytical_matches,
    compute_bilateral_outcome,
    selected_match_metrics,
    select_top1,
)

from Experiments.peoplejoin.llm_client import PeopleJoinLLMClient  # noqa: E402
from Experiments.peoplejoin.reactive_controller import (  # noqa: E402
    PeopleJoinBudget,
    run_peoplejoin_reactive,
)


COST_FIELDS = [
    "search_calls",
    "candidate_messages",
    "unique_candidates_contacted",
    "controller_api_calls",
    "responder_api_calls",
    "total_api_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "wall_clock_seconds",
]


def evaluate_lappas(
    task_entry: dict,
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
    m_true_per_cand: Dict[str, float],
    m_optimum: float,
) -> dict:
    started = time.perf_counter()
    _, selected_id, _, _ = run_lappas_coverage(candidate_pool, task)
    wall = time.perf_counter() - started
    matches = analytical_matches(requester, candidate_pool, task, "mapscore")
    outcome, process = compute_bilateral_outcome(task_entry, selected_id)
    rho = m_true_per_cand[selected_id] / m_optimum if m_optimum > 0 else 0.0
    mapscore_top1 = select_top1(matches)
    return {
        "method": "lappas_coverage",
        "selected_candidate_id": selected_id,
        "rho_last": round(float(rho), 4),
        "rho_mode": round(float(rho), 4),
        "selected_match": selected_match_metrics(matches[selected_id]),
        "equals_mapscore_greedy_top1": selected_id == mapscore_top1,
        "outcome": outcome,
        "process": process,
        "interaction_cost": {f: 0.0 for f in COST_FIELDS} | {"wall_clock_seconds": round(wall, 6)},
        "retrieval_diagnostics": {},
    }


def evaluate_mapscore_greedy(
    task_entry: dict,
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
    m_true_per_cand: Dict[str, float],
    m_optimum: float,
) -> dict:
    started = time.perf_counter()
    _, selected_id, _, scored = run_mapscore_greedy(requester, candidate_pool, task)
    wall = time.perf_counter() - started
    matches = scored["matches"]
    outcome, process = compute_bilateral_outcome(task_entry, selected_id)
    rho = m_true_per_cand[selected_id] / m_optimum if m_optimum > 0 else 0.0
    return {
        "method": "mapscore_greedy",
        "selected_candidate_id": selected_id,
        "rho_last": round(float(rho), 4),
        "rho_mode": round(float(rho), 4),
        "selected_match": selected_match_metrics(matches[selected_id]),
        "equals_mapscore_greedy_top1": True,
        "outcome": outcome,
        "process": process,
        "interaction_cost": {f: 0.0 for f in COST_FIELDS} | {"wall_clock_seconds": round(wall, 6)},
        "retrieval_diagnostics": {},
    }


def evaluate_peoplejoin(
    task_entry: dict,
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
    m_true_per_cand: Dict[str, float],
    m_optimum: float,
    *,
    seed: int,
    mode: str,
    model: str,
    base_url: str,
    temperature: float,
    budget: PeopleJoinBudget,
    oracle_best_id: str,
) -> dict:
    llm = PeopleJoinLLMClient(
        mode=mode,
        model=model,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
        allow_mock_fallback=(mode == "mock"),
    )
    result = run_peoplejoin_reactive(
        task_entry,
        llm=llm,
        budget=budget,
        seed=seed,
        mode=mode,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )
    selected_id = result.selected_candidate_id
    # Post-hoc MapScore diagnostics only — not used for selection.
    matches = analytical_matches(requester, candidate_pool, task, "mapscore")
    mapscore_top1 = select_top1(matches)
    outcome, process = compute_bilateral_outcome(task_entry, selected_id)
    rho = m_true_per_cand[selected_id] / m_optimum if m_optimum > 0 else 0.0

    first_top5 = result.retrieval_diagnostics.get("first_search_top5", [])
    retrieval = dict(result.retrieval_diagnostics)
    retrieval.update(
        {
            "selected_in_first_search_top5": selected_id in first_top5,
            "first_search_recall_at_5_selected": float(selected_id in first_top5),
            "first_search_recall_at_5_oracle_best": float(oracle_best_id in first_top5),
            "oracle_best_candidate_id": oracle_best_id,
            "mapscore_greedy_top1": mapscore_top1,
        }
    )
    return {
        "method": "peoplejoin_reactive",
        "seed": seed,
        "selected_candidate_id": selected_id,
        "rationale": result.rationale,
        "rho_last": round(float(rho), 4),
        "rho_mode": round(float(rho), 4),
        "selected_match": selected_match_metrics(matches[selected_id]),
        "equals_mapscore_greedy_top1": selected_id == mapscore_top1,
        "outcome": outcome,
        "process": process,
        "interaction_cost": result.interaction_cost,
        "retrieval_diagnostics": retrieval,
        "trace": result.trace,
        "finish_fallback": result.finish_fallback,
        "api_degraded": bool(result.interaction_cost.get("api_degraded", False)),
        "errors": result.errors,
        "mock_mode": result.mock_mode,
    }


def evaluate_task(
    task_entry: dict,
    *,
    seeds: List[int],
    mode: str,
    model: str,
    base_url: str,
    temperature: float,
    budget: PeopleJoinBudget,
) -> dict:
    task_dict = task_entry["task"]
    task = build_task(task_dict)
    requester = build_requester_for_match(task_entry["proposer_profile"])
    candidate_entries = task_entry["candidates"]
    candidate_pool = [build_learning_state_from_priors(c) for c in candidate_entries]
    m_true_per_cand, optimum_id, m_optimum = compute_ground_truth_M(
        requester,
        candidate_entries,
        task,
    )
    results = {
        "task_id": task_dict["task_id"],
        "title": task_dict.get("title", ""),
        "optimum_candidate_id": optimum_id,
        "M_optimum": round(float(m_optimum), 4),
        "conditions": {},
    }
    results["conditions"]["lappas_coverage"] = evaluate_lappas(
        task_entry, requester, candidate_pool, task, m_true_per_cand, m_optimum
    )
    results["conditions"]["mapscore_greedy"] = evaluate_mapscore_greedy(
        task_entry, requester, candidate_pool, task, m_true_per_cand, m_optimum
    )
    pj_runs = []
    for seed in seeds:
        pj_runs.append(
            evaluate_peoplejoin(
                task_entry,
                requester,
                candidate_pool,
                task,
                m_true_per_cand,
                m_optimum,
                seed=seed,
                mode=mode,
                model=model,
                base_url=base_url,
                temperature=temperature,
                budget=budget,
                oracle_best_id=optimum_id,
            )
        )
    results["conditions"]["peoplejoin_reactive"] = pj_runs
    return results


def _values_for_condition(per_task: List[dict], method: str, field_path: List[str]) -> List[float]:
    vals: List[float] = []
    for task_result in per_task:
        cond = task_result["conditions"][method]
        runs = cond if method == "peoplejoin_reactive" else [cond]
        for run in runs:
            node: Any = run
            missing = False
            for key in field_path:
                if not isinstance(node, dict) or key not in node:
                    missing = True
                    break
                node = node[key]
            if missing:
                # Backward-compatible default for newly added diagnostic fields.
                vals.append(0.0)
            else:
                vals.append(float(node))
    return vals


def _summary_stats(vals: List[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(vals)), 4),
        "std": round(float(np.std(vals)), 4),
        "median": round(float(np.median(vals)), 4),
        "min": round(float(np.min(vals)), 4),
        "max": round(float(np.max(vals)), 4),
    }


def aggregate_results(per_task: List[dict]) -> dict:
    summary: Dict[str, Any] = {}
    for method in ("lappas_coverage", "mapscore_greedy", "peoplejoin_reactive"):
        method_summary: Dict[str, Any] = {
            "rho_last": _summary_stats(_values_for_condition(per_task, method, ["rho_last"])),
            "rho_mode": _summary_stats(_values_for_condition(per_task, method, ["rho_mode"])),
        }
        for field in ("S_cap", "S_need", "MapScore"):
            method_summary[field] = _summary_stats(
                _values_for_condition(per_task, method, ["selected_match", field])
            )
        method_summary["equals_mapscore_greedy_top1_rate"] = _summary_stats(
            _values_for_condition(per_task, method, ["equals_mapscore_greedy_top1"])
        )
        method_summary["outcome"] = {
            field: _summary_stats(_values_for_condition(per_task, method, ["outcome", field]))
            for field in OUTCOME_FIELDS
        }
        method_summary["interaction_cost"] = {
            field: _summary_stats(
                _values_for_condition(per_task, method, ["interaction_cost", field])
            )
            for field in COST_FIELDS
        }
        if method == "peoplejoin_reactive":
            method_summary["retrieval"] = {
                "selected_in_first_search_top5": _summary_stats(
                    _values_for_condition(
                        per_task, method, ["retrieval_diagnostics", "first_search_recall_at_5_selected"]
                    )
                ),
                "first_search_recall_at_5_oracle_best": _summary_stats(
                    _values_for_condition(
                        per_task,
                        method,
                        ["retrieval_diagnostics", "first_search_recall_at_5_oracle_best"],
                    )
                ),
                "unique_candidates_contacted": _summary_stats(
                    _values_for_condition(
                        per_task, method, ["interaction_cost", "unique_candidates_contacted"]
                    )
                ),
                "finish_fallback_rate": _summary_stats(
                    _values_for_condition(per_task, method, ["finish_fallback"])
                ),
                "api_degraded_rate": _summary_stats(
                    _values_for_condition(per_task, method, ["api_degraded"])
                ),
            }
            # malformed / invalid rates from diagnostics
            malformed = []
            invalid = []
            for task_result in per_task:
                for run in task_result["conditions"]["peoplejoin_reactive"]:
                    diag = run.get("retrieval_diagnostics", {})
                    malformed.append(float(diag.get("malformed_json_count", 0)))
                    invalid.append(float(diag.get("invalid_action_count", 0)))
            method_summary["retrieval"]["malformed_json_count"] = _summary_stats(malformed)
            method_summary["retrieval"]["invalid_action_count"] = _summary_stats(invalid)
        summary[method] = method_summary
    return summary


def write_markdown_summary(payload: dict, out_path: Path) -> None:
    summary = payload["summary"]
    baseline = summary["mapscore_greedy"]
    meta = payload["metadata"]
    lines = [
        "# V3 PeopleJoin-Reactive Baseline Summary",
        "",
        f"Generated at: `{meta['generated_at']}`",
        f"Testset: `{meta['testset']}`",
        f"PeopleJoin mode: `{meta['peoplejoin_mode']}`",
        f"Model: `{meta['model']}`",
        f"Seeds: `{meta['seeds']}`",
        f"Budget: search≤{meta['budget']['max_search']}, ask≤{meta['budget']['max_ask']}, "
        f"actions≤{meta['budget']['max_actions']}",
        "",
        "> Mock mode results are for code validation only and must not be reported as formal baseline numbers.",
        "",
        "## A. Matching Metrics",
        "",
        "| method | rho_last | delta vs MapScore | S_cap | S_need | MapScore | =mapscore top1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("lappas_coverage", "mapscore_greedy", "peoplejoin_reactive"):
        row = summary[method]
        delta = row["rho_last"]["mean"] - baseline["rho_last"]["mean"]
        lines.append(
            f"| {method} | {row['rho_last']['mean']:.4f} ± {row['rho_last']['std']:.4f} "
            f"| {delta:+.4f} | {row['S_cap']['mean']:.4f} | {row['S_need']['mean']:.4f} "
            f"| {row['MapScore']['mean']:.4f} "
            f"| {row['equals_mapscore_greedy_top1_rate']['mean']:.4f} |"
        )

    lines.extend([
        "",
        "## B. Simulator Outcomes",
        "",
        "| method | mutual | completion | req_sat | cand_sat | joint_reward | total_reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method in ("lappas_coverage", "mapscore_greedy", "peoplejoin_reactive"):
        out = summary[method]["outcome"]
        base_out = baseline["outcome"]

        def fmt_out(field: str) -> str:
            mean = out[field]["mean"]
            std = out[field]["std"]
            delta = mean - base_out[field]["mean"]
            return f"{mean:.4f} ± {std:.4f} ({delta:+.4f})"

        lines.append(
            f"| {method} | {fmt_out('mutual_accept_probability')} | "
            f"{fmt_out('completion_probability')} | "
            f"{fmt_out('requester_satisfaction')} | "
            f"{fmt_out('candidate_satisfaction')} | "
            f"{fmt_out('joint_reward')} | {fmt_out('total_reward')} |"
        )

    lines.extend([
        "",
        "Values are mean ± std; parentheses show delta relative to `mapscore_greedy`.",
        "",
        "## C. Interaction Cost (PeopleJoin)",
        "",
        "| metric | mean ± std |",
        "|---|---:|",
    ])
    pj_cost = summary["peoplejoin_reactive"]["interaction_cost"]
    for field in COST_FIELDS:
        s = pj_cost[field]
        lines.append(f"| {field} / task | {s['mean']:.4f} ± {s['std']:.4f} |")

    lines.extend([
        "",
        "## D. Retrieval Diagnostics (PeopleJoin)",
        "",
    ])
    ret = summary["peoplejoin_reactive"].get("retrieval", {})
    for key, label in (
        ("selected_in_first_search_top5", "selected in first BM25 top-5"),
        ("first_search_recall_at_5_oracle_best", "first-search recall@5 vs oracle-best"),
        ("unique_candidates_contacted", "unique candidates contacted"),
        ("finish_fallback_rate", "finish fallback rate"),
        ("malformed_json_count", "malformed JSON count / task"),
        ("invalid_action_count", "invalid action count / task"),
    ):
        s = ret.get(key, {"mean": 0.0, "std": 0.0})
        lines.append(f"- {label}: {s['mean']:.4f} ± {s['std']:.4f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--mode", choices=("mock", "api"), default="mock")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-search", type=int, default=3)
    parser.add_argument("--max-ask", type=int, default=6)
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--partial-output", type=Path, default=None)
    parser.add_argument("--resume-partial", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "api":
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "ERROR: --mode api requires OPENAI_API_KEY. "
                "Refusing to invent formal results. Use --mode mock for smoke tests.",
                file=sys.stderr,
            )
            return 2

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    data = json.loads(args.testset.read_text(encoding="utf-8"))
    tasks = data["tasks"][: args.max_tasks] if args.max_tasks and args.max_tasks > 0 else data["tasks"]
    budget = PeopleJoinBudget(
        max_search=args.max_search,
        max_ask=args.max_ask,
        max_actions=args.max_actions,
        search_top_k=args.search_top_k,
    )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    per_task: List[dict] = []
    completed_task_ids = set()
    if args.resume_partial and args.partial_output is not None and args.partial_output.exists():
        partial = json.loads(args.partial_output.read_text(encoding="utf-8"))
        per_task = list(partial.get("per_task", []))
        completed_task_ids = {r["task_id"] for r in per_task}
        print(f"Resuming from {args.partial_output}: {len(completed_task_ids)} completed.", flush=True)

    for idx, task_entry in enumerate(tasks, start=1):
        if task_entry["task"]["task_id"] in completed_task_ids:
            continue
        result = evaluate_task(
            task_entry,
            seeds=seeds,
            mode=args.mode,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            budget=budget,
        )
        per_task.append(result)
        mg = result["conditions"]["mapscore_greedy"]["selected_candidate_id"]
        la = result["conditions"]["lappas_coverage"]["selected_candidate_id"]
        pj = [r["selected_candidate_id"] for r in result["conditions"]["peoplejoin_reactive"]]
        print(
            f"[{idx:>2}/{len(tasks)}] {result['task_id']} "
            f"lappas={la} mapscore={mg} peoplejoin={pj}",
            flush=True,
        )
        if args.partial_output is not None:
            args.partial_output.parent.mkdir(parents=True, exist_ok=True)
            partial_payload = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "partial": True,
                    "num_tasks_completed": len(per_task),
                    "num_tasks_target": len(tasks),
                    "peoplejoin_mode": args.mode,
                    "api_key_logged": False,
                },
                "per_task": per_task,
            }
            args.partial_output.write_text(json.dumps(partial_payload, indent=2), encoding="utf-8")

    summary = aggregate_results(per_task)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"v3_peoplejoin_{args.mode}_n{len(tasks)}"
    out_json = LOGS_DIR / f"{tag}_{timestamp}.json"
    out_md = EXPERIMENTS_DIR / f"{tag}_{timestamp}.md"
    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "testset": str(args.testset),
            "peoplejoin_mode": args.mode,
            "model": args.model,
            "base_url": args.base_url,
            "seeds": seeds,
            "temperature": args.temperature,
            "budget": {
                "max_search": budget.max_search,
                "max_ask": budget.max_ask,
                "max_actions": budget.max_actions,
                "search_top_k": budget.search_top_k,
            },
            "num_tasks": len(tasks),
            "api_key_logged": False,
            "formal_baseline": args.mode == "api",
            "note": (
                "Mock mode is for code validation only; do not cite as formal baseline."
                if args.mode == "mock"
                else "API mode formal run."
            ),
        },
        "per_task": per_task,
        "summary": summary,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_summary(payload, out_md)

    print()
    print(f"Wrote JSON log to {out_json}")
    print(f"Wrote Markdown summary to {out_md}")
    for method in ("lappas_coverage", "mapscore_greedy", "peoplejoin_reactive"):
        s = summary[method]
        print(
            f"{method:<22} rho={s['rho_last']['mean']:.3f}±{s['rho_last']['std']:.3f} "
            f"mutual={s['outcome']['mutual_accept_probability']['mean']:.3f} "
            f"completion={s['outcome']['completion_probability']['mean']:.3f} "
            f"total={s['outcome']['total_reward']['mean']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
