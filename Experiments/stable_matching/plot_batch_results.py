"""Plot / table helpers for batch stable matching results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METHOD_ORDER = ("GaleShapley-SkillCoverage", "CoWeaver-DA", "Oracle-MaxWeight")


def _fmt_mean_std(stat: dict[str, float], digits: int = 4) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def results_table_markdown(summary: dict[str, Any]) -> str:
    """Build a markdown results table from aggregated summary."""
    methods = summary.get("methods", {})
    rows = []
    header = (
        "| method | total batch reward | mean total reward | "
        "matched task rate | mutual accept | completion | "
        "req sat | cand sat | min pair reward |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows.extend([header, sep])
    for method in METHOD_ORDER:
        if method not in methods:
            continue
        m = methods[method]
        label = method
        if method == "Oracle-MaxWeight":
            label = f"{method} (oracle upper bound)"
        rows.append(
            "| {method} | {tbr} | {mtr} | {mtr_rate} | {ma} | {cp} | {rs} | {cs} | {mn} |".format(
                method=label,
                tbr=_fmt_mean_std(m["total_batch_reward"]),
                mtr=_fmt_mean_std(m["mean_total_reward"]),
                mtr_rate=_fmt_mean_std(m["matched_task_rate"]),
                ma=_fmt_mean_std(m["mean_mutual_accept_probability"]),
                cp=_fmt_mean_std(m["mean_completion_probability"]),
                rs=_fmt_mean_std(m["mean_requester_satisfaction"]),
                cs=_fmt_mean_std(m["mean_candidate_satisfaction"]),
                mn=_fmt_mean_std(m["min_matched_pair_total_reward"]),
            )
        )
    return "\n".join(rows)


def write_markdown_report(
    summary: dict[str, Any],
    output_path: Path,
    *,
    commands: list[str] | None = None,
) -> None:
    lines = [
        "# Batch Stable Matching Results",
        "",
        f"- Benchmark seeds: `{summary.get('seeds')}`",
        f"- Batches per seed: `{summary.get('num_batches')}`",
        f"- Tasks / batch: `{summary.get('tasks_per_batch')}`",
        f"- Candidates / batch: `{summary.get('candidates_per_batch')}`",
        f"- Candidate capacity: `{summary.get('candidate_capacity')}`",
        f"- Tie-break seed: `{summary.get('tie_break_seed')}`",
        "",
        "## Aggregate (mean ± std across batches × seeds)",
        "",
        results_table_markdown(summary),
        "",
        "Oracle-MaxWeight is an **oracle upper bound** using hidden "
        "`total_reward`; it is not a fair method comparison target.",
        "",
        "## Preference diagnostics",
        "",
    ]
    diag = summary.get("preference_diagnostics", {})
    if diag:
        lines.append(
            f"- Mean requester ranking Kendall τ (GS vs CoWeaver-DA): "
            f"`{_fmt_mean_std(diag.get('mean_requester_ranking_kendall_tau', {'mean': 0, 'std': 0}))}`"
        )
        lines.append(
            f"- Mean #tasks with different assignment: "
            f"`{_fmt_mean_std(diag.get('n_tasks_assignment_differ', {'mean': 0, 'std': 0}))}`"
        )
        lines.append(
            f"- CoWeaver matched mean S_need vs GS: "
            f"`{_fmt_mean_std(diag.get('cw_mean_matched_s_need', {'mean': 0, 'std': 0}))}` vs "
            f"`{_fmt_mean_std(diag.get('gs_mean_matched_s_need', {'mean': 0, 'std': 0}))}`"
        )
        lines.append(
            f"- GS matched mean coverage vs CoWeaver: "
            f"`{_fmt_mean_std(diag.get('gs_mean_matched_coverage', {'mean': 0, 'std': 0}))}` vs "
            f"`{_fmt_mean_std(diag.get('cw_mean_matched_coverage', {'mean': 0, 'std': 0}))}`"
        )
        lines.append("")

    if commands:
        lines.extend(["## Reproducible commands", ""])
        for cmd in commands:
            lines.append(f"```bash\n{cmd}\n```")
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = load_summary(args.summary)
    write_markdown_report(summary, args.output)
    print(f"Wrote markdown report to {args.output}")


if __name__ == "__main__":
    main()
