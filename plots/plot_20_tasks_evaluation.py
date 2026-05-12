"""Plot the results of run_20_tasks_evaluation.py.

Reads the latest (or specified) `logs/online_learning_20tasks_eval_<ts>.json`
and writes a 2x2 figure with:
  - n_rounds to consensus (per task, grouped by condition)
  - n_rounds to first optimal pick (per task, grouped by condition)
  - rho_mode  (per task, grouped by condition)
  - rho_mode  distribution across tasks (boxplot per condition)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
PLOTS_DIR = REPO_ROOT / "plots"

CONDITIONS = ["online_ucb", "scap_greedy", "random"]
COLORS = {
    "online_ucb": "#1f77b4",
    "scap_greedy": "#ff7f0e",
    "random": "#2ca02c",
}
LABELS = {
    "online_ucb": "Online UCB",
    "scap_greedy": "S_cap greedy (no UCB)",
    "random": "Random (mean over seeds)",
}


def _find_latest_eval(arg: Optional[str]) -> Path:
    """Locate the eval JSON. `arg` may be a path, a timestamp, or a full
    suffix like 'gaussian0.2_20260512-063000'."""
    if arg:
        as_path = Path(arg)
        if as_path.exists():
            return as_path
        candidate = LOGS_DIR / f"online_learning_20tasks_eval_{arg}.json"
        if candidate.exists():
            return candidate
        # last-resort: glob match
        matches = sorted(LOGS_DIR.glob(f"online_learning_20tasks_eval_*{arg}*.json"))
        if matches:
            return matches[-1]
        raise FileNotFoundError(f"Could not find eval file for '{arg}'")
    matches = sorted(LOGS_DIR.glob("online_learning_20tasks_eval_*.json"))
    if not matches:
        raise FileNotFoundError("No 20-task evaluation files found in logs/")
    return matches[-1]


def _pick(value: dict, condition: str, metric: str) -> float:
    """Map condition+metric → field name in the per-task JSON."""
    if condition == "random":
        mapping = {
            "n_consensus": "n_rounds_consensus_mean",
            "n_first_optimal": "n_rounds_first_optimal_mean",
            "rho_mode": "rho_mode_mean",
            "rho_last": "rho_last_mean",
        }
    else:
        mapping = {
            "n_consensus": "n_rounds_consensus",
            "n_first_optimal": "n_rounds_first_optimal",
            "rho_mode": "rho_mode",
            "rho_last": "rho_last",
        }
    return float(value[condition][mapping[metric]])


def _grouped_bars(
    ax: plt.Axes,
    per_task: List[dict],
    metric: str,
    title: str,
    ylabel: str,
    ylim: Optional[tuple] = None,
):
    task_ids = [t["task_id"] for t in per_task]
    n_tasks = len(task_ids)
    n_cond = len(CONDITIONS)
    width = 0.8 / n_cond
    x = np.arange(n_tasks)
    for i, cond in enumerate(CONDITIONS):
        vals = [_pick(t["conditions"], cond, metric) for t in per_task]
        ax.bar(x + (i - (n_cond - 1) / 2) * width, vals,
               width=width, color=COLORS[cond], label=LABELS[cond])
    ax.set_xticks(x)
    ax.set_xticklabels([tid.replace("task_", "") for tid in task_ids],
                       rotation=0, fontsize=8)
    ax.set_xlabel("Task id")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="best")


def _boxplot(ax: plt.Axes, per_task: List[dict], metric: str, title: str, ylabel: str):
    box_data = []
    for cond in CONDITIONS:
        vals = [_pick(t["conditions"], cond, metric) for t in per_task]
        box_data.append(vals)
    bp = ax.boxplot(
        box_data,
        labels=[LABELS[c] for c in CONDITIONS],
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    for patch, cond in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(COLORS[cond])
        patch.set_alpha(0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.tick_params(axis="x", labelsize=8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "timestamp", nargs="?", default=None,
        help="Optional timestamp to plot (default: latest)",
    )
    args = parser.parse_args()

    eval_path = _find_latest_eval(args.timestamp)
    data = json.loads(eval_path.read_text())
    per_task = data["per_task"]
    summary = data["summary"]
    meta = data["metadata"]
    timestamp = eval_path.stem.split("_")[-1]
    n_max = meta["n_max_rounds"]
    noise_mode = meta.get("noise_mode", "none")
    noise_sigma = meta.get("noise_sigma", 0.0)
    noise_str = (
        f"noise={noise_mode}"
        + (f"(σ={noise_sigma})" if noise_mode == "gaussian" else "")
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    _grouped_bars(
        axes[0, 0], per_task, "n_consensus",
        title=f"Rounds to Consensus  (cap={n_max})",
        ylabel="rounds",
        ylim=(0, n_max + 2),
    )
    _grouped_bars(
        axes[0, 1], per_task, "n_first_optimal",
        title=f"Rounds to First Optimal Pick  (cap={n_max})",
        ylabel="rounds",
        ylim=(0, n_max + 2),
    )
    _grouped_bars(
        axes[1, 0], per_task, "rho_mode",
        title="Quality: rho_mode = M(most-frequent pick) / M_optimum",
        ylabel="rho",
        ylim=(0.5, 1.05),
    )
    _boxplot(
        axes[1, 1], per_task, "rho_mode",
        title="rho_mode distribution across tasks",
        ylabel="rho",
    )

    summary_lines = []
    for cond in CONDITIONS:
        s = summary[cond]
        summary_lines.append(
            f"{LABELS[cond]:<24}  n_cons={s['n_rounds_consensus']['mean']:>5.2f}  "
            f"n_first_opt={s['n_rounds_first_optimal']['mean']:>5.2f}  "
            f"rho_mode={s['rho_mode']['mean']:.3f}  rho_last={s['rho_last']['mean']:.3f}"
        )
    fig.suptitle(
        f"20-Task Evaluation ({timestamp})  |  {noise_str}\n" + "\n".join(summary_lines),
        fontsize=10, family="monospace",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out_path = PLOTS_DIR / f"online_learning_20tasks_eval_{eval_path.stem.replace('online_learning_20tasks_eval_', '')}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
