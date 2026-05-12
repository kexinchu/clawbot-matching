"""Compare the 20-task evaluation across multiple noise levels.

Pass two or more eval JSON suffixes (e.g. 'none_<ts>', 'gaussian0.2_<ts>',
'uniform_<ts>'); the script produces a 2x2 figure summarising how the three
conditions (online_ucb, scap_greedy, random) degrade as priors get noisier.

  Usage:
    python plots/plot_20_tasks_noise_comparison.py \
        none_20260512-062916 \
        gaussian0.2_20260512-062918 \
        uniform_20260512-062920
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
    "scap_greedy": "S_cap greedy",
    "random": "Random",
}


def _load(arg: str) -> Tuple[str, dict]:
    p = Path(arg)
    if not p.exists():
        p = LOGS_DIR / f"online_learning_20tasks_eval_{arg}.json"
    if not p.exists():
        raise FileNotFoundError(arg)
    data = json.loads(p.read_text())
    meta = data["metadata"]
    nm = meta.get("noise_mode", "?")
    ns = meta.get("noise_sigma", 0.0)
    if nm == "gaussian":
        tag = f"gaussian σ={ns}"
    elif nm == "uniform":
        tag = "uniform random"
    else:
        tag = "true priors"
    return tag, data


def _per_task_vals(per_task: List[dict], cond: str, key: str) -> List[float]:
    if cond == "random":
        key = {
            "n_rounds_consensus": "n_rounds_consensus_mean",
            "n_rounds_first_optimal": "n_rounds_first_optimal_mean",
            "rho_mode": "rho_mode_mean",
            "rho_last": "rho_last_mean",
        }[key]
    return [float(t["conditions"][cond][key]) for t in per_task]


def _grouped_means_bars(
    ax: plt.Axes,
    runs: List[Tuple[str, dict]],
    metric: str,
    title: str,
    ylabel: str,
    ylim=None,
):
    """Bars grouped by noise level; one bar per condition."""
    tags = [t for t, _ in runs]
    n_runs = len(tags)
    n_cond = len(CONDITIONS)
    width = 0.8 / n_cond
    x = np.arange(n_runs)
    for i, cond in enumerate(CONDITIONS):
        means = []
        stds = []
        for _, data in runs:
            vals = _per_task_vals(data["per_task"], cond, metric)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        ax.bar(x + (i - (n_cond - 1) / 2) * width, means,
               yerr=stds, capsize=3, width=width,
               color=COLORS[cond], label=LABELS[cond],
               error_kw={"alpha": 0.5, "linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=9)
    ax.set_xlabel("Prior noise")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="best")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_args", nargs="+",
                        help="Two or more eval JSON suffixes/paths to compare.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: plots/online_learning_20tasks_noise_comparison.png)")
    args = parser.parse_args()

    runs = [_load(a) for a in args.eval_args]
    if len(runs) < 2:
        raise SystemExit("Need at least two runs to compare.")

    n_max = runs[0][1]["metadata"]["n_max_rounds"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _grouped_means_bars(
        axes[0, 0], runs, "n_rounds_consensus",
        title=f"Rounds to Consensus  (cap={n_max}, lower=better)",
        ylabel="rounds",
        ylim=(0, n_max + 5),
    )
    _grouped_means_bars(
        axes[0, 1], runs, "n_rounds_first_optimal",
        title=f"Rounds to First Optimal Pick  (cap={n_max}, lower=better)",
        ylabel="rounds",
        ylim=(0, n_max + 5),
    )
    _grouped_means_bars(
        axes[1, 0], runs, "rho_mode",
        title="rho_mode = M(most-frequent pick) / M_optimum   (higher=better)",
        ylabel="rho",
        ylim=(0.7, 1.05),
    )
    _grouped_means_bars(
        axes[1, 1], runs, "rho_last",
        title="rho_last = M(last selected pick) / M_optimum   (higher=better)",
        ylabel="rho",
        ylim=(0.7, 1.05),
    )

    # Build a concise textual summary
    lines = ["Summary (mean over 20 tasks):"]
    for tag, data in runs:
        lines.append(f"  [{tag}]")
        for cond in CONDITIONS:
            s = data["summary"][cond]
            lines.append(
                f"    {LABELS[cond]:<13}  n_first_opt={s['n_rounds_first_optimal']['mean']:>5.2f}  "
                f"rho_mode={s['rho_mode']['mean']:.3f}  "
                f"rho_last={s['rho_last']['mean']:.3f}"
            )
    fig.suptitle("20-Task Evaluation — Effect of Prior Noise", fontsize=12)
    fig.text(0.5, 0.95, "\n".join(lines),
             family="monospace", fontsize=8.5, ha="center", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    out_path = Path(args.out) if args.out else PLOTS_DIR / "online_learning_20tasks_noise_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
