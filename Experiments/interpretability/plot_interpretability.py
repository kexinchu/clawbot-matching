"""Render the 5 figures for the weight-interpretability experiment.

Reads the latest `logs/interpretability_*.json` (or a path passed via
--input) and writes 5 PNGs under
`Experiments/interpretability/plots/`:

    01_wj_vs_gap.png            — per-task w_j vs Gap-weighted importance,
                                   plus aggregate Top-1 hit bars
    02_attention_heatmaps.png    — 3 example tasks: α(req → v's cap) heatmap
                                   with GT-cap highlighted
    03_theta_scatter.png         — final w_c vs complement-vs-motivation
                                   ratio (per task), with regression line
    04_mu_convergence.png        — |μ - μ_true| over rounds for
                                   critical vs non-critical caps
    05_sigma_shrinkage.png       — σ over rounds for critical vs
                                   non-critical caps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
LOGS_DIR = _REPO_ROOT / "logs"
PLOTS_DIR = _HERE / "plots"


def _find_latest(arg: Optional[str]) -> Path:
    if arg:
        p = Path(arg)
        if p.exists():
            return p
        cands = sorted(LOGS_DIR.glob(f"interpretability_*{arg}*.json"))
        if cands:
            return cands[-1]
        raise FileNotFoundError(arg)
    cands = sorted(LOGS_DIR.glob("interpretability_*.json"))
    if not cands:
        raise FileNotFoundError("No interpretability_*.json under logs/")
    return cands[-1]


# ---------------------------------------------------------------------------
# Figure 1: w_j vs Gap-weighted importance
# ---------------------------------------------------------------------------

def plot_wj_vs_gap(data: Dict, out_path: Path) -> None:
    per = data["per_task"]
    summary = data["summary"]
    n = len(per)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1])

    ax_top = fig.add_subplot(gs[0, :])
    task_labels = [r["task_id"] for r in per]
    wj_top = [
        max(r["w_j_alignment"]["w_j"]) for r in per
    ]
    gap_top = []
    for r in per:
        gap_vals = r["w_j_alignment"]["gap_weighted"]
        total = sum(gap_vals)
        gap_top.append(max(gap_vals) / total if total > 0 else 0.0)
    width = 0.4
    xs = np.arange(n)
    ax_top.bar(xs - width / 2, wj_top, width, label="w_j top share")
    ax_top.bar(xs + width / 2, gap_top, width,
               label="Gap-weighted top share")
    ax_top.set_xticks(xs)
    ax_top.set_xticklabels(task_labels, rotation=70, fontsize=8)
    ax_top.set_ylabel("Share of total going to top-1 requirement")
    ax_top.set_title(
        "Per-task: which fraction does the 'most important' requirement own?\n"
        "(w_j = q_j/Σq vs. Gap-weighted q_j·Gap_j)"
    )
    ax_top.legend()
    ax_top.grid(axis="y", alpha=0.3)

    ax_hit = fig.add_subplot(gs[1, 0])
    hits = ["wj_top1_q", "wj_top1_gap_weighted", "wj_kendall_tau_vs_gap"]
    vals = [summary[k] for k in hits]
    labels = ["Top-1 w_j vs q_j\n(trivial bound)",
              "Top-1 w_j vs Gap-weighted",
              "Kendall τ(w_j, Gap-weighted)"]
    bars = ax_hit.bar(labels, vals)
    ax_hit.set_ylim(0, 1.05)
    ax_hit.set_ylabel("score")
    ax_hit.set_title("Aggregate w_j alignment")
    ax_hit.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax_hit.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", fontsize=10)

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    txt = (
        f"N tasks = {n}\n"
        f"Mean(top-1 q match)    = {summary['wj_top1_q']:.2f}\n"
        f"Mean(top-1 gap match)  = "
        f"{summary['wj_top1_gap_weighted']:.2f}\n"
        f"Mean(Kendall τ)        = "
        f"{summary['wj_kendall_tau_vs_gap']:.2f}\n\n"
        f"Reading:\n"
        f"  Top-1 q = 1 is built-in\n"
        f"  Top-1 gap < 1 → w_j alone\n"
        f"  doesn't capture what u\n"
        f"  actually LACKS; Gap term\n"
        f"  in S_cap is necessary."
    )
    ax_text.text(0.0, 1.0, txt, va="top", ha="left",
                 family="monospace", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2: attention heatmaps for 3 example tasks
# ---------------------------------------------------------------------------

def plot_attention_heatmaps(data: Dict, out_path: Path) -> None:
    per = data["per_task"]
    if len(per) < 3:
        picks = per
    else:
        picks = [per[0], per[len(per) // 2], per[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, task in zip(axes, picks):
        attn = task["attention_req_to_cap"]["optimum"]
        matrix = np.asarray(attn.get("alpha_matrix", []), dtype=float)
        cap_labels = attn.get("cap_labels", [])
        req_labels = [pr["req"] for pr in attn["per_req"]]
        if matrix.size == 0:
            ax.set_title(f"{task['task_id']} (no caps)")
            ax.axis("off")
            continue
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(cap_labels)))
        ax.set_xticklabels(cap_labels, rotation=60, fontsize=8, ha="right")
        ax.set_yticks(range(len(req_labels)))
        ax.set_yticklabels(req_labels, fontsize=8)
        ax.set_xlabel("v capability")
        ax.set_ylabel("task requirement")
        for i, pr in enumerate(attn["per_req"]):
            if pr["gt_cap"] is None:
                continue
            try:
                j = cap_labels.index(pr["gt_cap"])
            except ValueError:
                continue
            ax.scatter([j], [i], marker="*", s=120,
                       edgecolors="white", facecolors="red", linewidth=0.5)
        ax.set_title(
            f"{task['task_id']} — Top-1 hit "
            f"= {attn['mean_top1']:.2f}\n"
            f"mean entropy={attn['mean_entropy']:.2f}",
            fontsize=10,
        )
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(
        "Attention α(req → v's capability) — red ★ marks the ground-truth cap\n"
        "(optimum candidate per task)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3: θ scatter — final w_c vs complement ratio
# ---------------------------------------------------------------------------

def plot_theta_scatter(data: Dict, out_path: Path) -> None:
    per = data["per_task"]
    ratios = np.array([r["complement_info"]["ratio"] for r in per])
    w_cs = np.array([r["final_w"]["w_c"] for r in per])
    labels = [r["task_id"] for r in per]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    colors = ["#1f77b4" if r["complement_info"]["label"] == "complement_dominant"
              else "#ff7f0e" for r in per]
    ax.scatter(ratios, w_cs, c=colors, s=70, edgecolors="black", linewidth=0.5)
    for x, y, lab in zip(ratios, w_cs, labels):
        ax.annotate(lab, (x, y), fontsize=7, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points")
    if len(ratios) >= 2:
        m, b = np.polyfit(ratios, w_cs, 1)
        x_line = np.linspace(ratios.min(), ratios.max(), 20)
        ax.plot(x_line, m * x_line + b, "k--", alpha=0.5,
                label=f"linear fit (slope={m:.3f})")
        ax.legend(loc="best")
    ax.axhline(0.5, color="gray", lw=0.5)
    ax.axvline(0.5, color="gray", lw=0.5)
    ax.set_xlabel("Task complement ratio "
                  "(complement_strength / (complement + motivation))")
    ax.set_ylabel("Final w_c after SGD")
    ax.set_title(
        f"Final w_c vs task complement ratio\n"
        f"Spearman ρ = "
        f"{data['summary']['theta_spearman_wc_vs_complement_ratio']:.3f}, "
        f"Kendall τ = "
        f"{data['summary']['theta_kendall_wc_vs_complement_ratio']:.3f}",
        fontsize=10,
    )
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    for r in per:
        traj = [tw["w_c"] for tw in r["weights_trajectory"]]
        ax2.plot(range(len(traj)), traj, alpha=0.35, lw=1)
    ax2.set_xlabel("round")
    ax2.set_ylabel("w_c")
    ax2.set_title("w_c trajectories across all tasks")
    ax2.grid(alpha=0.3)

    blue_patch = mpatches.Patch(color="#1f77b4", label="complement-dominant task")
    orange_patch = mpatches.Patch(color="#ff7f0e", label="motivation-dominant task")
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0]
                   + [blue_patch, orange_patch], loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 4: μ convergence curves
# ---------------------------------------------------------------------------

def _mean_with_band(traj_per_task: List[List[float]]) -> tuple:
    """Pad/truncate each trace to same length and return (mean, lo, hi)."""
    if not traj_per_task:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(t) for t in traj_per_task)
    arr = np.array([t[:min_len] for t in traj_per_task], dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, mean - std, mean + std


def plot_mu_convergence(data: Dict, out_path: Path) -> None:
    per = data["per_task"]
    critical_trajs = [r["mu_sigma_convergence"]["critical_mu_abs_err"] for r in per]
    irrelevant_trajs = [r["mu_sigma_convergence"]["irrelevant_mu_abs_err"]
                        for r in per]

    crit_mean, crit_lo, crit_hi = _mean_with_band(critical_trajs)
    ir_mean, ir_lo, ir_hi = _mean_with_band(irrelevant_trajs)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(crit_mean))
    ax.plot(x, crit_mean, label="critical caps (mean over tasks)",
            color="#d62728", lw=2)
    ax.fill_between(x, crit_lo, crit_hi, color="#d62728", alpha=0.18)
    ax.plot(x, ir_mean, label="non-critical caps (mean over tasks)",
            color="#7f7f7f", lw=2, linestyle="--")
    ax.fill_between(x, ir_lo, ir_hi, color="#7f7f7f", alpha=0.15)
    ax.set_xlabel("round")
    ax.set_ylabel("mean |μ - μ_true|")
    ax.set_title(
        "Bayesian μ convergence — critical vs non-critical capabilities\n"
        f"frac. tasks where critical converges below non-critical = "
        f"{data['summary']['frac_tasks_critical_converges_faster']:.2f}",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 5: σ shrinkage curves
# ---------------------------------------------------------------------------

def plot_sigma_shrinkage(data: Dict, out_path: Path) -> None:
    per = data["per_task"]
    crit_trajs = [r["mu_sigma_convergence"]["critical_sigma"] for r in per]
    ir_trajs = [r["mu_sigma_convergence"]["irrelevant_sigma"] for r in per]

    crit_mean, crit_lo, crit_hi = _mean_with_band(crit_trajs)
    ir_mean, ir_lo, ir_hi = _mean_with_band(ir_trajs)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(crit_mean))
    ax.plot(x, crit_mean, label="critical caps (mean σ)",
            color="#1f77b4", lw=2)
    ax.fill_between(x, crit_lo, crit_hi, color="#1f77b4", alpha=0.18)
    ax.plot(x, ir_mean, label="non-critical caps (mean σ)",
            color="#9467bd", lw=2, linestyle="--")
    ax.fill_between(x, ir_lo, ir_hi, color="#9467bd", alpha=0.15)
    ax.set_xlabel("round")
    ax.set_ylabel("mean σ (posterior uncertainty)")
    ax.set_title(
        "Bayesian σ shrinkage — critical caps should shrink faster\n"
        "(non-critical never get updated, so σ is flat)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=None,
                        help="logs/interpretability_*.json path or suffix")
    parser.add_argument("--out-dir", default=str(PLOTS_DIR))
    args = parser.parse_args(argv)

    in_path = _find_latest(args.input)
    print(f"[plot] reading {in_path}")
    data = json.loads(in_path.read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_wj_vs_gap(data, out_dir / "01_wj_vs_gap.png")
    plot_attention_heatmaps(data, out_dir / "02_attention_heatmaps.png")
    plot_theta_scatter(data, out_dir / "03_theta_scatter.png")
    plot_mu_convergence(data, out_dir / "04_mu_convergence.png")
    plot_sigma_shrinkage(data, out_dir / "05_sigma_shrinkage.png")
    print(f"[plot] all figures under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
