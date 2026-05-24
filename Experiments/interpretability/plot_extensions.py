"""Plots for §7 extension experiments (08–11).

Reads logs tagged ext-* and writes:
  08_skill_feedback_comparison.png
  09_theta_contrast_scatter.png
  10_multi_seed_ci.png
  11_team_shapley.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
LOGS = _REPO / "logs"
PLOTS = _HERE / "plots"


def _find(tag: str) -> Optional[Path]:
    cands = sorted(LOGS.glob(f"interpretability_*{tag}*.json"))
    return cands[-1] if cands else None


def _load(p: Optional[Path]) -> Optional[Dict]:
    return json.loads(p.read_text()) if p and p.is_file() else None


def plot_skill_feedback(sim: Dict, skill: Dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["sim", "sim+skill"]
    vals = [
        sim["summary"]["mean_final_critical_err"],
        skill["summary"]["mean_final_critical_err"],
    ]
    ax.bar(labels, vals, color=["#e07a5f", "#3d405b"])
    ax.set_ylabel("mean final critical μ-error")
    ax.set_title("Skill-level feedback closes sim → gt gap?")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_theta_contrast(data: Dict, out: Path) -> None:
    per = data["per_task"]
    wc = [t["final_w"]["w_c"] for t in per]
    ratio = [t["complement_info"]["ratio"] for t in per]
    colors = ["#457b9d" if t["task_id"].startswith("theta_comp") else "#e9c46a"
              for t in per]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ratio, wc, c=colors, s=60, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("complement / (complement + motivation)")
    ax.set_ylabel("final w_c")
    ax.set_title("θ contrast testset — w_c vs complement ratio")
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_multi_seed(data: Dict, out: Path) -> None:
    per_seed = data.get("per_seed", {})
    if not per_seed:
        return
    key = "mean_final_critical_err"
    vals = [per_seed[s]["summary"][key] for s in sorted(per_seed, key=int)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        range(len(vals)), [np.mean(vals)] * len(vals),
        yerr=[[np.mean(vals) - np.min(vals)], [np.max(vals) - np.mean(vals)]],
        fmt="o", capsize=8, color="#2a9d8f",
    )
    ax.scatter(range(len(vals)), vals, color="#264653", zorder=3)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(sorted(per_seed, key=int))
    ax.set_xlabel("seed")
    ax.set_ylabel(key)
    ax.set_title("Cross-seed variance (critical μ-error)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_team_shapley(data: Dict, out: Path) -> None:
    per = data["per_task"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cov = [t["collective_coverage"] for t in per]
    axes[0].hist(cov, bins=8, color="#81b29a", edgecolor="white")
    axes[0].set_xlabel("collective coverage R(S)")
    axes[0].set_title("1-N greedy teams")

    sel = [t["mean_shapley_selected"] for t in per]
    unsel = [t["mean_shapley_unselected"] for t in per]
    x = np.arange(len(per))
    w = 0.35
    axes[1].bar(x - w / 2, sel, w, label="selected", color="#457b9d")
    axes[1].bar(x + w / 2, unsel, w, label="unselected", color="#f4a261")
    axes[1].set_xlabel("task index")
    axes[1].set_ylabel("mean Shapley φ")
    axes[1].legend()
    axes[1].set_title("Selected vs unselected Shapley")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PLOTS))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = _load(_find("ext-sim-baseline"))
    skill = _load(_find("ext-skill-feedback"))
    if sim and skill:
        plot_skill_feedback(sim, skill, out_dir / "08_skill_feedback_comparison.png")

    theta = _load(_find("ext-theta-contrast"))
    if theta:
        plot_theta_contrast(theta, out_dir / "09_theta_contrast_scatter.png")

    ms = _load(_find("ext-multi-seed"))
    if ms:
        plot_multi_seed(ms, out_dir / "10_multi_seed_ci.png")

    team_path = LOGS / "team_interpretability.json"
    if team_path.is_file():
        plot_team_shapley(json.loads(team_path.read_text()),
                          out_dir / "11_team_shapley.png")

    print(f"Extension plots → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
