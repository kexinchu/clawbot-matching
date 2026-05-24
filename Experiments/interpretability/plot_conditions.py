"""Cross-condition plots for the interpretability experiment.

Reads three logs/interpretability_*.json files (gt / sim / sim_dream by
default) and renders:

    06_condition_comparison.png — bar chart of the 5 key alignment
        metrics across feedback / dreaming conditions
    07_dreaming_vs_analytical.png — per-task L3.1 vs L3.2 ranking
        comparison + recommendation distribution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
LOGS_DIR = _REPO_ROOT / "logs"
PLOTS_DIR = _HERE / "plots"


def _latest(tag: str) -> Optional[Path]:
    cands = sorted(LOGS_DIR.glob(f"interpretability_{tag}_*.json"))
    return cands[-1] if cands else None


def _load_or_die(path_or_tag: str) -> Dict:
    p = Path(path_or_tag)
    if p.exists():
        return json.loads(p.read_text())
    found = _latest(path_or_tag)
    if found is None:
        raise FileNotFoundError(path_or_tag)
    return json.loads(found.read_text())


# ---------------------------------------------------------------------------
# Figure 6 — bar chart of key metrics across conditions
# ---------------------------------------------------------------------------

# (key, label, ylim_hint) — keep order matching the plotting loop
METRICS = [
    ("wj_top1_gap_weighted",
     "w_j Top-1 vs Gap-weighted",
     (0, 1)),
    ("attn_req_to_cap_top1_optimum",
     "Attn req→cap Top-1\n(optimum candidate)",
     (0, 1)),
    ("frac_tasks_critical_converges_faster",
     "Frac tasks: critical μ\nconverges below irrelevant",
     (0, 1)),
    ("mean_final_critical_err",
     "Mean final |μ - μ_true|\n(critical caps)",
     None),
    ("mean_final_irrelevant_err",
     "Mean final |μ - μ_true|\n(irrelevant caps)",
     None),
    ("theta_spearman_wc_vs_complement_ratio",
     "θ Spearman ρ\n(w_c vs complement ratio)",
     (-1, 1)),
]


def plot_condition_comparison(
    data_by_label: Dict[str, Dict],
    out_path: Path,
) -> None:
    labels = list(data_by_label.keys())
    n_cond = len(labels)
    n_metrics = len(METRICS)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    width = 0.7

    for i, (key, title, ylim) in enumerate(METRICS):
        ax = axes[i]
        vals = []
        for lab in labels:
            v = data_by_label[lab]["summary"].get(key, float("nan"))
            vals.append(v)
        xs = np.arange(n_cond)
        bars = ax.bar(xs, vals, width, color=palette[:n_cond],
                      edgecolor="black", linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            label_txt = "n/a" if (val is None or (isinstance(val, float)
                                                  and np.isnan(val))) else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    label_txt, ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Cross-condition alignment metrics — "
        "feedback source × dreaming toggle",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 7 — dreaming vs analytical ranking comparison
# ---------------------------------------------------------------------------

def plot_dreaming_vs_analytical(
    data_dream: Dict,
    out_path: Path,
) -> None:
    per = data_dream["per_task"]
    overlaps = [r["dreaming"]["mean_layer3_top_k_overlap"]
                for r in per
                if not np.isnan(r["dreaming"]["mean_layer3_top_k_overlap"])]
    top1s = [r["dreaming"]["mean_layer3_top1_agree"]
             for r in per
             if not np.isnan(r["dreaming"]["mean_layer3_top1_agree"])]
    kts = [r["dreaming"]["mean_layer3_kendall_tau"]
           for r in per
           if not np.isnan(r["dreaming"]["mean_layer3_kendall_tau"])]
    task_labels = [r["task_id"] for r in per
                   if not np.isnan(r["dreaming"]["mean_layer3_top_k_overlap"])]

    rec_counts: Dict[str, int] = (
        data_dream["summary"].get("dreaming_recommendation_counts", {}) or {}
    )

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(2, 3)

    # Per-task overlap bars
    ax1 = fig.add_subplot(gs[0, :2])
    xs = np.arange(len(task_labels))
    width = 0.27
    ax1.bar(xs - width, overlaps, width, label="Top-3 overlap")
    ax1.bar(xs, top1s, width, label="Top-1 agree")
    ax1.bar(xs + width, kts, width, label="Kendall τ (common items)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(task_labels, rotation=60, fontsize=8)
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_ylabel("agreement (1 = identical)")
    ax1.set_title(
        "Per-task: how much does Dreaming reorder the Top-K?\n"
        "(higher = analytical and dream rankings agree)"
    )
    ax1.legend(loc="lower center", ncol=3, fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    # Recommendation pie
    ax2 = fig.add_subplot(gs[0, 2])
    if rec_counts:
        labs = list(rec_counts.keys())
        sizes = [rec_counts[k] for k in labs]
        ax2.pie(sizes, labels=labs, autopct="%1.0f%%", startangle=90)
        ax2.set_title(
            "Dreaming recommendation\ndistribution (all rounds)",
            fontsize=10,
        )
    else:
        ax2.text(0.5, 0.5, "no recommendations",
                 ha="center", va="center")
        ax2.axis("off")

    # Summary numbers
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    summary = data_dream["summary"]
    meta = data_dream.get("metadata", {})
    backend = meta.get("dreaming_backend", "unknown")
    is_real = "real" in str(backend)
    if is_real:
        reading = (
            f"  · Top-3 overlap = 1.0 by construction here "
            f"(pool_size = top_k = {meta.get('pool_size', '?')}, so all "
            f"candidates are always in top-3); the informative signal is "
            f"Top-1 agree and Kendall τ.\n"
            f"  · Top-1 agree {summary['dreaming_mean_layer3_top1_agree']:.2f}"
            f" means the LLM dream pass swaps L3.1's #1 pick in "
            f"{(1 - summary['dreaming_mean_layer3_top1_agree']) * 100:.0f}% of"
            f" rounds.\n"
            f"  · The recommendation distribution stops being a constant "
            f"`good_match`, exposing real LLM disagreement across "
            f"compatibility dimensions (time / priority / style / "
            f"personality).\n"
            f"  · Conclusion: Dreaming carries interpretable signal "
            f"orthogonal to MapScore; analytical weights point at 'who "
            f"covers the skill gap', dreaming points at 'who is realistic"
            f" to collaborate with'."
        )
    else:
        reading = (
            f"  · Overlap close to 1.0 means the dream pass barely changes"
            f" the analytical ranking.\n"
            f"  · With the mock dreaming backend (no API_KEY) judge scores"
            f" are mostly constant, so combined_score ≈ analytical and "
            f"rankings stay put.\n"
            f"  · A real LLM judge with diverse persona priors would lower"
            f" these numbers, exposing where Dreaming reorders L3.1's "
            f"choices."
        )
    txt = (
        f"Backend: {backend}\n"
        f"Mean Top-3 overlap across {len(overlaps)} tasks "
        f"= {summary['dreaming_mean_layer3_top_k_overlap']:.3f}\n"
        f"Mean Top-1 agree                          "
        f"= {summary['dreaming_mean_layer3_top1_agree']:.3f}\n"
        f"Mean Kendall τ on common items           "
        f"= {summary['dreaming_mean_layer3_kendall_tau']:.3f}\n\n"
        f"Reading:\n{reading}"
    )
    ax3.text(0.0, 0.95, txt, va="top", ha="left",
             family="monospace", fontsize=10)

    fig.suptitle(
        "Layer 3.1 (analytical) vs Layer 3.2 (dreaming) ranking — "
        "interpretability via reordering",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="fb-gt",
                        help="Tag or path for ground-truth-feedback run.")
    parser.add_argument("--sim", default="fb-sim",
                        help="Tag or path for simulator-feedback run.")
    parser.add_argument("--sim-dream", default="fb-sim_dream",
                        help="Tag or path for simulator + mock-dreaming run.")
    parser.add_argument("--sim-dream-real", default=None,
                        help="Optional tag or path for simulator + real-LLM "
                             "dreaming run. When provided, used for "
                             "plot 07; also added as a 4th bar in plot 06.")
    parser.add_argument("--out-dir", default=str(PLOTS_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_gt = _load_or_die(args.gt)
    data_sim = _load_or_die(args.sim)
    data_dream = _load_or_die(args.sim_dream)
    data_dream_real = _load_or_die(args.sim_dream_real) if args.sim_dream_real else None

    data_by_label: Dict[str, Dict] = {
        "ground-truth\n(M against true μ)": data_gt,
        "simulator\n(bilateral mock)": data_sim,
        "simulator\n+ dreaming (mock)": data_dream,
    }
    if data_dream_real is not None:
        data_by_label["simulator\n+ dreaming (real LLM)"] = data_dream_real

    plot_condition_comparison(
        data_by_label, out_dir / "06_condition_comparison.png",
    )
    # Plot 07 prefers the real-LLM run if available, otherwise mock.
    plot_dreaming_vs_analytical(
        data_dream_real if data_dream_real is not None else data_dream,
        out_dir / "07_dreaming_vs_analytical.png",
    )
    print(f"[plot] all condition plots under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
