"""
Plot Layer 5 model performance from wandb exports.
Reads both .csv and .numbers files.

Outputs: model_performance.png (2×2 grid)
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numbers_parser import Document
from pathlib import Path

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wandb", "run-20260418-171149")


# ============================================================
# File readers
# ============================================================

def read_csv(path):
    """Read wandb CSV export → (steps[], values[])."""
    steps, vals = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            steps.append(int(row[0]))
            vals.append(float(row[1]))
    return np.array(steps), np.array(vals)

def load(name):
    """Try .csv first, then .numbers."""
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    if os.path.exists(csv_path):
        return read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No file found for {name}")


# ============================================================
# Load all data
# ============================================================

mu_bay_steps,   mu_bay   = load("wandb_mu_bayesian")
mu_py_steps,    mu_py    = load("wandb_mu_python")
mu_pw_steps,    mu_pw    = load("wandb_mu_paper_writing")

sig_bay_steps,  sig_bay  = load("wandb_sigma_bayesian")
sig_py_steps,   sig_py   = load("wandb_sigma_python")
sig_pw_steps,   sig_pw   = load("wandb_sigma_paper_writing")

match_steps,    match_m  = load("wandb_match_M")
reward_steps,   reward_r = load("wandb_reward")

# ============================================================
# Plot configuration
# ============================================================

# Colors
PURPLE   = "#6C5CE7"
TEAL     = "#00B894"
CORAL    = "#E17055"
AMBER    = "#BA7517"
BLUE     = "#0984E3"
BG_LIGHT = "#FAFAF8"
GRID_CLR = "#E5E4DF"
TEXT_CLR = "#3D3D3A"
TEXT2    = "#8B8B82"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DM Sans", "Helvetica Neue", "Arial"],
    "font.size": 11,
    "axes.facecolor": BG_LIGHT,
    "figure.facecolor": "#FFFFFF",
    "axes.edgecolor": GRID_CLR,
    "axes.labelcolor": TEXT_CLR,
    "xtick.color": TEXT2,
    "ytick.color": TEXT2,
    "grid.color": GRID_CLR,
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.subplots_adjust(hspace=0.35, wspace=0.28, top=0.92, bottom=0.08, left=0.08, right=0.95)
fig.suptitle("MBRL matching system — Layer 5 convergence", fontsize=16, fontweight=600, color=TEXT_CLR, y=0.97)


# ============================================================
# Chart 1: μ convergence (line chart)
# ============================================================

ax1 = axes[0, 0]
ax1.plot(mu_bay_steps, mu_bay, color=PURPLE, linewidth=2, label="Bayesian", zorder=3)
ax1.plot(mu_py_steps,  mu_py,  color=TEAL,   linewidth=2, label="Python",   zorder=3)
ax1.plot(mu_pw_steps,  mu_pw,  color=CORAL,  linewidth=2, label="Paper writing", zorder=3)

# Start/end annotations
for steps, vals, color, name in [
    (mu_bay_steps, mu_bay, PURPLE, "Bayesian"),
    (mu_pw_steps,  mu_pw,  CORAL,  "Paper writing"),
]:
    ax1.annotate(f"{vals[0]:.3f}", xy=(steps[0], vals[0]), fontsize=8, color=color,
                 fontweight=500, ha="left", va="bottom", xytext=(4, 4), textcoords="offset points")
    ax1.annotate(f"{vals[-1]:.3f}", xy=(steps[-1], vals[-1]), fontsize=8, color=color,
                 fontweight=500, ha="right", va="top", xytext=(-4, -6), textcoords="offset points")

ax1.set_title("Capability μ estimates", fontsize=13, fontweight=500, color=TEXT_CLR, pad=10)
ax1.set_xlabel("Round")
ax1.set_ylabel("μ")
ax1.legend(fontsize=9, framealpha=0.9, edgecolor=GRID_CLR)
ax1.set_ylim(0.4, 1.0)


# ============================================================
# Chart 2: σ convergence (line chart)
# ============================================================

ax2 = axes[0, 1]
ax2.plot(sig_bay_steps, sig_bay, color=PURPLE, linewidth=2, linestyle="--", label="Bayesian", zorder=3)
ax2.plot(sig_py_steps,  sig_py,  color=TEAL,   linewidth=2, linestyle="--", label="Python",   zorder=3)
ax2.plot(sig_pw_steps,  sig_pw,  color=CORAL,  linewidth=2, linestyle="--", label="Paper writing", zorder=3)

# Annotate reduction %
for steps, vals, color, name in [
    (sig_bay_steps, sig_bay, PURPLE, "Bayesian"),
    (sig_py_steps,  sig_py,  TEAL,   "Python"),
    (sig_pw_steps,  sig_pw,  CORAL,  "Paper writing"),
]:
    reduction = (1 - vals[-1] / vals[0]) * 100
    ax2.annotate(f"-{reduction:.0f}%", xy=(steps[-1], vals[-1]), fontsize=8, color=color,
                 fontweight=600, ha="left", va="center", xytext=(6, 0), textcoords="offset points")

ax2.set_title("Uncertainty σ convergence", fontsize=13, fontweight=500, color=TEXT_CLR, pad=10)
ax2.set_xlabel("Round")
ax2.set_ylabel("σ")
ax2.legend(fontsize=9, framealpha=0.9, edgecolor=GRID_CLR)


# ============================================================
# Chart 3: Match score M (scatter)
# ============================================================

ax3 = axes[1, 0]
ax3.scatter(match_steps, match_m, color=BLUE, s=28, alpha=0.65, edgecolors="white", linewidth=0.5, zorder=3)

ax3.set_title("Match score M per round", fontsize=13, fontweight=500, color=TEXT_CLR, pad=10)
ax3.set_xlabel("Round")
ax3.set_ylabel("M")

# Mean line
mean_m = match_m.mean()
ax3.axhline(mean_m, color=BLUE, linewidth=0.8, linestyle=":", alpha=0.5)
ax3.annotate(f"mean = {mean_m:.4f}", xy=(match_steps[-1], mean_m), fontsize=8, color=BLUE,
             ha="right", va="bottom", xytext=(0, 4), textcoords="offset points")


# ============================================================
# Chart 4: Reward R (scatter)
# ============================================================

ax4 = axes[1, 1]

# Color-code by reward level
colors = [TEAL if r >= 0.85 else (AMBER if r >= 0.7 else CORAL) for r in reward_r]
ax4.scatter(reward_steps, reward_r, c=colors, s=28, alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)

# Mean line
mean_r = reward_r.mean()
ax4.axhline(mean_r, color=TEXT2, linewidth=0.8, linestyle=":", alpha=0.5)
ax4.annotate(f"mean = {mean_r:.4f}", xy=(reward_steps[-1], mean_r), fontsize=8, color=TEXT2,
             ha="right", va="bottom", xytext=(0, 4), textcoords="offset points")

ax4.set_title("Reward R per round", fontsize=13, fontweight=500, color=TEXT_CLR, pad=10)
ax4.set_xlabel("Round")
ax4.set_ylabel("R")
ax4.set_ylim(0.55, 1.0)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAL,  markersize=7, label="R ≥ 0.85"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=AMBER, markersize=7, label="0.70 ≤ R < 0.85"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CORAL, markersize=7, label="R < 0.70"),
]
ax4.legend(handles=legend_elements, fontsize=8, framealpha=0.9, edgecolor=GRID_CLR, loc="lower right")


# ============================================================
# Save
# ============================================================

out_path = os.path.join(DATA_DIR, "model_performance.svg")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
print(f"Saved to {out_path}")
plt.close()