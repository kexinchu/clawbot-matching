"""Plot OnlineLearning JSON traces as a 2x2 figure.

Reads the four JSON files written by:
    Online_learning/tests/run_one_round_and_log.py

By default, the script finds the latest timestamped trace set under `logs/`
and saves a PNG in `plots/`.

Usage:
    python plots/plot_online_learning_traces.py
    python plots/plot_online_learning_traces.py 20260511-165346
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"
PLOTS_DIR = REPO_ROOT / "plots"

FILE_PREFIXES = {
    "means": "online_learning_candidate_means",
    "sigmas": "online_learning_candidate_sigmas",
    "match_scores": "online_learning_match_scores",
    "rewards": "online_learning_reward_scores",
}

COLORS = [
    "#6C5CE7",
    "#00B894",
    "#E17055",
    "#0984E3",
    "#D63031",
    "#FDCB6E",
    "#00CEC9",
    "#A29BFE",
]


def _extract_timestamp(path: Path) -> str | None:
    match = re.search(r"(\d{8}-\d{6})", path.name)
    return match.group(1) if match else None


def _latest_timestamp() -> str:
    matches = sorted(LOGS_DIR.glob(f"{FILE_PREFIXES['means']}_*.json"))
    if not matches:
        raise FileNotFoundError("No OnlineLearning trace JSON files found in logs/")
    timestamps = [_extract_timestamp(path) for path in matches]
    timestamps = [ts for ts in timestamps if ts is not None]
    if not timestamps:
        raise FileNotFoundError("Could not determine timestamp from trace filenames.")
    return sorted(timestamps)[-1]


def _trace_path(prefix: str, timestamp: str) -> Path:
    return LOGS_DIR / f"{prefix}_{timestamp}.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_nested_series(ax, trace: dict, series_key: str, title: str, ylabel: str) -> None:
    rounds = [entry["round"] for entry in trace["rounds"]]
    labels = []
    values = {}

    for entry in trace["rounds"]:
        for candidate_id, metric_map in entry[series_key].items():
            for metric_name, metric_value in metric_map.items():
                label = f"{candidate_id}:{metric_name}"
                if label not in values:
                    values[label] = []
                    labels.append(label)
                values[label].append(metric_value)

    for idx, label in enumerate(labels):
        ax.plot(
            rounds,
            values[label],
            label=label,
            linewidth=1.8,
            color=COLORS[idx % len(COLORS)],
        )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def _plot_match_scores(ax, trace: dict) -> None:
    rounds = [entry["round"] for entry in trace["rounds"]]
    m_no_ucb = [entry["match_score"]["M (no UCB)"] for entry in trace["rounds"]]
    m_ucb = [entry["match_score"]["M (with UCB)"] for entry in trace["rounds"]]
    s_cap = [entry["match_score"]["S_cap"] for entry in trace["rounds"]]
    s_need = [entry["match_score"]["S_need"] for entry in trace["rounds"]]

    ax.plot(rounds, m_no_ucb, label="M (no UCB)", linewidth=2.0, color="#0984E3")
    ax.plot(rounds, m_ucb, label="M (with UCB)", linewidth=2.0, color="#6C5CE7")
    ax.plot(rounds, s_cap, label="S_cap", linewidth=1.6, linestyle="--", color="#00B894")
    ax.plot(rounds, s_need, label="S_need", linewidth=1.6, linestyle="--", color="#E17055")
    ax.set_title("Match Scores")
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def _plot_rewards(ax, trace: dict) -> None:
    rounds = [entry["round"] for entry in trace["rounds"]]
    total_r = [entry["reward"]["R"] for entry in trace["rounds"]]
    r_feedback = [entry["reward"]["r_feedback"] for entry in trace["rounds"]]
    r_efficiency = [entry["reward"]["r_efficiency"] for entry in trace["rounds"]]
    r_quality = [entry["reward"]["r_quality"] for entry in trace["rounds"]]

    ax.plot(rounds, total_r, label="R", linewidth=2.2, color="#D63031")
    ax.plot(rounds, r_feedback, label="r_feedback", linewidth=1.4, linestyle="--", color="#0984E3")
    ax.plot(rounds, r_efficiency, label="r_efficiency", linewidth=1.4, linestyle="--", color="#00B894")
    ax.plot(rounds, r_quality, label="r_quality", linewidth=1.4, linestyle="--", color="#E17055")
    ax.set_title("Reward Scores")
    ax.set_xlabel("Round")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def main() -> int:
    timestamp = sys.argv[1] if len(sys.argv) > 1 else _latest_timestamp()

    means = _load_json(_trace_path(FILE_PREFIXES["means"], timestamp))
    sigmas = _load_json(_trace_path(FILE_PREFIXES["sigmas"], timestamp))
    match_scores = _load_json(_trace_path(FILE_PREFIXES["match_scores"], timestamp))
    rewards = _load_json(_trace_path(FILE_PREFIXES["rewards"], timestamp))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"OnlineLearning Trace Summary ({timestamp})", fontsize=15)

    _plot_nested_series(axes[0, 0], means, "candidate_means", "Candidate Means", "Mean")
    _plot_nested_series(axes[0, 1], sigmas, "candidate_sigmas", "Candidate Standard Deviations", "Sigma")
    _plot_match_scores(axes[1, 0], match_scores)
    _plot_rewards(axes[1, 1], rewards)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / f"online_learning_traces_{timestamp}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
