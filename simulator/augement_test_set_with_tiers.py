"""
augment_testset_with_tiers.py
==============================

Augments a 20-tasks × 20-candidates JSON test set with tier-based
capability priors (μ_init, σ_init) for testing σ-modulated UCB
exploration vs greedy and vanilla UCB1.

Each candidate is assigned to one of 4 experience tiers, distributed
evenly within each task (5 candidates per tier). The σ_init and
μ_init noise are coupled — a veteran has both low σ and accurate μ;
a newcomer has both high σ and noisy μ.

Tier specs (defaults):
  | tier         | σ_base | μ_noise_std | meaning                       |
  |--------------|--------|-------------|-------------------------------|
  | veteran      | 0.05   | 0.02        | platform veterans             |
  | mid          | 0.15   | 0.10        | 2-3 past collaborations       |
  | new_explicit | 0.30   | 0.20        | new, but detailed self-report |
  | new_implicit | 0.50   | 0.30        | new, only inferred from chat  |

Usage:
    python augment_testset_with_tiers.py \
        --input  simulator/20_Tasks_Testset.json \
        --output simulator/20_Tasks_Testset_tiered.json \
        --seed 42
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# Tier specification
# ============================================================

@dataclass
class TierSpec:
    name: str
    sigma_base: float        # central σ value
    sigma_jitter: float      # ± range to avoid identical σ across same-tier candidates
    mu_noise_std: float      # std of N(0, σ) noise added to μ_true → μ_init

    def sample_sigma(self, rng: np.random.Generator) -> float:
        """One σ value drawn from [sigma_base - jitter, sigma_base + jitter]."""
        return float(np.clip(
            rng.uniform(self.sigma_base - self.sigma_jitter,
                        self.sigma_base + self.sigma_jitter),
            0.01, 0.99,
        ))

    def sample_mu_init(self, mu_true: float, rng: np.random.Generator) -> float:
        """μ_init = clip(μ_true + N(0, mu_noise_std), 0, 1)."""
        noise = rng.normal(0.0, self.mu_noise_std)
        return float(np.clip(mu_true + noise, 0.0, 1.0))


DEFAULT_TIERS = [
    TierSpec(name="veteran",      sigma_base=0.05, sigma_jitter=0.02, mu_noise_std=0.02),
    TierSpec(name="mid",          sigma_base=0.15, sigma_jitter=0.04, mu_noise_std=0.10),
    TierSpec(name="new_explicit", sigma_base=0.30, sigma_jitter=0.06, mu_noise_std=0.20),
    TierSpec(name="new_implicit", sigma_base=0.50, sigma_jitter=0.08, mu_noise_std=0.30),
]


# ============================================================
# Per-task augmentation
# ============================================================

def assign_tiers_to_candidates(
    n_candidates: int,
    tiers: list[TierSpec],
    rng: np.random.Generator,
) -> list[TierSpec]:
    """
    Distribute `n_candidates` slots across `tiers` evenly, then shuffle.
    Returns a list of length n_candidates with tier assignments.
    """
    per_tier = n_candidates // len(tiers)
    remainder = n_candidates - per_tier * len(tiers)

    assignments = []
    for i, tier in enumerate(tiers):
        count = per_tier + (1 if i < remainder else 0)
        assignments.extend([tier] * count)

    # Shuffle so tier assignment doesn't correlate with candidate index
    rng.shuffle(assignments)
    return assignments


def augment_candidate(
    candidate: dict,
    tier: TierSpec,
    rng: np.random.Generator,
) -> dict:
    """
    Add `capability_priors`, `tier`, `tier_meta` to a single candidate.
    Does not modify `capabilities` (kept as ground truth).
    """
    capabilities = candidate["candidate_profile"]["capabilities"]

    # One σ_init per candidate (could be per-skill, but per-candidate is cleaner)
    sigma_init = tier.sample_sigma(rng)

    priors = {}
    for skill, mu_true in capabilities.items():
        priors[skill] = {
            "mu_init":    round(tier.sample_mu_init(mu_true, rng), 4),
            "sigma_init": round(sigma_init, 4),
        }

    candidate["capability_priors"] = priors
    candidate["tier"] = tier.name
    candidate["tier_meta"] = {
        "sigma_base":   tier.sigma_base,
        "mu_noise_std": tier.mu_noise_std,
    }
    return candidate


def augment_task(task_entry: dict, tiers: list[TierSpec], rng: np.random.Generator) -> dict:
    """Augment all candidates in one task entry."""
    candidates = task_entry["candidates"]
    tier_assignments = assign_tiers_to_candidates(len(candidates), tiers, rng)

    for cand, tier in zip(candidates, tier_assignments):
        augment_candidate(cand, tier, rng)

    return task_entry


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON test set path")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    data = json.loads(Path(args.input).read_text())
    tasks = data["tasks"]

    for task_entry in tasks:
        augment_task(task_entry, DEFAULT_TIERS, rng)

    Path(args.output).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # Print summary
    print(f"Augmented {len(tasks)} tasks × {len(tasks[0]['candidates'])} candidates")
    print()
    print(f"{'tier':<14} {'σ_base':>8} {'μ_noise':>8} {'count':>8}")
    print("-" * 44)
    tier_counts = {t.name: 0 for t in DEFAULT_TIERS}
    for task_entry in tasks:
        for cand in task_entry["candidates"]:
            tier_counts[cand["tier"]] += 1
    for tier in DEFAULT_TIERS:
        print(f"{tier.name:<14} {tier.sigma_base:>8.2f} {tier.mu_noise_std:>8.2f} "
              f"{tier_counts[tier.name]:>8d}")
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()