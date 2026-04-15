"""Global configuration — corresponds to §4A.2 of clawbot-nips.md"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchConfig:
    """All tunable parameters.

    Core model uses 2 weights (w_c, w_n), w_c + w_n = 1.
    """
    # Embedding
    embedding_dim: int = 768

    # Attention soft-matching
    temperature: float = 0.1        # τ: softmax temperature

    # Gate
    tau_hard: float = 0.7           # hard constraint similarity threshold

    # S_need
    alpha_infer: float = 0.5        # inferred offer discount

    # Numerical stability
    epsilon: float = 1e-8

    # UCB exploration
    ucb_beta_scale: float = 2.0     # β_t = sqrt(scale × log(t))

    # Team building
    n_max: int = 5
    gap_epsilon: float = 1e-4

    # FAISS
    faiss_pre_k: int = 100
    faiss_min_pool: int = 200

    # Scene weights θ ∈ R² (before softmax)
    theta_default: np.ndarray = field(
        default_factory=lambda: np.array([0.4, -0.4])
    )  # → w ≈ (0.69, 0.31)

    theta_balanced: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )  # → w = (0.5, 0.5)

    theta_need_heavy: np.ndarray = field(
        default_factory=lambda: np.array([-0.4, 0.4])
    )  # → w ≈ (0.31, 0.69)

    # Bayesian update
    sigma_obs_explicit: float = 0.2
    sigma_obs_implicit: float = 0.4
    sigma_obs_meta: float = 0.6
    sigma_init: float = 1.0

    # Capability management
    tau_merge: float = 0.85
    tau_new: float = 0.5
    tau_update: float = 0.5
