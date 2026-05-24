import sys
import os
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mapping-algo"); sys.path.append(_p) if _p not in sys.path else None  # os.path.join(os.path.dirname(__file__), '..', 'mapping-algo'))

from datatypes import UserState, Task, MatchResult
from config import MatchConfig
from utils import cosine_sim
from WorldModel import WorldModel
from Reward_function import RewardSignals
from typing import Dict, List, Optional
import numpy as np


class BayesianUpdater:
    """
    Update capability parameters (μ, σ) using Gaussian conjugate posterior.

    For each task requirement, find the best-matching capability in the candidate
    via embedding cosine similarity, then apply:
      observation:  x_k = R · req.level
      obs noise:    σ_obs — source-aware (explicit < implicit < meta)
      posterior:    μ ← weighted average of prior μ and observation x
                    σ ← shrinks (more certain)
    """

    def __init__(self, config: MatchConfig = None):
        self.config = config if config is not None else MatchConfig()

    def update(
        self,
        candidate: UserState,
        task: Task,
        R: float,
        skill_observations: Optional[Dict[str, float]] = None,
    ) -> Dict[str, dict]:
        """
        Update candidate's capability parameters using reward R.

        For each task requirement, finds the highest-similarity capability
        in candidate (sim >= tau_update). Returns per-requirement update log.
        """
        if not candidate.capabilities:
            return {}

        cfg = self.config
        source_sigma = {
            "explicit": cfg.sigma_obs_explicit,
            "implicit": cfg.sigma_obs_implicit,
            "meta":     cfg.sigma_obs_meta,
        }

        v_embs = np.stack([c.embedding for c in candidate.capabilities])
        updates: Dict[str, dict] = {}

        for req in task.requirements:
            # Find best-matching capability by embedding similarity
            sims = np.array([cosine_sim(req.embedding, emb) for emb in v_embs])
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim < cfg.tau_update:
                continue

            cap = candidate.capabilities[best_idx]
            mu_old = cap.mu
            sigma_old = cap.sigma

            # Source-aware observation noise
            sigma_obs = source_sigma.get(cap.source, cfg.sigma_obs_implicit)

            # Observation: x_k = R * req.level, unless skill-level feedback
            # supplies a direct proficiency signal for this capability.
            cap_desc = cap.description or ""
            if skill_observations and cap_desc in skill_observations:
                x_k = float(skill_observations[cap_desc])
            else:
                x_k = R * req.level

            # Bayesian posterior update (precision form)
            sigma_obs_sq = sigma_obs ** 2
            sigma_v_sq = cap.sigma ** 2

            mu_new = (sigma_obs_sq * cap.mu + sigma_v_sq * x_k) / (sigma_v_sq + sigma_obs_sq)
            precision_new = 1.0 / sigma_v_sq + 1.0 / sigma_obs_sq
            sigma_new = 1.0 / np.sqrt(precision_new)

            cap.mu = round(float(np.clip(mu_new, 0.0, 1.0)), 4)
            cap.sigma = round(float(sigma_new), 4)

            key = req.description or f"req_{best_idx}"
            updates[key] = {
                "x_k": round(x_k, 4),
                "sigma_obs": round(sigma_obs, 4),
                "sim": round(best_sim, 4),
                "cap_source": cap.source,
                "mu": f"{mu_old:.4f} → {cap.mu:.4f}",
                "sigma": f"{sigma_old:.4f} → {cap.sigma:.4f}",
            }

        return updates


# ============================================================
# Layer 5: Parameter Update — Path 2: Weight SGD
# ============================================================

class WeightUpdater:
    """
    Update match score weights θ via SGD on MSE loss.

    Loss:     L = (M - R)²
    Gradient: ∂L/∂θ via chain rule through softmax
    Update:   θ ← θ - η·∂L/∂θ - λ·(θ - θ_prior)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        reg_lambda: float = 0.1,
        theta_prior: Optional[np.ndarray] = None,
    ):
        self.lr = learning_rate
        self.reg_lambda = reg_lambda
        # Default prior: slightly favor S_cap
        self.theta_prior = theta_prior if theta_prior is not None else np.array([0.4, -0.1])

    def update(
        self,
        world_model: WorldModel,
        match_result: MatchResult,
        R: float,
    ) -> dict:
        """
        One step of SGD on θ.

        Returns update details for logging.
        """
        M = match_result.match_score
        S_cap = match_result.s_cap
        S_need = match_result.s_need
        w = world_model.weights
        theta_old = world_model.theta.copy()

        # Loss
        loss = (M - R) ** 2

        # ∂L/∂M = 2(M - R)
        dL_dM = 2.0 * (M - R)

        # ∂M/∂w_c = S_cap, ∂M/∂w_n = S_need
        dM_dw = np.array([S_cap, S_need])

        # ∂w_i/∂θ_j = w_i(δ_ij - w_j)  (softmax Jacobian)
        # Full gradient via chain rule:
        # ∂L/∂θ_i = dL_dM · Σ_j [dM_dw_j · w_j · (δ_ji - w_i)]
        jacobian = np.diag(w) - np.outer(w, w)  # 2x2 softmax Jacobian
        dL_dtheta = dL_dM * (dM_dw @ jacobian)

        # Regularization: pull toward prior
        reg_term = self.reg_lambda * (world_model.theta - self.theta_prior)

        # SGD step
        world_model.theta -= self.lr * dL_dtheta + self.lr * reg_term

        w_new = world_model.weights

        return {
            "loss": round(loss, 6),
            "dL_dM": round(dL_dM, 4),
            "theta": f"[{theta_old[0]:.4f}, {theta_old[1]:.4f}] → [{world_model.theta[0]:.4f}, {world_model.theta[1]:.4f}]",
            "weights": f"w_c={w_new[0]:.4f}, w_n={w_new[1]:.4f}",
        }


# ============================================================
# Layer 5: UCB Exploration
# ============================================================

class UCBExplorer:
    """
    UCB exploration: μ̃ = μ + β_t · σ

    β_t = sqrt(2 · ln(t)), where t is the round number.
    As σ shrinks from Bayesian updates, UCB bonus automatically decreases.
    """

    def __init__(self):
        self.round = 0

    @property
    def beta(self) -> float:
        if self.round <= 1:
            return 1.0
        return np.sqrt(2.0 * np.log(self.round))

    def step(self):
        self.round += 1

    def compute_ucb_scores(self, candidate: UserState) -> Dict[str, dict]:
        """Compute UCB-enhanced capability estimates for all capabilities."""
        scores = {}
        for i, cap in enumerate(candidate.capabilities):
            bonus = self.beta * cap.sigma
            ucb_val = min(cap.mu + bonus, 1.0)
            key = cap.description or f"cap_{i}"
            scores[key] = {
                "mu": round(cap.mu, 4),
                "sigma": round(cap.sigma, 4),
                "beta_sigma": round(bonus, 4),
                "ucb": round(ucb_val, 4),
            }
        return scores
