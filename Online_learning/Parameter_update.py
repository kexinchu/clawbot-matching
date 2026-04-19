from UserProfile import UserProfile
from Task import Task
from typing import Dict, Optional
import numpy as np
from WorldModel import WorldModel
from Reward_function import RewardSignals
from MatchResult import MatchResult

class BayesianUpdater:
    """
    Update capability parameters (μ, σ) using Gaussian conjugate posterior.
 
    For each dimension k:
      observation:  x^k = R · Q_T^k
      obs noise:    σ_obs^k = σ_base / (Q_T^k + ε)
      posterior:    μ ← weighted average of prior μ and observation x
                    σ ← shrinks (more certain)
    """
 
    def __init__(self, sigma_base: float = 0.3, epsilon: float = 0.01):
        self.sigma_base = sigma_base
        self.epsilon = epsilon
 
    def update(
        self,
        candidate: UserProfile,
        task: Task,
        R: float,
    ) -> Dict[str, dict]:
        """
        Update candidate's capability parameters using reward R.
 
        Returns dict of per-dimension update details for logging.
        """
        updates = {}
        for domain, Q_k in task.Q_T.items():
            if domain not in candidate.capability:
                continue
 
            cap = candidate.capability[domain]
            mu_old = cap.mu
            sigma_old = cap.sigma
 
            # Step 1: Construct observation
            x_k = R * Q_k
 
            # Step 2: Observation noise (inversely proportional to task demand)
            sigma_obs = self.sigma_base / (Q_k + self.epsilon)
 
            # Step 3: Bayesian posterior update
            sigma_obs_sq = sigma_obs ** 2
            sigma_v_sq = cap.sigma ** 2
 
            # Update μ: weighted average of prior and observation
            mu_new = (sigma_obs_sq * cap.mu + sigma_v_sq * x_k) / (sigma_v_sq + sigma_obs_sq)
 
            # Update σ: precision adds up
            precision_new = 1.0 / sigma_v_sq + 1.0 / sigma_obs_sq
            sigma_new = 1.0 / np.sqrt(precision_new)
 
            # Apply update
            cap.mu = round(np.clip(mu_new, 0.0, 1.0), 4)
            cap.sigma = round(sigma_new, 4)
 
            updates[domain] = {
                "x_k": round(x_k, 4),
                "sigma_obs": round(sigma_obs, 4),
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
        M = match_result.M
        S_cap = match_result.S_cap
        S_need = match_result.S_need
        w = world_model.weights
        theta_old = world_model.theta.copy()
 
        # Loss
        loss = (M - R) ** 2
 
        # ∂L/∂M = 2(M - R)
        dL_dM = 2.0 * (M - R)
 
        # ∂M/∂w_c = S_cap, ∂M/∂w_n = S_need
        dM_dw = np.array([S_cap, S_need])
 
        # ∂w_i/∂θ_j = w_i(δ_ij - w_j)  (softmax Jacobian)
        # For diagonal: ∂w_i/∂θ_i = w_i(1 - w_i)
        # Full gradient via chain rule:
        # ∂L/∂θ_i = dL_dM · Σ_j [dM_dw_j · ∂w_j/∂θ_i]
        # = dL_dM · Σ_j [dM_dw_j · w_j · (δ_ji - w_i)]
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
 
    def compute_ucb_scores(self, candidate: UserProfile) -> Dict[str, dict]:
        """Compute UCB-enhanced capability estimates for a candidate."""
        scores = {}
        for domain, cap in candidate.capability.items():
            bonus = self.beta * cap.sigma
            ucb_val = min(cap.mu + bonus, 1.0)
            scores[domain] = {
                "mu": round(cap.mu, 4),
                "sigma": round(cap.sigma, 4),
                "beta_sigma": round(bonus, 4),
                "ucb": round(ucb_val, 4),
            }
        return scores