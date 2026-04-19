import numpy as np
from UserProfile import UserProfile
from Task import Task
from MatchResult import MatchResult

class WorldModel:
    """
    Analytical Match Score: M = σ · (w_c · S_cap + w_n · S_need)
 
    Weights w = softmax(θ), where θ is the learnable parameter.
    """
 
    def __init__(self, theta_c: float = 0.4, theta_n: float = -0.1):
        self.theta = np.array([theta_c, theta_n])   # unconstrained params
        self.alpha = 0.5    # offer coefficient
        self.epsilon = 1e-8
 
    @property
    def weights(self) -> np.ndarray:
        """w = softmax(θ), always sums to 1."""
        exp_theta = np.exp(self.theta - np.max(self.theta))  # numerical stability
        return exp_theta / exp_theta.sum()
 
    @property
    def w_c(self) -> float:
        return self.weights[0]
 
    @property
    def w_n(self) -> float:
        return self.weights[1]
 
    def compute_match(
        self,
        requester: UserProfile,
        candidate: UserProfile,
        task: Task,
        use_ucb: bool = False,
        beta_t: float = 0.0,
    ) -> MatchResult:
        """
        Compute M(u, v, T).
 
        If use_ucb=True, replaces μ_v^k with μ_v^k + β·σ_v^k
        in S_cap calculation for exploration.
        """
        domains = task.Q_T.keys()
 
        # Step 1: Gap = ReLU(Q_T - μ_u)
        gap = {}
        for d in domains:
            mu_u = requester.capability[d].mu if d in requester.capability else 0.0
            gap[d] = max(task.Q_T[d] - mu_u, 0.0)
 
        # Step 2: S_cap — coverage of gap by candidate
        # uniform domain weight for now (w_k = 1 for all k)
        numerator_cap = 0.0
        denominator_cap = 0.0
        for d in domains:
            if d in candidate.capability:
                mu_v = candidate.capability[d].mu
                if use_ucb:
                    mu_v = candidate.capability[d].ucb(beta_t)
                numerator_cap += min(mu_v, gap[d])
            denominator_cap += gap[d]
        S_cap = numerator_cap / (denominator_cap + self.epsilon)
 
        # Step 3: S_need — how much does the task satisfy candidate's needs?
        numerator_need = 0.0
        denominator_need = 0.0
        for d in domains:
            offer_k = self.alpha * task.Q_T[d]
            need_k = candidate.need.get(d, 0.0)
            numerator_need += min(offer_k, need_k)
            denominator_need += need_k
        S_need = numerator_need / (denominator_need + self.epsilon)
 
        # Step 4: M = w_c · S_cap + w_n · S_need (σ=1 assumed)
        M = self.w_c * S_cap + self.w_n * S_need
 
        return MatchResult(S_cap=S_cap, S_need=S_need, M=M, gap=gap)
 