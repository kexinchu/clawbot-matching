"""Layer 2 world model — thin wrapper around mapping-algo/scoring.

Owns θ (softmax-parameterized weights) so WeightUpdater can mutate them
in place during SGD. All actual computation is delegated to compute_match_score.
"""

import sys
import os
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mapping-algo"); sys.path.append(_p) if _p not in sys.path else None  # os.path.join(os.path.dirname(__file__), '..', 'mapping-algo'))

import numpy as np
from datatypes import UserState, Task, MatchResult
from scoring import compute_match_score
from config import MatchConfig


class WorldModel:
    """
    Analytical Match Score: M = σ · (w_c · S_cap + w_n · S_need)

    Weights w = softmax(θ), where θ is the learnable parameter.
    Delegates all scoring (gate, attention soft-matching, UCB) to scoring.py.
    """

    def __init__(
        self,
        config: MatchConfig = None,
        theta_c: float = 0.4,
        theta_n: float = -0.1,
    ):
        # Use dim=64 default since test harness uses SimpleEncoder(dim=64).
        # In production, pass config=MatchConfig(embedding_dim=768).
        self.config = config if config is not None else MatchConfig(embedding_dim=64)
        self.theta = np.array([theta_c, theta_n])

    @property
    def weights(self) -> np.ndarray:
        e = np.exp(self.theta - self.theta.max())
        return e / e.sum()

    @property
    def w_c(self) -> float:
        return float(self.weights[0])

    @property
    def w_n(self) -> float:
        return float(self.weights[1])

    def compute_match(
        self,
        requester: UserState,
        candidate: UserState,
        task: Task,
        use_ucb: bool = False,
        round_t: int = 1,
    ) -> MatchResult:
        """Compute M(u, v, T) using the full scoring pipeline."""
        return compute_match_score(
            requester, candidate, task,
            theta=self.theta,
            cfg=self.config,
            use_ucb=use_ucb,
            round_t=round_t,
        )
