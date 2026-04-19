from dataclasses import dataclass

@dataclass
class Capability:
    """Single dimension of a user's capability: mean estimate + uncertainty."""
    mu: float       # ability estimate ∈ [0, 1]
    sigma: float    # uncertainty (high = system doesn't know yet)
 
    def ucb(self, beta: float) -> float:
        """Upper confidence bound: optimistic estimate for exploration."""
        return min(self.mu + beta * self.sigma, 1.0)