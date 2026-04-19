class RewardSignals:
    """Raw signals collected at different time points."""
    def __init__(self, r_feedback: float, r_efficiency: float, r_quality: float, R: float):
        self.r_feedback = r_feedback
        self.r_efficiency = r_efficiency
        self.r_quality = r_quality
        self.R = R

class RewardFunction:
    """
    R = (r_feedback + λ1·r_efficiency + λ2·r_quality) / (1 + λ1 + λ2)
 
    Three signals:
      r_feedback:   (r_u + r_v) / 2           — immediate user reaction
      r_efficiency: clip(1 - (n-n_min)/(n_ref-n_min))  — dialogue rounds
      r_quality:    0.4·f_completion + 0.6·f_rating     — outcome
    """
 
    def __init__(
        self,
        lambda1: float = 0.5,
        lambda2: float = 1.0,
        n_min: int = 3,
        n_ref: int = 20,
    ):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_min = n_min
        self.n_ref = n_ref
 
    def compute(self, feedback: dict) -> RewardSignals:
        """
        Synthesize reward from raw feedback data.
 
        Parameters
        ----------
        feedback : dict with keys:
            r_u, r_v      — requester/candidate acceptance (0-1)
            n_rounds      — negotiation dialogue rounds
            f_completion  — task completion (0, 0.5, 1.0)
            stars_u       — requester rates candidate (1-5)
            stars_v       — candidate rates requester (1-5)
 
        Returns
        -------
        RewardSignals with r_feedback, r_efficiency, r_quality, R
        """
 
        # --- Signal 1: User feedback ---
        r_feedback = (feedback["r_u"] + feedback["r_v"]) / 2.0
 
        # --- Signal 2: Collaboration efficiency ---
        n = feedback["n_rounds"]
        if n <= self.n_min:
            r_efficiency = 1.0
        elif n >= self.n_ref:
            r_efficiency = 0.0
        else:
            r_efficiency = 1.0 - (n - self.n_min) / (self.n_ref - self.n_min)
 
        # --- Signal 3: Result quality ---
        f_completion = feedback["f_completion"]
 
        s_u = (feedback["stars_u"] - 1) / 4.0   # normalize 1-5 → 0-1
        s_v = (feedback["stars_v"] - 1) / 4.0
        f_rating = (s_u + s_v) / 2.0
 
        r_quality = 0.4 * f_completion + 0.6 * f_rating
 
        # --- Synthesis ---
        R = (r_feedback + self.lambda1 * r_efficiency + self.lambda2 * r_quality)
        R /= (1.0 + self.lambda1 + self.lambda2)
 
        return RewardSignals(
            r_feedback=round(r_feedback, 4),
            r_efficiency=round(r_efficiency, 4),
            r_quality=round(r_quality, 4),
            R=round(R, 4),
        )
 