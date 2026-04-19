from WorldModel import WorldModel
from Reward_function import RewardFunction
from Parameter_update import BayesianUpdater, WeightUpdater, UCBExplorer
from UserProfile import UserProfile
from Task import Task
from typing import List, Optional
from utils import dummy_user_feedback

class OnlineLearning:
    """
    Orchestrates the full Layer 5 learning loop:
      1. Compute match score (Layer 2)
      2. Simulate/collect feedback (Layer 4)
      3. Compute reward R
      4. Update parameters via three paths
    """
 
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.reward_fn = RewardFunction()
        self.bayesian_updater = BayesianUpdater()
        self.weight_updater = WeightUpdater()
        self.ucb_explorer = UCBExplorer()
        self.history: List[dict] = []
 
    def run_one_round(
        self,
        requester: UserProfile,
        candidate: UserProfile,
        task: Task,
        feedback_override: Optional[dict] = None,
    ) -> dict:
        """
        Execute one full learning round.
 
        Parameters
        ----------
        requester : the user who posted the task
        candidate : the user being matched
        task : the task to match for
        feedback_override : if provided, use this instead of dummy feedback
 
        Returns
        -------
        dict with all intermediate results for inspection
        """
 
        self.ucb_explorer.step()
        round_num = self.ucb_explorer.round
 
        # --- Layer 2: Compute match score ---
        match_result = self.world_model.compute_match(
            requester, candidate, task,
            use_ucb=True,
            beta_t=self.ucb_explorer.beta,
        )
 
        # Also compute without UCB for comparison
        match_no_ucb = self.world_model.compute_match(
            requester, candidate, task,
            use_ucb=False,
        )
 
        # --- Layer 4: Collect feedback ---
        if feedback_override:
            feedback = feedback_override
        else:
            feedback = dummy_user_feedback(match_no_ucb)
 
        # --- Layer 5.1: Compute reward ---
        reward = self.reward_fn.compute(feedback)
 
        # --- Layer 5.2: Path 1 — Bayesian update (μ, σ) ---
        bayes_updates = self.bayesian_updater.update(candidate, task, reward.R)
 
        # --- Layer 5.3: Path 2 — Weight SGD ---
        weight_updates = self.weight_updater.update(
            self.world_model, match_no_ucb, reward.R
        )
 
        # --- Layer 5.4: Path 3 — UCB state ---
        ucb_scores = self.ucb_explorer.compute_ucb_scores(candidate)
 
        # --- Assemble round report ---
        report = {
            "round": round_num,
            "beta_t": round(self.ucb_explorer.beta, 4),
            "match_score": {
                "M (no UCB)": round(match_no_ucb.M, 4),
                "M (with UCB)": round(match_result.M, 4),
                "S_cap": round(match_no_ucb.S_cap, 4),
                "S_need": round(match_no_ucb.S_need, 4),
            },
            "feedback": {
                "r_u": feedback["r_u"],
                "r_v": feedback["r_v"],
                "n_rounds": feedback["n_rounds"],
                "completion": feedback["f_completion"],
                "stars": f"u={feedback['stars_u']}, v={feedback['stars_v']}",
            },
            "reward": {
                "r_feedback": reward.r_feedback,
                "r_efficiency": reward.r_efficiency,
                "r_quality": reward.r_quality,
                "R": reward.R,
            },
            "path1_bayesian": bayes_updates,
            "path2_weights": weight_updates,
            "path3_ucb": ucb_scores,
        }
 
        self.history.append(report)
        return report