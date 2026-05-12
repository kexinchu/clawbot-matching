import sys
import os
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mapping-algo"); sys.path.append(_p) if _p not in sys.path else None  # os.path.join(os.path.dirname(__file__), '..', 'mapping-algo'))
_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(_repo) if _repo not in sys.path else None

import math
from collections import defaultdict

from datatypes import UserState, Task
from WorldModel import WorldModel
from Reward_function import RewardFunction
from Parameter_update import BayesianUpdater, WeightUpdater, UCBExplorer
from pipeline import match_one_to_one
from typing import Dict, List, Optional, Tuple
from feedback_provider import FeedbackProvider, DummyFeedback
from LLM_Dreaming.LLM_dreaming import DreamSimulator, PlanningLayer


class OnlineLearning:
    """
    Orchestrates the full Layer 5 learning loop:
      1. Compute match score (Layer 2, via WorldModel → scoring.py)
      2. UCB+Greedy match (Layer 3.1)
      3. Dreaming filtering for top 3 candidates (Layer 3.2)
      4. Collect feedback (Layer 4) from a pluggable FeedbackProvider
      5. Compute reward R (Layer 5.1) — original formula preserved
      6. Update parameters via three paths:
           Path 1 — Bayesian update (μ, σ) per capability
           Path 2 — Weight SGD (θ → w_c, w_n)
           Path 3 — UCB state tracking
    """

    def __init__(
        self,
        world_model: WorldModel,
        feedback_provider: Optional[FeedbackProvider] = None,
        dream_simulator: Optional[DreamSimulator] = None,
        top_k: int = 6,
        top_n: int = 3,
        enable_dreaming: bool = True,
        candidate_ucb_c: float = math.sqrt(2.0),
        enable_candidate_ucb: bool = True,
        enable_weight_update: bool = True,
        use_capability_ucb: bool = True,
    ):
        self.world_model = world_model
        self.feedback_provider = feedback_provider or DummyFeedback()
        self.reward_fn = RewardFunction()
        self.bayesian_updater = BayesianUpdater()
        self.weight_updater = WeightUpdater()
        self.ucb_explorer = UCBExplorer()
        self.layer3_top_k = top_k
        self.layer3_top_n = top_n
        self.enable_dreaming = enable_dreaming
        self.enable_candidate_ucb = enable_candidate_ucb
        self.enable_weight_update = enable_weight_update
        self.use_capability_ucb = use_capability_ucb
        self.candidate_ucb_c = candidate_ucb_c
        # Per-candidate selection counter (n_v in UCB1).
        self.selection_counts: Dict[str, int] = defaultdict(int)
        self.dream_simulator = dream_simulator or DreamSimulator()
        self.planning_layer = PlanningLayer(
            world_model=self.world_model,
            dream_simulator=self.dream_simulator,
            top_k=top_k,
            top_n=top_n,
        )
        self.history: List[dict] = []

    def _build_candidate_lookup(self, pool: List[UserState]) -> Dict[str, UserState]:
        return {candidate.user_id: candidate for candidate in pool}

    def run_ucb_ranking(
        self,
        requester: UserState,
        candidate_pool: List[UserState],
        task: Task,
        round_num: int,
    ) -> List:
        """
        Layer 3.1: UCB ranking over a candidate pool.
        """
        return match_one_to_one(
            requester,
            task,
            candidate_pool,
            self.world_model.theta,
            self.world_model.config,
            top_k=self.layer3_top_k,
            use_ucb=self.use_capability_ucb,
            round_t=round_num,
        )

    def run_dreaming_filtering(
        self,
        requester: UserState,
        candidate_pool: List[UserState],
        task: Task,
        layer3_1_results: List,
    ) -> List:
        """
        Layer 3.2: Dream refinement for the top ranked candidates.
        """
        if not self.enable_dreaming or not layer3_1_results:
            return []

        return self.planning_layer.run_from_layer3_output(
            requester=requester,
            task=task,
            candidate_profiles=candidate_pool,
            layer3_output=layer3_1_results,
            verbose=False,
        )

    @staticmethod
    def _avg_capability_sigma(candidate: Optional[UserState]) -> float:
        """Mean σ over a candidate's capabilities (0.0 if no capabilities)."""
        if candidate is None or not candidate.capabilities:
            return 0.0
        sigmas = [float(cap.sigma) for cap in candidate.capabilities]
        return sum(sigmas) / len(sigmas)

    def select_with_candidate_ucb(
        self,
        entries: List[Tuple[str, float]],
        round_num: int,
        candidates_by_id: Optional[Dict[str, UserState]] = None,
    ) -> Tuple[Optional[str], List[dict]]:
        """Pick a candidate from a ranked pool using a sigma-weighted UCB.

            M̃_v = M_v + c · σ̄_v · sqrt( log(t) / (n_v + 1) )

        where ``t`` is the global round counter (``round_num``), ``n_v`` is
        the number of times candidate ``v`` has been selected so far, and
        ``σ̄_v`` is the mean Capability.sigma over candidate v's abilities.

        Compared to vanilla UCB1, σ̄_v scales the exploration bonus by the
        agent's own uncertainty about that candidate's capabilities, so
        well-known candidates (low σ̄_v) do not get force-explored. The
        ``+ 1`` in the denominator removes the special ``n_v == 0 → ∞``
        warmup branch — unseen candidates simply receive the largest finite
        bonus, still gated by σ̄_v. Ties are broken by the base score.
        """
        if not entries:
            return None, []

        log_t = math.log(max(round_num, 2))
        breakdown: List[dict] = []
        best_id: Optional[str] = None
        best_key: Tuple[float, float] = (-math.inf, -math.inf)

        for cand_id, base in entries:
            n_v = self.selection_counts.get(cand_id, 0)
            candidate = candidates_by_id.get(cand_id) if candidates_by_id else None
            avg_sigma_v = self._avg_capability_sigma(candidate)
            if not self.enable_candidate_ucb:
                bonus = 0.0
                adjusted = float(base)
            else:
                bonus = (
                    self.candidate_ucb_c
                    * avg_sigma_v
                    * math.sqrt(log_t / (n_v + 1))
                )
                adjusted = float(base) + bonus

            breakdown.append({
                "candidate_id": cand_id,
                "base_score": round(float(base), 4),
                "n_v": n_v,
                "avg_sigma_v": round(avg_sigma_v, 4),
                "bonus": round(bonus, 4),
                "adjusted_score": round(adjusted, 4),
            })

            key = (adjusted, float(base))
            if key > best_key:
                best_key = key
                best_id = cand_id

        return best_id, breakdown

    def run_one_round(
        self,
        requester: UserState,
        candidate: Optional[UserState],
        task: Task,
        feedback_override: Optional[dict] = None,
        candidate_pool: Optional[List[UserState]] = None,
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
        if candidate_pool is None and candidate is None:
            raise ValueError("run_one_round requires either a candidate or a candidate_pool")

        candidate_pool = list(candidate_pool) if candidate_pool is not None else [candidate]
        if candidate is None:
            candidate = candidate_pool[0]
        if not any(c.user_id == candidate.user_id for c in candidate_pool):
            candidate_pool.append(candidate)
        candidates_by_id = self._build_candidate_lookup(candidate_pool)

        # --- Layer 3.1: UCB+Greedy style pool ranking ---
        ucb_ranking_results = self.run_ucb_ranking(
            requester, candidate_pool, task, round_num
        )

        # --- Layer 3.2: Dream refinement for the top candidates ---
        dreaming_filtered_results = self.run_dreaming_filtering(
            requester, candidate_pool, task, ucb_ranking_results
        )

        # --- Candidate-level UCB selection (UCB1 over candidates) ---
        # Build (candidate_id, base_score) entries: prefer dreaming combined
        # score, fall back to layer 3.1 match score.
        if dreaming_filtered_results:
            entries: List[Tuple[str, float]] = [
                (r.candidate_id, float(r.combined_score))
                for r in dreaming_filtered_results
            ]
            selection_source = "layer3_2"
        elif ucb_ranking_results:
            entries = [
                (r.candidate_id, float(r.match_score))
                for r in ucb_ranking_results
            ]
            selection_source = "layer3_1"
        else:
            entries = []
            selection_source = "fallback"

        ucb_selected_id, candidate_ucb_breakdown = self.select_with_candidate_ucb(
            entries, round_num, candidates_by_id=candidates_by_id
        )
        selected_candidate_id = ucb_selected_id or candidate.user_id
        selected_candidate = candidates_by_id.get(selected_candidate_id, candidate)
        self.selection_counts[selected_candidate.user_id] += 1

        # --- Layer 2: Compute match score for the selected candidate ---
        match_result = self.world_model.compute_match(
            requester, selected_candidate, task,
            use_ucb=self.use_capability_ucb,
            round_t=round_num,
        )

        # Also compute without UCB for reward/feedback
        match_no_ucb = self.world_model.compute_match(
            requester, selected_candidate, task,
            use_ucb=False,
            round_t=round_num,
        )

        # --- Layer 4: Collect feedback ---
        if feedback_override:
            feedback = feedback_override
        else:
            feedback = self.feedback_provider.collect(
                requester, selected_candidate, task, match_no_ucb
            )

        # --- Layer 5.1: Compute reward ---
        reward = self.reward_fn.compute(feedback)

        # --- Layer 5.2: Path 1 — Bayesian update (μ, σ) ---
        bayes_updates = self.bayesian_updater.update(selected_candidate, task, reward.R)

        # --- Layer 5.3: Path 2 — Weight SGD ---
        if self.enable_weight_update:
            weight_updates = self.weight_updater.update(
                self.world_model, match_no_ucb, reward.R
            )
        else:
            weight_updates = {"skipped": True}

        # --- Layer 5.4: Path 3 — UCB state ---
        ucb_scores = self.ucb_explorer.compute_ucb_scores(selected_candidate)

        report = {
            "round": round_num,
            "beta_t": round(self.ucb_explorer.beta, 4),
            "selected_candidate_id": selected_candidate.user_id,
            "selection_counts": dict(self.selection_counts),
            "candidate_ucb": {
                "source": selection_source,
                "c": round(self.candidate_ucb_c, 4),
                "formula": "M_v + c * avg_sigma_v * sqrt(log(t)/(n_v+1))",
                "breakdown": candidate_ucb_breakdown,
            },
            "layer3_1": {
                "candidate_ids": [result.candidate_id for result in ucb_ranking_results],
                "num_ranked_candidates": len(ucb_ranking_results),
            },
            "layer3_2": {
                "candidate_ids": [result.candidate_id for result in dreaming_filtered_results],
                "num_refined_candidates": len(dreaming_filtered_results),
                "recommendations": [
                    result.recommendation for result in dreaming_filtered_results
                ],
            },
            "match_score": {
                "M (no UCB)": round(match_no_ucb.match_score, 4),
                "M (with UCB)": round(match_result.match_score, 4),
                "S_cap": round(match_no_ucb.s_cap, 4),
                "S_need": round(match_no_ucb.s_need, 4),
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
