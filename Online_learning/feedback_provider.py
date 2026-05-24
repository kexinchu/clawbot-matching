"""L4 feedback providers — pluggable sources for user-side feedback.

Three implementations correspond to the three-stage evolution in the
original 5-layer design:

    DummyFeedback      — synthetic dict, cheap, deterministic per-seed (unit tests)
    SimulatorFeedback  — multi-persona deliberation (Bilateral-Matching-Simulator)
    HumanFeedback      — real users (Phase 3+)

All return the same six-field dict that the L5 pipeline already consumes:
    {r_u, r_v, n_rounds, f_completion, stars_u, stars_v}

Reward synthesis stays in Reward_function.RewardFunction so the original
formula R = (r_fb + λ₁·r_eff + λ₂·r_q) / (1+λ₁+λ₂) is untouched.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [
    os.path.join(_HERE, '..'),                  # repo root → simulator package
    os.path.join(_HERE, '..', 'mapping-algo'),  # datatypes
]:
    if _p not in sys.path:
        sys.path.append(_p)

from datatypes import UserState, Task, MatchResult
from ol_utils import dummy_user_feedback
from sim_adapter import build_matching_context

import simulator as sim
from simulator.types import Action


# ── Protocol ──────────────────────────────────────────────────────────

class FeedbackProvider(ABC):
    """Abstract L4 source. Returns the legacy six-field feedback dict."""

    @abstractmethod
    def collect(
        self,
        requester: UserState,
        candidate: UserState,
        task: Task,
        match: MatchResult,
    ) -> dict:
        ...


# ── Implementation 1: dummy ────────────────────────────────────────────

class DummyFeedback(FeedbackProvider):
    """Wrap the legacy dummy_user_feedback for the new protocol.

    Cheap, deterministic with np.random.seed, used by unit tests that need
    full control over reward direction.
    """

    def collect(self, requester, candidate, task, match) -> dict:
        return dummy_user_feedback(match)


# ── Implementation 2: simulator ────────────────────────────────────────

# Map simulator.Action to a numeric acceptance signal in [0, 1].
# Original L4 design: r_u/r_v ∈ {0.3, 0.5, 0.7, 1.0}.
# Map: accept→1.0, skip→0.5, reject→0.3.
_ACTION_TO_R = {
    Action.ACCEPT: 1.0,
    Action.SKIP:   0.5,
    Action.REJECT: 0.3,
}


class SimulatorFeedback(FeedbackProvider):
    """Use BilateralSimulator + OutcomeSimulator to synthesize feedback.

    We extract event-level signals (actions, expected_rounds, completion,
    satisfactions) and project them into the legacy six-field dict.
    The simulator's own compute_reward is intentionally NOT called —
    reward composition stays with Reward_function.RewardFunction.
    """

    def __init__(self, config: Optional[sim.SimulatorConfig] = None):
        self.config = config or sim.SimulatorConfig(
            backend_type="mock",
            random_seed=42,
            trace_verbose=False,
        )
        self.config.apply_seed()
        self.backend = sim.RuleBasedBackend(self.config)
        self.bilateral = sim.BilateralSimulator(
            self.backend, self.config,
            logger=sim.TraceLogger(verbose=False),
        )
        self.outcome_sim = sim.OutcomeSimulator(self.config)

    def collect(self, requester, candidate, task, match) -> dict:
        ctx = build_matching_context(requester, candidate, task, match)

        bilateral = self.bilateral.run(ctx)
        outcome = self.outcome_sim.simulate(ctx, bilateral)

        # Project decisions → r_u / r_v (∈ [0.3, 1.0])
        r_u = _ACTION_TO_R[bilateral.requester_decision.action]
        r_v = _ACTION_TO_R[bilateral.candidate_decision.action]

        # n_rounds: continuous expected_rounds → integer in [3, 25]
        n_rounds = max(3, min(25, int(round(outcome.expected_rounds))))

        # f_completion: continuous prob → quantized {0.0, 0.5, 1.0}
        cp = float(outcome.completion_probability)
        if cp >= 0.7:
            f_completion = 1.0
        elif cp >= 0.4:
            f_completion = 0.5
        else:
            f_completion = 0.0

        # stars: satisfaction in [0,1] → integer 1..5
        stars_u = max(1, min(5, int(round(outcome.requester_satisfaction * 5))))
        stars_v = max(1, min(5, int(round(outcome.candidate_satisfaction * 5))))

        return {
            "r_u": r_u,
            "r_v": r_v,
            "n_rounds": n_rounds,
            "f_completion": f_completion,
            "stars_u": stars_u,
            "stars_v": stars_v,
            # Extra fields below are ignored by RewardFunction but preserved
            # for analysis/logging. RewardFunction reads only the six above.
            "_sim_joint_action": bilateral.joint_action.value,
            "_sim_agreement_prob": float(outcome.agreement_probability),
            "_sim_joint_accept_prob": float(bilateral.joint_accept_prob),
        }


# ── Implementation 3: simulator + oracle skill-level observations ───────

class SkillLevelFeedback(FeedbackProvider):
    """Simulator feedback augmented with per-skill proficiency observations.

    The bilateral simulator still drives r_u / r_v / completion (Layer 4),
    but we also attach ``_skill_observations`` — a map from capability
    description → observed μ in [0, 1] — derived from the candidate's
    *true* profile.  This models "we asked peers to rate each skill after
    collaboration" without changing Reward_function's six-field formula.

    BayesianUpdater consumes ``_skill_observations`` when present, using
    the direct skill score as x_k instead of R · q_j.
    """

    def __init__(
        self,
        true_states_by_id: dict,
        config: Optional[sim.SimulatorConfig] = None,
    ):
        self.true_states = true_states_by_id
        self.sim = SimulatorFeedback(config=config)

    def collect(self, requester, candidate, task, match) -> dict:
        base = self.sim.collect(requester, candidate, task, match)
        true_v = self.true_states.get(candidate.user_id, candidate)
        skill_obs: dict = {}
        for cap in true_v.capabilities:
            if cap.description:
                skill_obs[cap.description] = float(cap.mu)
        base["_skill_observations"] = skill_obs
        return base


# ── Implementation 4: human (placeholder) ─────────────────────────────

class HumanFeedback(FeedbackProvider):
    """Wired in Phase 3+: collect feedback from real users via API/form."""

    def collect(self, requester, candidate, task, match) -> dict:
        raise NotImplementedError(
            "HumanFeedback requires Phase 3 API + form integration."
        )
