"""InterpretabilityProbe — wraps an OnlineLearning engine and records
per-round snapshots without touching the production code.

What we capture per round:
  - selected_candidate_id
  - theta + softmax weights (w_c, w_n)
  - match_no_ucb's gap_details (per soft requirement: gap, p_tilde_u,
    p_tilde_v, coverage)
  - match_no_ucb's need_details (per need: intensity, offer_matched,
    satisfied)
  - μ, σ for every candidate's capabilities (so we can plot μ-error
    trajectories per critical/non-critical cap)
  - reward.R
  - When dreaming is enabled:
      * Layer 3.1 ranking (candidate_ids by analytical M)
      * Layer 3.2 ranking (candidate_ids by combined_score)
      * Per dream-refined candidate: analytical / dream / combined score,
        recommendation, top_risk, top_synergy
  - When the feedback dict carries simulator metadata (_sim_joint_action,
    _sim_joint_accept_prob, _sim_agreement_prob), we record those too.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_ONLINE = _REPO_ROOT / "Online_learning"
_MAPPING_ALGO = _REPO_ROOT / "mapping-algo"
for _p in (str(_ONLINE), str(_MAPPING_ALGO), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datatypes import UserState, Task  # noqa: E402
from Online_learning import OnlineLearning  # noqa: E402


def _snapshot_candidate_caps(candidate: UserState) -> List[Dict[str, float]]:
    return [
        {
            "description": cap.description,
            "mu": float(cap.mu),
            "sigma": float(cap.sigma),
            "source": cap.source,
        }
        for cap in candidate.capabilities
    ]


def _snapshot_pool(pool: List[UserState]) -> Dict[str, List[Dict[str, float]]]:
    return {c.user_id: _snapshot_candidate_caps(c) for c in pool}


class InterpretabilityProbe:
    """Drive an OnlineLearning engine and accumulate per-round snapshots.

    Usage:
        probe = InterpretabilityProbe(engine)
        for t in range(n_rounds):
            probe.step(requester, task, candidate_pool)
        snapshots = probe.snapshots  # list of dicts, one per round
    """

    def __init__(self, engine: OnlineLearning):
        self.engine = engine
        self.snapshots: List[Dict] = []

    def step(
        self,
        requester: UserState,
        task: Task,
        candidate_pool: List[UserState],
    ) -> Dict:
        report = self.engine.run_one_round(
            requester=requester,
            candidate=None,
            task=task,
            candidate_pool=candidate_pool,
        )

        wm = self.engine.world_model
        gap_details = []
        need_details = []
        try:
            match_no_ucb = wm.compute_match(
                requester,
                next(c for c in candidate_pool
                     if c.user_id == report["selected_candidate_id"]),
                task,
                use_ucb=False,
                round_t=report["round"],
            )
            gap_details = [g.to_dict() | {"req_description": g.req_description,
                                          "q_j": g.q_j,
                                          "p_tilde_u": g.p_tilde_u,
                                          "p_tilde_v": g.p_tilde_v}
                           for g in match_no_ucb.gap_details]
            need_details = [n.to_dict() | {"need_description": n.need_description,
                                           "need_intensity": n.need_intensity,
                                           "offer_matched": n.offer_matched}
                            for n in match_no_ucb.need_details]
        except (AttributeError, StopIteration):
            pass

        theta = np.asarray(wm.theta, dtype=float)
        weights = wm.weights

        l31 = report.get("layer3_1", {}) or {}
        l32 = report.get("layer3_2", {}) or {}
        layer3_1_ids: List[str] = list(l31.get("candidate_ids") or [])
        layer3_2_ids: List[str] = list(l32.get("candidate_ids") or [])

        feedback_meta: Dict[str, object] = {}
        try:
            fb = self.engine.history[-1]["feedback"]
            feedback_meta = {
                "r_u": fb.get("r_u"),
                "r_v": fb.get("r_v"),
                "n_rounds": fb.get("n_rounds"),
                "f_completion": fb.get("f_completion"),
            }
        except (IndexError, AttributeError, KeyError):
            pass

        snap = {
            "round": int(report["round"]),
            "selected_candidate_id": report["selected_candidate_id"],
            "theta": [float(theta[0]), float(theta[1])],
            "weights": {"w_c": float(weights[0]), "w_n": float(weights[1])},
            "match_no_ucb": float(report["match_score"]["M (no UCB)"]),
            "match_with_ucb": float(report["match_score"]["M (with UCB)"]),
            "s_cap": float(report["match_score"]["S_cap"]),
            "s_need": float(report["match_score"]["S_need"]),
            "reward_R": float(report["reward"]["R"]),
            "gap_details": gap_details,
            "need_details": need_details,
            "candidate_caps": _snapshot_pool(candidate_pool),
            "layer3_1_ranking": layer3_1_ids,
            "layer3_2_ranking": layer3_2_ids,
            "layer3_2_recommendations": list(l32.get("recommendations") or []),
            "feedback": feedback_meta,
        }
        self.snapshots.append(snap)
        return snap

    def initial_snapshot(
        self,
        requester: UserState,
        task: Task,
        candidate_pool: List[UserState],
    ) -> Dict:
        """Record the pre-training state (round 0). Useful to compare
        μ-error trajectories starting from the priors.
        """
        wm = self.engine.world_model
        theta = np.asarray(wm.theta, dtype=float)
        weights = wm.weights
        snap = {
            "round": 0,
            "selected_candidate_id": None,
            "theta": [float(theta[0]), float(theta[1])],
            "weights": {"w_c": float(weights[0]), "w_n": float(weights[1])},
            "match_no_ucb": None,
            "match_with_ucb": None,
            "s_cap": None,
            "s_need": None,
            "reward_R": None,
            "gap_details": [],
            "need_details": [],
            "candidate_caps": _snapshot_pool(candidate_pool),
            "layer3_1_ranking": [],
            "layer3_2_ranking": [],
            "layer3_2_recommendations": [],
            "feedback": {},
        }
        self.snapshots.insert(0, snap)
        return snap
