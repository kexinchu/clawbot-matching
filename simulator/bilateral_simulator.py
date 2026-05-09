"""
simulator/bilateral_simulator.py
=================================
Orchestrates the full bilateral matching simulation:
  1. Generate personas for each side
  2. Run deliberation for each side
  3. Rank importance + top-k pruning for each side
  4. Compute side decisions
  5. Combine into joint action
"""
from __future__ import annotations

from simulator.config import SimulatorConfig
from simulator.decision_model import SideDecisionEngine
from simulator.deliberation import DeliberationEngine
from simulator.importance_ranker import rank_persona_importance, select_top_k
from simulator.persona_generator import CandidatePersonaGenerator, RequesterPersonaGenerator
from simulator.types import (
    BilateralDecisionResult,
    JointAction,
    MatchingContext,
    SideType,
)
from simulator.utils import TraceLogger


class BilateralSimulator:
    """
    Full bilateral matching simulator.

    Usage:
        sim = BilateralSimulator(backend, config)
        result = sim.run(context)
    """

    def __init__(
        self,
        backend,
        config: SimulatorConfig,
        logger: TraceLogger | None = None,
    ):
        self.backend = backend
        self.config = config
        self.logger = logger or TraceLogger(verbose=config.trace_verbose)
        self.delib_engine = DeliberationEngine()
        self.decision_engine = SideDecisionEngine()

    def run(self, context: MatchingContext) -> BilateralDecisionResult:
        """
        Run the full bilateral simulation pipeline.

        Returns BilateralDecisionResult with requester_decision, candidate_decision,
        joint_action, and joint_accept_prob.
        """
        # ---- Persona generation ----
        req_gen = RequesterPersonaGenerator(self.backend, self.config)
        cand_gen = CandidatePersonaGenerator(self.backend, self.config)

        requester_personas = req_gen.generate(context)
        candidate_personas = cand_gen.generate(context)

        self.logger.log("persona_generation", {
            "requester_personas": [p.persona_id for p in requester_personas],
            "candidate_personas": [p.persona_id for p in candidate_personas],
        })

        # ---- Deliberation: requester side ----
        req_trace = self.delib_engine.run(
            requester_personas, context, self.backend, self.config, SideType.REQUESTER
        )
        self.logger.log("requester_deliberation", {
            "stop_reason": req_trace.stop_reason,
            "num_rounds": len(req_trace.per_round_updates),
            "initial_utilities": {
                pid: round(op.utility_score, 3)
                for pid, op in req_trace.initial_opinions.items()
            },
            "final_utilities": {
                pid: round(op.utility_score, 3)
                for pid, op in req_trace.final_opinions.items()
            },
        })

        # ---- Deliberation: candidate side ----
        cand_trace = self.delib_engine.run(
            candidate_personas, context, self.backend, self.config, SideType.CANDIDATE
        )
        self.logger.log("candidate_deliberation", {
            "stop_reason": cand_trace.stop_reason,
            "num_rounds": len(cand_trace.per_round_updates),
            "initial_utilities": {
                pid: round(op.utility_score, 3)
                for pid, op in cand_trace.initial_opinions.items()
            },
            "final_utilities": {
                pid: round(op.utility_score, 3)
                for pid, op in cand_trace.final_opinions.items()
            },
        })

        # ---- Importance ranking: requester ----
        req_importance = rank_persona_importance(
            req_trace.final_opinions, context, req_trace, self.config
        )
        req_selected, req_pruned = select_top_k(
            requester_personas, req_importance, self.config
        )

        # ---- Importance ranking: candidate ----
        cand_importance = rank_persona_importance(
            cand_trace.final_opinions, context, cand_trace, self.config
        )
        cand_selected, cand_pruned = select_top_k(
            candidate_personas, cand_importance, self.config
        )

        self.logger.log("importance_ranking", {
            "requester_weights": {
                imp.persona_id: round(imp.normalized_weight, 4)
                for imp in req_importance
            },
            "requester_selected": [p.persona_id for p in req_selected],
            "requester_pruned": [p.persona_id for p in req_pruned],
            "candidate_weights": {
                imp.persona_id: round(imp.normalized_weight, 4)
                for imp in cand_importance
            },
            "candidate_selected": [p.persona_id for p in cand_selected],
            "candidate_pruned": [p.persona_id for p in cand_pruned],
        })

        # ---- Side decisions ----
        req_decision = self.decision_engine.decide(
            personas=requester_personas,
            final_opinions=req_trace.final_opinions,
            importance_scores=req_importance,
            selected_personas=req_selected,
            context=context,
            config=self.config,
            side=SideType.REQUESTER,
            trace=req_trace,
        )

        cand_decision = self.decision_engine.decide(
            personas=candidate_personas,
            final_opinions=cand_trace.final_opinions,
            importance_scores=cand_importance,
            selected_personas=cand_selected,
            context=context,
            config=self.config,
            side=SideType.CANDIDATE,
            trace=cand_trace,
        )

        self.logger.log("side_decisions", {
            "requester": {
                "utility": round(req_decision.utility, 4),
                "action": req_decision.action.value,
                "action_probs": {k: round(v, 4) for k, v in req_decision.action_probs.items()},
                "confidence": round(req_decision.confidence, 4),
                "rationale": req_decision.final_rationale,
            },
            "candidate": {
                "utility": round(cand_decision.utility, 4),
                "action": cand_decision.action.value,
                "action_probs": {k: round(v, 4) for k, v in cand_decision.action_probs.items()},
                "confidence": round(cand_decision.confidence, 4),
                "rationale": cand_decision.final_rationale,
            },
        })

        # ---- Joint action ----
        joint_action, joint_accept_prob = self._compute_joint_action(
            req_decision, cand_decision
        )

        self.logger.log("bilateral_result", {
            "joint_action": joint_action.value,
            "joint_accept_prob": round(joint_accept_prob, 4),
            "requester_action": req_decision.action.value,
            "candidate_action": cand_decision.action.value,
        })

        return BilateralDecisionResult(
            requester_decision=req_decision,
            candidate_decision=cand_decision,
            joint_action=joint_action,
            joint_accept_prob=joint_accept_prob,
            metadata={
                "trace": self.logger.entries,
            },
        )

    # ------------------------------------------------------------------
    # Joint action computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_joint_action(
        req: "DecisionResult",
        cand: "DecisionResult",
    ) -> tuple[JointAction, float]:
        """
        Map two side decisions to a joint action.

        joint_accept_prob is the product of both sides' accept probabilities,
        which serves as a proxy for bilateral success probability.
        """
        from simulator.types import Action

        req_act = req.action
        cand_act = cand.action

        # Joint accept probability
        joint_accept_prob = (
            req.action_probs.get("accept", 0.0)
            * cand.action_probs.get("accept", 0.0)
        )

        if req_act == Action.ACCEPT and cand_act == Action.ACCEPT:
            joint = JointAction.MUTUAL_ACCEPT
        elif req_act == Action.REJECT or cand_act == Action.REJECT:
            joint = JointAction.ONE_REJECT
        elif req_act == Action.ACCEPT and cand_act == Action.SKIP:
            joint = JointAction.REQUESTER_ACCEPT_CANDIDATE_SKIP
        elif req_act == Action.SKIP and cand_act == Action.ACCEPT:
            joint = JointAction.REQUESTER_SKIP_CANDIDATE_ACCEPT
        elif req_act == Action.REJECT and cand_act == Action.REJECT:
            joint = JointAction.MUTUAL_REJECT
        else:
            joint = JointAction.MUTUAL_SKIP

        return joint, joint_accept_prob
