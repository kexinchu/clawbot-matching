"""
simulator/outcome_simulator.py
==============================
Simulates post-match interaction outcomes given bilateral decisions.

OutcomeSimulator (rule-based) produces:
  - agreement_probability
  - expected_rounds
  - completion_probability
  - requester_satisfaction
  - candidate_satisfaction
  - outcome_rationale
"""
from __future__ import annotations

from simulator.config import SimulatorConfig
from simulator.types import (
    BilateralDecisionResult,
    MatchingContext,
    OutcomeResult,
)
from simulator.utils import offer_need_fit, skill_coverage


class OutcomeSimulator:
    """
    Rule-based outcome simulator.

    The heuristics are:
      - agreement_probability: base rate modulated by joint action, utility gap,
        skill coverage, and highlighted risks
      - expected_rounds: proportional to disagreement and uncertainty
      - completion_probability: proportional to skill coverage and agreement prob
      - satisfaction: asymmetric utilities, scaled by outcome quality
    """

    def __init__(self, config: SimulatorConfig):
        self.config = config

    def simulate(
        self,
        context: MatchingContext,
        bilateral: BilateralDecisionResult,
    ) -> OutcomeResult:
        """
        Run outcome simulation.

        Returns an OutcomeResult with all outcome metrics.
        """
        req = bilateral.requester_decision
        cand = bilateral.candidate_decision

        # ---- Skill coverage ----
        skill_cov = skill_coverage(
            context.task.required_skills,
            context.candidate.capabilities,
        )

        # ---- Agreement probability ----
        agreement_prob = self._compute_agreement_prob(
            req=req, cand=cand, skill_cov=skill_cov,
            num_risks=len(context.card.highlighted_risks),
            bilateral=bilateral,
            context=context,
        )

        # ---- Expected rounds ----
        expected_rounds = self._compute_expected_rounds(
            req=req, cand=cand, skill_cov=skill_cov,
        )

        # ---- Completion probability ----
        completion_prob = self._compute_completion_prob(
            skill_cov=skill_cov,
            agreement_prob=agreement_prob,
            req_satisfaction=0.0,  # computed below
            cand_satisfaction=0.0,  # computed below
        )

        # ---- Satisfaction scores ----
        req_sat = self._compute_requester_satisfaction(
            req=req, skill_cov=skill_cov,
            agreement_prob=agreement_prob, completion_prob=completion_prob,
        )
        cand_sat = self._compute_candidate_satisfaction(
            cand=cand,
            skill_cov=skill_cov,
            agreement_prob=agreement_prob,
            context=context,
        )

        # ---- Refine completion prob with satisfaction ----
        completion_prob = self._compute_completion_prob(
            skill_cov=skill_cov,
            agreement_prob=agreement_prob,
            req_satisfaction=req_sat,
            cand_satisfaction=cand_sat,
        )

        # ---- Rationale ----
        rationale = self._build_rationale(
            context, bilateral, agreement_prob, expected_rounds,
            completion_prob, req_sat, cand_sat,
        )

        return OutcomeResult(
            agreement_probability=agreement_prob,
            expected_rounds=expected_rounds,
            completion_probability=completion_prob,
            requester_satisfaction=req_sat,
            candidate_satisfaction=cand_sat,
            outcome_rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Heuristic sub-methods
    # ------------------------------------------------------------------

    def _compute_agreement_prob(
        self,
        req,
        cand,
        skill_cov: float,
        num_risks: int,
        bilateral,
        context: MatchingContext,
    ) -> float:
        """
        agreement_probability ∈ [0, 1].

        Base rate from config, modulated by:
          - joint_action: mutual_accept boosts, one_reject penalizes
          - utility gap: larger gap → more negotiation friction
          - skill coverage: better coverage → higher agreement
          - observable offer/need fit: better candidate-side upside → higher agreement
          - highlighted risks: more risks → lower agreement
          - risk persona concerns
        """
        prob = self.config.base_agreement_prob

        # Joint action modifier
        from simulator.types import Action, JointAction
        if bilateral.joint_action == JointAction.MUTUAL_ACCEPT:
            prob += 0.20
        elif bilateral.joint_action == JointAction.MUTUAL_SKIP:
            prob -= 0.10
        elif bilateral.joint_action == JointAction.ONE_REJECT:
            prob -= 0.40
        elif bilateral.joint_action == JointAction.MUTUAL_REJECT:
            prob -= 0.50

        # Skill coverage contribution
        prob += 0.15 * (skill_cov - 0.5)  # +0.075 at full coverage, -0.075 at 0

        # Candidate-side observable upside. This is the first outcome-stage
        # point where TaskSpec.offers directly affects bilateral validity.
        fit = offer_need_fit(context.task.offers, context.candidate.needs)
        prob += 0.12 * (fit - 0.5)

        # Risk penalty
        prob -= 0.05 * num_risks

        # Utility gap penalty
        utility_gap = abs(req.utility - cand.utility)
        prob -= 0.10 * utility_gap

        # Risk persona concerns
        risk_concerns_req = sum(
            1 for op in req.all_persona_opinions.values()
            if "risk" in op.persona_role or "trust" in op.persona_role
        )
        risk_concerns_cand = sum(
            1 for op in cand.all_persona_opinions.values()
            if "risk" in op.persona_role or "trust" in op.persona_role
        )
        prob -= 0.03 * (risk_concerns_req + risk_concerns_cand)

        prob += self._deterministic_noise(context, scale=0.025)

        return _clamp(prob)

    def _compute_expected_rounds(
        self,
        req,
        cand,
        skill_cov: float,
    ) -> float:
        """
        expected_rounds ∈ [1, ∞).

        More rounds needed when:
          - The two sides disagree (different actions)
          - Utility gap is large
          - Confidence is low (uncertainty)
          - Skill coverage is poor (need renegotiation)
        """
        base = 2.0

        # Action disagreement
        if req.action != cand.action:
            base += 1.0

        # Utility gap
        gap = abs(req.utility - cand.utility)
        base += gap * 2.0

        # Low confidence
        avg_conf = (req.confidence + cand.confidence) / 2.0
        if avg_conf < 0.4:
            base += 1.5
        elif avg_conf < 0.6:
            base += 0.5

        # Poor skill coverage
        if skill_cov < 0.5:
            base += 1.0

        return round(base, 1)

    def _compute_completion_prob(
        self,
        skill_cov: float,
        agreement_prob: float,
        req_satisfaction: float,
        cand_satisfaction: float,
    ) -> float:
        """
        completion_probability ∈ [0, 1].

        Proportional to skill coverage, agreement probability,
        and mutual satisfaction.
        """
        prob = (
            0.30 * agreement_prob
            + 0.30 * skill_cov
            + 0.20 * req_satisfaction
            + 0.20 * cand_satisfaction
        )
        return _clamp(prob)

    def _compute_requester_satisfaction(
        self,
        req,
        skill_cov: float,
        agreement_prob: float,
        completion_prob: float,
    ) -> float:
        """
        requester_satisfaction ∈ [0, 1].

        Depends on:
          - utility score (direct)
          - skill coverage
          - whether mutual accept occurred
          - confidence
        """
        sat = 0.0
        sat += 0.35 * (req.utility + 1.0) / 2.0  # shift [-1,1] → [0,1]
        sat += 0.25 * skill_cov
        sat += 0.20 * agreement_prob
        sat += 0.10 * req.confidence
        sat += 0.10 * completion_prob
        return _clamp(sat)

    def _compute_candidate_satisfaction(
        self,
        cand,
        skill_cov: float,
        agreement_prob: float,
        context: MatchingContext,
    ) -> float:
        """
        candidate_satisfaction ∈ [0, 1].

        Fixed v3 oracle rule:
          - observable offer/need fit directly increases candidate-side value
          - workload and availability lower capacity/satisfaction
          - visibility and budget add observable task upside
          - hidden interpersonal and opportunity latents add private variation
          - small deterministic nonlinear noise avoids directly copying MapScore
        """
        fit = offer_need_fit(context.task.offers, context.candidate.needs)
        workload = float(context.candidate.constraints.get(
            "workload",
            context.candidate.preferences.get("current_load", 0.5),
        ))
        availability = context.candidate.preferences.get("availability", "medium")
        availability_score = {"high": 1.0, "medium": 0.6, "low": 0.25}.get(str(availability), 0.6)
        capacity = _clamp(0.60 * (1.0 - workload) + 0.40 * availability_score)

        visibility = context.task.metadata.get("visibility", "medium")
        budget = context.task.metadata.get("budget", "competitive")
        visibility_score = {"high": 1.0, "medium": 0.6, "low": 0.25}.get(str(visibility), 0.6)
        budget_score = {"premium": 1.0, "competitive": 0.65, "lean": 0.3}.get(str(budget), 0.65)
        visible_upside = 0.45 * visibility_score + 0.55 * budget_score

        affinity = context.latent_interpersonal_affinity or 0.0
        opportunity = context.latent_opportunity_bias or 0.0
        latent_bonus = 0.5 + 0.25 * affinity + 0.25 * opportunity

        nonlinear_fit = fit ** 1.35
        sat = 0.0
        sat += 0.26 * (cand.utility + 1.0) / 2.0
        sat += 0.24 * nonlinear_fit
        sat += 0.15 * capacity
        sat += 0.14 * visible_upside
        sat += 0.10 * agreement_prob
        sat += 0.07 * latent_bonus
        sat += 0.04 * cand.confidence
        sat += self._deterministic_noise(context, scale=0.02)
        return _clamp(sat)

    @staticmethod
    def _deterministic_noise(context: MatchingContext, scale: float) -> float:
        key = f"{context.task.task_id}:{context.candidate.user_id}"
        raw = sum((idx + 1) * ord(ch) for idx, ch in enumerate(key)) % 997
        centered = (raw / 996.0) - 0.5
        return scale * centered

    # ------------------------------------------------------------------
    # Rationale builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rationale(
        context: MatchingContext,
        bilateral: BilateralDecisionResult,
        agreement: float,
        rounds: float,
        completion: float,
        req_sat: float,
        cand_sat: float,
    ) -> str:
        parts = [
            f"Outcome simulation for task '{context.task.title}'",
            f"between requester '{context.requester.user_id}'",
            f"and candidate '{context.candidate.user_id}'.",
            f"Agreement probability: {agreement:.1%}.",
            f"Expected negotiation rounds: ~{rounds:.1f}.",
            f"Projected completion probability: {completion:.1%}.",
            f"Requester satisfaction: {req_sat:.1%}.",
            f"Candidate satisfaction: {cand_sat:.1%}.",
        ]
        return " ".join(parts)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
