"""
simulator/mock_backend.py
=========================
Rule-based mock backend — the default backend that lets the simulator run
fully offline with deterministic, reasonable outputs.
"""
from __future__ import annotations

import math
import random
from typing import Any

from simulator.config import SimulatorConfig
from simulator.llm_backend import BaseBackend
from simulator.types import (
    Action,
    CandidateCard,
    MatchingContext,
    PersonaOpinion,
    PersonaSpec,
    SideType,
    TaskSpec,
    UserProfile,
)
from simulator.utils import risk_score_from_concerns, skill_coverage


class RuleBasedBackend(BaseBackend):
    """
    Offline rule-based backend for all LLM-generation needs.

    This class provides deterministic mock responses for:
      - Persona generation (generate_personas)
      - Persona opinion / judgment (generate_opinion)
      - Deliberation summarization (summarize_discussion)

    All methods use the provided MatchingContext and config to produce
    sensible, reproducible outputs without any external API.
    """

    def __init__(self, config: SimulatorConfig):
        self.config = config

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> str:
        """Return a mock string response (prompt is unused in mock mode)."""
        return "[mock response]"

    def structured_generate(self, prompt: str,
                            schema: dict[str, Any] | None = None,
                            **kwargs) -> dict[str, Any] | str:
        return "[mock structured response]"

    # ------------------------------------------------------------------
    # Persona generation helpers
    # ------------------------------------------------------------------

    REQUESTER_PERSONA_TEMPLATES: list[PersonaSpec] = []

    CANDIDATE_PERSONA_TEMPLATES: list[PersonaSpec] = []

    @classmethod
    def _build_templates(cls) -> None:
        """Lazily build persona templates on first use."""

        if cls.REQUESTER_PERSONA_TEMPLATES:
            return

        cls.REQUESTER_PERSONA_TEMPLATES = [
            PersonaSpec(
                persona_id="rq_skill",
                persona_name="Skill Match Analyst",
                persona_role="skill_match_evaluator",
                focus_dimension="skill coverage of the task",
                short_instruction=(
                    "Evaluate how well the candidate's skills meet the task requirements. "
                    "Score high if required skills are covered; penalize gaps."
                ),
            ),
            PersonaSpec(
                persona_id="rq_goal",
                persona_name="Goal Coverage Evaluator",
                persona_role="goal_coverage_evaluator",
                focus_dimension="alignment between candidate abilities and project goals",
                short_instruction=(
                    "Assess whether the candidate can advance the requester's goals. "
                    "Consider their track record and stated interests."
                ),
            ),
            PersonaSpec(
                persona_id="rq_collab",
                persona_name="Collaboration Style Analyst",
                persona_role="collaboration_style_evaluator",
                focus_dimension="compatibility of working styles and communication preferences",
                short_instruction=(
                    "Judge whether the candidate's collaboration style fits the requester's. "
                    "Look for evidence of responsiveness and teamwork."
                ),
            ),
            PersonaSpec(
                persona_id="rq_time",
                persona_name="Time & Risk Analyst",
                persona_role="time_risk_evaluator",
                focus_dimension="likelihood of on-time delivery and risk factors",
                short_instruction=(
                    "Estimate delivery risk. Consider workload, prior deadlines, and "
                    "candidate availability. Flag overloaded or uncertain profiles."
                ),
            ),
            PersonaSpec(
                persona_id="rq_incentive",
                persona_name="Incentive Alignment Analyst",
                persona_role="incentive_evaluator",
                focus_dimension="alignment of incentives and expected effort",
                short_instruction=(
                    "Judge whether the candidate has strong intrinsic incentive to do "
                    "this task well. Look for interest alignment and upside."
                ),
            ),
            PersonaSpec(
                persona_id="rq_trust",
                persona_name="Trust & Safety Analyst",
                persona_role="trust_safety_evaluator",
                focus_dimension="reliability, past disputes, and safety concerns",
                short_instruction=(
                    "Evaluate trust signals: review history, dispute rate, "
                    "profile completeness. Flag any safety red flags."
                ),
            ),
        ]

        cls.CANDIDATE_PERSONA_TEMPLATES = [
            PersonaSpec(
                persona_id="cd_fit",
                persona_name="Capability Fit Analyst",
                persona_role="capability_fit_evaluator",
                focus_dimension="fit between own skills and task requirements",
                short_instruction=(
                    "Judge whether you (as candidate) have the required skill level. "
                    "Score high if well-matched, penalize if over- or under-qualified."
                ),
            ),
            PersonaSpec(
                persona_id="cd_interest",
                persona_name="Task Interest Analyst",
                persona_role="task_interest_evaluator",
                focus_dimension="genuine interest and motivation in this task",
                short_instruction=(
                    "Assess how interesting and meaningful this task is to you. "
                    "Higher interest → higher motivation and quality."
                ),
            ),
            PersonaSpec(
                persona_id="cd_opp",
                persona_name="Opportunity Cost Analyst",
                persona_role="opportunity_cost_evaluator",
                focus_dimension="opportunity cost relative to other tasks or uses of time",
                short_instruction=(
                    "Consider what else you could do with this time. "
                    "Higher opportunity cost → lower willingness to accept."
                ),
            ),
            PersonaSpec(
                persona_id="cd_workload",
                persona_name="Workload Balance Analyst",
                persona_role="workload_evaluator",
                focus_dimension="current workload and capacity to take on more",
                short_instruction=(
                    "Evaluate your current commitments. "
                    "If already near capacity, flag workload risk."
                ),
            ),
            PersonaSpec(
                persona_id="cd_recip",
                persona_name="Reciprocity Benefit Analyst",
                persona_role="reciprocity_benefit_evaluator",
                focus_dimension="long-term reputational or network upside",
                short_instruction=(
                    "Judge the networking, portfolio, and reputational upside. "
                    "High-visibility or career-advancing tasks score higher."
                ),
            ),
            PersonaSpec(
                persona_id="cd_trust",
                persona_name="Trust & Safety Analyst",
                persona_role="trust_safety_evaluator",
                focus_dimension="requester's reliability, payment history, and safety",
                short_instruction=(
                    "Check requester's history, payment reliability, and any flags. "
                    "Be cautious with low-history or high-risk requesters."
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Persona generation
    # ------------------------------------------------------------------

    def generate_personas(
        self,
        side: SideType,
        context: MatchingContext,
        num_personas: int | None = None,
    ) -> list[PersonaSpec]:
        """
        Select a diverse subset of persona templates for the given side.

        Returns a list of PersonaSpec objects (not yet evaluated).
        Selection is deterministic based on random.seed from config.
        """
        self._build_templates()

        templates = (
            self.REQUESTER_PERSONA_TEMPLATES
            if side == SideType.REQUESTER
            else self.CANDIDATE_PERSONA_TEMPLATES
        )

        num = num_personas or (
            self.config.num_requester_personas
            if side == SideType.REQUESTER
            else self.config.num_candidate_personas
        )

        # Use diverse subset (not just first N) via shuffle with seed
        rng = random.Random(self.config.persona_selection_seed)
        shuffled = rng.sample(templates, k=len(templates))
        return shuffled[: min(num, len(shuffled))]

    # ------------------------------------------------------------------
    # Persona judgment
    # ------------------------------------------------------------------

    def generate_opinion(
        self,
        persona: PersonaSpec,
        context: MatchingContext,
        round_idx: int = 0,
    ) -> PersonaOpinion:
        """
        Rule-based persona opinion generator.

        Computes a utility_score and action_probs from persona role + context.
        This is the core "mock LLM judge" for each persona.
        """
        rng = random.Random(
            hash(persona.persona_id + str(round_idx)) % (2**31)
        )

        if persona.persona_role == "skill_match_evaluator":
            score, concerns = self._skill_match(context)
        elif persona.persona_role == "goal_coverage_evaluator":
            score, concerns = self._goal_coverage(context)
        elif persona.persona_role == "collaboration_style_evaluator":
            score, concerns = self._collab_style(context)
        elif persona.persona_role == "time_risk_evaluator":
            score, concerns = self._time_risk(context)
        elif persona.persona_role == "incentive_evaluator":
            score, concerns = self._incentive_alignment(context)
        elif persona.persona_role == "trust_safety_evaluator":
            score, concerns = self._trust_safety(context)
        elif persona.persona_role == "capability_fit_evaluator":
            score, concerns = self._capability_fit(context)
        elif persona.persona_role == "task_interest_evaluator":
            score, concerns = self._task_interest(context)
        elif persona.persona_role == "opportunity_cost_evaluator":
            score, concerns = self._opportunity_cost(context)
        elif persona.persona_role == "workload_evaluator":
            score, concerns = self._workload_balance(context)
        elif persona.persona_role == "reciprocity_benefit_evaluator":
            score, concerns = self._reciprocity_benefit(context)
        else:
            # Fallback
            score, concerns = 0.0, []

        # Inject small noise for realism
        noise = rng.uniform(-0.05, 0.05)
        score = max(-1.0, min(1.0, score + noise))
        confidence = self._confidence_for_score(score, concerns)

        # Build action probabilities from utility score
        action_probs = self._utility_to_probs(score)

        rationale = self._build_rationale(persona, context, score, concerns)

        return PersonaOpinion(
            persona_id=persona.persona_id,
            persona_role=persona.persona_role,
            utility_score=score,
            action_probs=action_probs,
            confidence=confidence,
            rationale=rationale,
            extracted_concerns=concerns,
        )

    # ------------------------------------------------------------------
    # Domain-specific scoring sub-methods (requester personas)
    # ------------------------------------------------------------------

    def _skill_match(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        cov = skill_coverage(
            ctx.task.required_skills,
            ctx.candidate.capabilities,
        )
        # Penalty for highlighted risks
        risk_penalty = len(ctx.card.highlighted_risks) * 0.05
        score = cov - risk_penalty
        concerns = []
        if cov < 0.5:
            concerns.append("Significant skill gaps detected")
        elif cov < 0.8:
            concerns.append("Some skill gaps remain")
        return clamp_score(score), concerns

    def _goal_coverage(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        """Goal coverage based on task title/description vs candidate profile."""
        task_words = set(ctx.task.title.lower().split())
        cand_words = set(
            ctx.candidate.preferences.get("interests", "")
            .replace("_", " ").replace(",", " ").split()
        )
        overlap = len(task_words & cand_words)
        score = min(1.0, overlap / max(1, len(task_words)))
        concerns = [] if overlap >= 2 else ["Weak goal alignment detected"]
        return clamp_score(score * 0.8), concerns

    def _collab_style(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        """Collaboration style: check for prior history signals and latent affinity."""
        history = ctx.history or {}
        affinity = ctx.latent_interpersonal_affinity
        if history.get("prior_collaboration"):
            base = 0.85
            concerns = []
        elif history.get("prior_skipped"):
            base = 0.45
            concerns = ["Prior skipped interaction — possible style mismatch"]
        else:
            base = 0.55
            concerns = ["No prior collaboration history — style unknown"]

        # Latent interpersonal affinity shifts the score ±0.15
        if affinity is not None:
            base += 0.15 * affinity  # affinity ∈ [-1, 1] → shift ∈ [-0.15, +0.15]

        return clamp_score(base), concerns

    def _time_risk(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        """Time risk: look at candidate availability and task urgency."""
        urgency = ctx.task.metadata.get("urgency", "medium")
        availability = ctx.candidate.preferences.get("availability", "medium")
        risk = 0.0
        concerns = []
        if urgency == "high" and availability in ("low", "medium"):
            risk = 0.3
            concerns.append("High urgency task with limited availability — time risk")
        elif urgency == "medium" and availability == "low":
            risk = 0.15
            concerns.append("Moderate time pressure with low availability")
        score = 0.7 - risk
        return clamp_score(score), concerns

    def _incentive_alignment(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        """Incentive alignment: budget/compensation vs market rate."""
        budget = ctx.task.metadata.get("budget", "competitive")
        if budget == "above_market":
            score = 0.85
            concerns = []
        elif budget == "competitive":
            score = 0.65
            concerns = []
        elif budget == "below_market":
            score = 0.35
            concerns = ["Below-market compensation may reduce candidate motivation"]
        else:
            score = 0.50
            concerns = ["Budget information unclear"]
        return clamp_score(score), concerns

    def _trust_safety(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        """Trust & safety: check highlighted risks, history flags, and latent risk tolerance."""
        risk_count = len(ctx.card.highlighted_risks)
        history = ctx.history or {}
        dispute_rate = history.get("dispute_rate", 0.0)
        base = 0.8 - risk_count * 0.1 - dispute_rate * 0.3
        concerns = []
        if risk_count >= 2:
            concerns.append("Multiple risk flags in candidate card")
        if dispute_rate > 0.1:
            concerns.append("Elevated dispute rate in prior history")

        # Latent risk tolerance shifts the trust evaluation ±0.10
        # High latent_risk_tolerance → willing to overlook minor flags
        risk_tol = ctx.latent_risk_tolerance
        if risk_tol is not None:
            base += 0.10 * (risk_tol - 0.5)  # risk_tol ∈ [0,1]; shift ∈ [-0.05, +0.05]

        return clamp_score(base), concerns

    # ------------------------------------------------------------------
    # Domain-specific scoring sub-methods (candidate personas)
    # ------------------------------------------------------------------

    def _capability_fit(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        cov = skill_coverage(
            ctx.task.required_skills,
            ctx.candidate.capabilities,
        )
        concerns = []
        if cov < 0.5:
            concerns.append("Insufficient skill coverage — may struggle with task")
        return clamp_score(cov), concerns

    def _task_interest(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        """Task interest: keyword overlap + latent candidate preferences signal."""
        interests = ctx.candidate.preferences.get("interests", "")
        desc = ctx.task.description.lower()
        kw_hits = sum(1 for kw in interests.replace(",", " ").split() if kw in desc)
        score = min(1.0, kw_hits / max(1, 3))
        concerns = [] if score >= 0.4 else ["Task description shows limited alignment with candidate interests"]

        # Latent candidate preferences: shift score by latent signal × 0.15
        latent_prefs = ctx.latent_candidate_preferences
        if latent_prefs is not None:
            latent_bonus = sum(v for k, v in latent_prefs.items() if k in ctx.task.description.lower()) / max(1, len(latent_prefs))
            score += 0.15 * latent_bonus

        return clamp_score(score), concerns

    def _opportunity_cost(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        """High opportunity cost → lower willingness (score is a penalty-avoidance metric)."""
        busy = ctx.candidate.preferences.get("current_load", "medium")
        if busy == "high":
            base = 0.35
            concerns = ["High current workload — opportunity cost is significant"]
        elif busy == "medium":
            base = 0.55
            concerns = []
        else:
            base = 0.80
            concerns = []
        # Flip: higher score = lower opportunity cost = better

        # Latent opportunity bias shifts the perceived cost ±0.12
        # Positive bias → underestimates opportunity cost (optimistic)
        bias = ctx.latent_opportunity_bias
        if bias is not None:
            base += 0.12 * bias  # bias ∈ [-1, 1] → shift ∈ [-0.12, +0.12]

        return clamp_score(base), concerns

    def _workload_balance(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        workload = ctx.candidate.constraints.get("workload", 0.5)
        score = 1.0 - workload
        concerns = []
        if workload > 0.7:
            concerns.append("Candidate near capacity — overload risk")
        return clamp_score(score), concerns

    def _reciprocity_benefit(self, ctx: MatchingContext) -> tuple[float, list[str]]:
        """Reciprocity benefit modulated by latent opportunity bias."""
        visibility = ctx.task.metadata.get("visibility", "medium")
        portfolio = ctx.task.metadata.get("portfolio_boost", False)
        score = 0.5
        if portfolio:
            score += 0.2
        if visibility == "high":
            score += 0.15
        elif visibility == "low":
            score -= 0.1
        concerns = [] if score >= 0.6 else ["Low reciprocity / reputational upside"]

        # Latent opportunity bias modulates reciprocity perception ±0.10
        # People with positive opportunity_bias tend to see more upside everywhere
        bias = ctx.latent_opportunity_bias
        if bias is not None:
            score += 0.10 * bias  # bias ∈ [-1, 1] → shift ∈ [-0.10, +0.10]

        return clamp_score(score), concerns

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _confidence_for_score(
        self,
        score: float,
        concerns: list[str],
    ) -> float:
        """Higher confidence when score is decisive and few concerns exist."""
        base = 0.5
        if abs(score) > 0.6:
            base += 0.2
        if len(concerns) == 0:
            base += 0.15
        elif len(concerns) >= 2:
            base -= 0.15
        return max(0.1, min(0.95, base))

    def _utility_to_probs(self, score: float) -> dict[str, float]:
        """
        Map continuous utility score ∈ [-1, 1] to action probability distribution.
        Probabilistic mode: softmax-style with decision thresholds.
        """
        # Raw logits
        accept_logit = score * 2.0
        reject_logit = -score * 2.0
        skip_logit = -abs(score) * 0.5

        logits = [accept_logit, skip_logit, reject_logit]
        names = ["accept", "skip", "reject"]
        # Softmax
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        total = sum(exps)
        return {n: e / total for n, e in zip(names, exps)}

    def _build_rationale(
        self,
        persona: PersonaSpec,
        ctx: MatchingContext,
        score: float,
        concerns: list[str],
    ) -> str:
        direction = "positive" if score > 0.2 else "neutral" if score > -0.2 else "negative"
        concern_str = (" Concerns: " + "; ".join(concerns)) if concerns else ""
        return (
            f"[{persona.persona_name}] assessed the match as {direction} "
            f"(utility≈{score:.2f}). {persona.short_instruction}.{concern_str}"
        )

    # ------------------------------------------------------------------
    # Deliberation summarization
    # ------------------------------------------------------------------

    def summarize_discussion(
        self,
        opinions: list[PersonaOpinion],
        round_idx: int,
    ) -> str:
        """
        Summarize the deliberation state at round_idx.
        Used by the deliberation engine as the leader's "summary" text.
        """
        if not opinions:
            return "No opinions expressed yet."

        avg_util = sum(o.utility_score for o in opinions) / len(opinions)
        all_concerns = []
        for o in opinions:
            all_concerns.extend(o.extracted_concerns)

        summary_parts = [
            f"Round {round_idx} summary:",
            f"  Average utility: {avg_util:.3f}",
            f"  Total concerns raised: {len(all_concerns)}",
        ]
        if all_concerns:
            # Deduplicate concerns
            seen = set()
            for c in all_concerns:
                key = c.lower()
                if key not in seen:
                    seen.add(key)
                    summary_parts.append(f"  - {c}")
        return "\n".join(summary_parts)

    # ------------------------------------------------------------------
    # Opinion revision
    # ------------------------------------------------------------------

    def revise_opinion(
        self,
        opinion: PersonaOpinion,
        others_opinions: list[PersonaOpinion],
        summary: str,
        round_idx: int,
    ) -> PersonaOpinion:
        """
        Simple revision: pull opinion slightly toward the group mean
        and amplify confidence on repeatedly raised concerns.
        """
        if not others_opinions:
            return opinion

        group_mean = sum(o.utility_score for o in others_opinions) / len(others_opinions)
        all_concerns = []
        for o in others_opinions:
            all_concerns.extend(o.extracted_concerns)

        # Pull toward group mean (dampening factor)
        pull = 0.15
        new_score = opinion.utility_score + pull * (group_mean - opinion.utility_score)
        new_score = max(-1.0, min(1.0, new_score))

        # Confidence: increase if aligned with group, decrease if divergent
        alignment = abs(group_mean - opinion.utility_score)
        delta_conf = -0.1 * alignment if alignment > 0.2 else 0.05 * (1 - alignment)
        new_conf = max(0.1, min(0.95, opinion.confidence + delta_conf))

        # Add concerns that appear in multiple opinions
        concern_counts: dict[str, int] = {}
        for c in all_concerns:
            key = c.lower()
            concern_counts[key] = concern_counts.get(key, 0) + 1

        new_concerns = list(opinion.extracted_concerns)
        for c, cnt in concern_counts.items():
            if cnt >= 2 and c not in [x.lower() for x in new_concerns]:
                new_concerns.append(c.title())

        # Recompute action_probs
        new_probs = self._utility_to_probs(new_score)

        return PersonaOpinion(
            persona_id=opinion.persona_id,
            persona_role=opinion.persona_role,
            utility_score=new_score,
            action_probs=new_probs,
            confidence=new_conf,
            rationale=(
                f"[Revised r{round_idx}] {opinion.rationale} "
                f"→ Group mean={group_mean:.2f}, new_score={new_score:.2f}"
            ),
            extracted_concerns=new_concerns,
        )


def clamp_score(value: float) -> float:
    return max(-1.0, min(1.0, value))
