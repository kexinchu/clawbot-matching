"""
simulator/decision_model.py
============================
Side-specific utility aggregation and action mapping.

SideDecisionEngine:
  - Aggregates persona opinions into a side utility score
  - Maps utility to an action (accept / skip / reject)
  - Supports probabilistic and threshold modes
"""
from __future__ import annotations

import math

from simulator.config import SimulatorConfig
from simulator.types import (
    Action,
    DeliberationTrace,
    DecisionResult,
    MatchingContext,
    PersonaImportance,
    PersonaOpinion,
    PersonaSpec,
    SideType,
)
from simulator.utils import weighted_sum, sample_action_from_probs


class SideDecisionEngine:
    """
    Aggregates per-persona opinions into a single side decision.

    Pipeline:
      1. Deliberation → final opinions
      2. Importance ranking → persona weights
      3. Top-k pruning → selected personas
      4. Weighted aggregation → side utility
      5. Action mapping → accept / skip / reject
    """

    def __init__(self):
        pass

    def decide(
        self,
        personas: list[PersonaSpec],
        final_opinions: dict[str, PersonaOpinion],
        importance_scores: list[PersonaImportance],
        selected_personas: list[PersonaSpec],
        context: MatchingContext,
        config: SimulatorConfig,
        side: SideType,
        trace: DeliberationTrace,
    ) -> DecisionResult:
        """
        Compute the final decision for one side.

        Returns a DecisionResult with utility, action, action_probs, etc.
        """
        # Build opinion dict for selected personas
        selected_ids = {p.persona_id for p in selected_personas}
        selected_opinions = {
            pid: op for pid, op in final_opinions.items()
            if pid in selected_ids
        }

        # Aggregate utility: U = sum(alpha_i * z_i)
        weights_map = {imp.persona_id: imp.normalized_weight for imp in importance_scores}
        utility = self._aggregate_utility(selected_opinions, weights_map)

        # Aggregate action probabilities (weighted average)
        action_probs = self._aggregate_action_probs(selected_opinions, weights_map)

        # Confidence: weighted average of selected personas' confidence
        confidence = sum(
            op.confidence * weights_map.get(op.persona_id, 0.0)
            for op in selected_opinions.values()
        ) / sum(weights_map.get(op.persona_id, 0.0) for op in selected_opinions.values()) if selected_opinions else 0.5

        # Map utility to action (probabilistic mode uses action_probs, not utility threshold)
        action = self._utility_to_action(utility, config, action_probs)

        # Build importance dict
        importance_dict = {imp.persona_id: imp.normalized_weight for imp in importance_scores}

        # Build rationale
        rationale = self._build_rationale(
            side, utility, action, selected_opinions, importance_scores
        )

        return DecisionResult(
            side=side,
            utility=utility,
            action=action,
            action_probs=action_probs,
            confidence=confidence,
            selected_personas=[p.persona_id for p in selected_personas],
            all_persona_opinions=final_opinions,
            importance_scores=importance_dict,
            final_rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_utility(
        opinions: dict[str, PersonaOpinion],
        weights: dict[str, float],
    ) -> float:
        """U = sum(alpha_i * z_i)"""
        if not opinions:
            return 0.0
        total = sum(
            op.utility_score * weights.get(op.persona_id, 0.0)
            for op in opinions.values()
        )
        weight_sum = sum(weights.get(op.persona_id, 0.0) for op in opinions.values())
        return total / weight_sum if weight_sum > 0 else 0.0

    @staticmethod
    def _aggregate_action_probs(
        opinions: dict[str, PersonaOpinion],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Weighted average of action probability distributions."""
        total_w = sum(weights.get(op.persona_id, 0.0) for op in opinions.values())
        if total_w == 0:
            return {"accept": 0.33, "skip": 0.34, "reject": 0.33}

        result = {"accept": 0.0, "skip": 0.0, "reject": 0.0}
        for op in opinions.values():
            w = weights.get(op.persona_id, 0.0)
            for action in result:
                result[action] += op.action_probs.get(action, 0.0) * w
        for action in result:
            result[action] /= total_w
        return result

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _utility_to_action(
        utility: float,
        config: SimulatorConfig,
        action_probs: dict[str, float] | None = None,
    ) -> Action:
        """
        Map aggregated side utility to an action.

        Probabilistic mode: sample from the provided action_probs distribution.
        Threshold mode: hard threshold on utility value.
        """
        if config.decision_mode == "probabilistic" and action_probs is not None:
            return sample_action_from_probs(action_probs)
        return _utility_to_action_deterministic(utility, config)


    # ------------------------------------------------------------------
    # Rationale builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rationale(
        side: SideType,
        utility: float,
        action: Action,
        selected_opinions: dict[str, PersonaOpinion],
        importance_scores: list[PersonaImportance],
    ) -> str:
        """Build a human-readable rationale for the side's decision."""
        side_name = side.value.capitalize()
        action_name = action.value.capitalize()

        top_personas = sorted(
            importance_scores, key=lambda x: x.normalized_weight, reverse=True
        )[:3]

        top_parts = [
            f"{imp.persona_id}(w={imp.normalized_weight:.2f})"
            for imp in top_personas
        ]

        avg_conf = (
            sum(op.confidence for op in selected_opinions.values())
            / max(1, len(selected_opinions))
        )

        rationale = (
            f"{side_name} side: {action_name} decision (utility={utility:.3f}, "
            f"avg_confidence={avg_conf:.2f}). "
            f"Top personas: {', '.join(top_parts)}."
        )
        return rationale


def _utility_to_action_deterministic(utility: float, config: SimulatorConfig) -> Action:
    """Hard threshold mapping from utility to action."""
    if utility >= config.accept_utility_threshold:
        return Action.ACCEPT
    elif utility <= config.reject_utility_threshold:
        return Action.REJECT
    else:
        return Action.SKIP


def action_from_probs(action_probs: dict[str, float]) -> Action:
    """Return the most probable action."""
    return Action(max(action_probs, key=action_probs.get))
