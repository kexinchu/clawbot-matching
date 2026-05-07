"""
simulator/deliberation.py
=========================
Multi-persona deliberation engine.

DeliberationEngine runs round-based internal discussion:
  - Round 0: all personas give initial opinions
  - Round 1+: leader summarizes conflicts, personas revise

Early stopping: if majority action probability exceeds threshold, stop.
"""
from __future__ import annotations

from simulator.config import SimulatorConfig
from simulator.llm_backend import BaseBackend
from simulator.mock_backend import RuleBasedBackend
from simulator.persona_judge import judge_persona_opinion
from simulator.types import (
    DeliberationTrace,
    MatchingContext,
    PersonaOpinion,
    PersonaSpec,
    RoundUpdate,
    SideType,
)


class DeliberationEngine:
    """
    Runs multi-persona deliberation over a fixed number of rounds.

    Public API:
        run(personas, context, backend, config) -> DeliberationTrace
    """

    def __init__(self):
        pass

    def run(
        self,
        personas: list[PersonaSpec],
        context: MatchingContext,
        backend: BaseBackend,
        config: SimulatorConfig,
        side: SideType,
    ) -> DeliberationTrace:
        """
        Execute the full deliberation process.

        Returns a DeliberationTrace containing initial opinions,
        per-round updates, final opinions, and stop reason.
        """
        config.apply_seed()

        # Determine which backend to use for summarization
        if isinstance(backend, RuleBasedBackend):
            summarizer = backend
        else:
            # TODO: wire up LLM summarizer
            mock_config = SimulatorConfig(backend_type="mock", random_seed=config.random_seed)
            summarizer = RuleBasedBackend(mock_config)

        # ---- Round 0: initial opinions ----
        initial_opinions: dict[str, PersonaOpinion] = {}
        for p in personas:
            opinion = judge_persona_opinion(p, context, backend, config, round_idx=0)
            initial_opinions[p.persona_id] = opinion

        current_opinions = dict(initial_opinions)
        per_round_updates: list[RoundUpdate] = []

        # ---- Check early stopping condition immediately after round 0 ----
        if self._check_early_stop(current_opinions, config):
            stop_reason = "early_stop_majority_after_round_0"
            return DeliberationTrace(
                initial_opinions=initial_opinions,
                per_round_updates=per_round_updates,
                final_opinions=dict(current_opinions),
                stop_reason=stop_reason,
            )

        # ---- Deliberation rounds ----
        for round_idx in range(1, config.max_deliberation_rounds + 1):
            # Leader summarizes the state
            summary = summarizer.summarize_discussion(
                list(current_opinions.values()), round_idx
            )

            # Each persona revises their opinion given the summary + others
            updated: dict[str, PersonaOpinion] = {}
            for p in personas:
                other_opinions = [
                    o for pid, o in current_opinions.items() if pid != p.persona_id
                ]
                revised = summarizer.revise_opinion(
                    current_opinions[p.persona_id],
                    other_opinions,
                    summary,
                    round_idx,
                )
                updated[p.persona_id] = revised

            current_opinions = updated

            per_round_updates.append(RoundUpdate(
                round_idx=round_idx,
                speaker_persona_id=personas[config.leader_persona_idx].persona_id,
                summary_text=summary,
                updated_opinions=dict(current_opinions),
            ))

            if self._check_early_stop(current_opinions, config):
                stop_reason = f"early_stop_majority_after_round_{round_idx}"
                return DeliberationTrace(
                    initial_opinions=initial_opinions,
                    per_round_updates=per_round_updates,
                    final_opinions=dict(current_opinions),
                    stop_reason=stop_reason,
                )

        stop_reason = f"max_rounds_reached_{config.max_deliberation_rounds}"
        return DeliberationTrace(
            initial_opinions=initial_opinions,
            per_round_updates=per_round_updates,
            final_opinions=dict(current_opinions),
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Early stopping check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_early_stop(
        opinions: dict[str, PersonaOpinion],
        config: SimulatorConfig,
    ) -> bool:
        """
        Stop early if the majority action (by expected probability mass)
        exceeds the early_stop_threshold.
        """
        if not opinions:
            return True

        # Aggregate action probabilities across personas
        total_accept = sum(o.action_probs.get("accept", 0.0) for o in opinions.values())
        total_reject = sum(o.action_probs.get("reject", 0.0) for o in opinions.values())
        total_skip = sum(o.action_probs.get("skip", 0.0) for o in opinions.values())
        n = len(opinions)

        avg_accept = total_accept / n
        avg_reject = total_reject / n
        avg_skip = total_skip / n

        if avg_accept >= config.early_stop_threshold:
            return True
        if avg_reject >= config.early_stop_threshold:
            return True
        return False
