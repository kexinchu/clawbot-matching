"""
simulator/config.py
===================
SimulatorConfig dataclass — all tunable hyper-parameters in one place.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Literal


@dataclasses.dataclass
class SimulatorConfig:
    # --- Persona selection ---
    num_requester_personas: int = 4
    num_candidate_personas: int = 4
    persona_selection_seed: int = 42

    # --- Deliberation ---
    max_deliberation_rounds: int = 3
    early_stop_threshold: float = 0.70   # stop early if majority prob > this
    leader_persona_idx: int = 0          # which persona speaks first each round

    # --- Importance ranking ---
    top_k_personas: int = 3
    importance_temperature: float = 1.5  # softmax temperature

    # --- Decision thresholds (probabilistic mode by default) ---
    decision_mode: Literal["probabilistic", "threshold"] = "probabilistic"
    accept_utility_threshold: float = 0.3
    reject_utility_threshold: float = -0.3

    # --- Outcome simulation heuristics ---
    base_agreement_prob: float = 0.70
    base_completion_prob: float = 0.80
    conflict_penalty: float = 0.15       # subtracted from agreement per major conflict

    # --- Reward weights ---
    reward_feedback_weight: float = 0.40
    reward_efficiency_weight: float = 0.20
    reward_quality_weight: float = 0.40

    # --- Backend ---
    backend_type: Literal["mock", "openai", "anthropic", "vllm"] = "mock"
    # For real backends:
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    vllm_base_url: str | None = None
    model_name: str = "gpt-4o-mini"

    # --- Misc ---
    random_seed: int = 42
    trace_verbose: bool = True

    def apply_seed(self) -> None:
        """Set Python random seed for reproducible runs."""
        random.seed(self.random_seed)
        try:
            import numpy as np
            np.random.seed(self.random_seed)
        except ImportError:
            pass


_DEFAULT = SimulatorConfig()


def default_config() -> SimulatorConfig:
    return dataclasses.replace(_DEFAULT)
