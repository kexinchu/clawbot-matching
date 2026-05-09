"""
simulator/__init__.py
====================
Public API for the bilateral matching simulator.

Quick usage:
    from simulator import BilateralSimulator, RewardResult
    from simulator.config import SimulatorConfig
    from simulator.mock_backend import RuleBasedBackend
    from simulator.types import MatchingContext, UserProfile, TaskSpec, CandidateCard

    config = SimulatorConfig(backend_type="mock", random_seed=42)
    backend = RuleBasedBackend(config)
    sim = BilateralSimulator(backend, config)
    result = sim.run(context)
"""
from __future__ import annotations

# ── Types ──────────────────────────────────────────────────────────────────
from simulator.types import (
    Action,
    BilateralDecisionResult,
    CandidateCard,
    DecisionResult,
    DeliberationTrace,
    JointAction,
    MatchingContext,
    OutcomeResult,
    PersonaImportance,
    PersonaOpinion,
    PersonaSpec,
    RewardResult,
    RoundUpdate,
    SideType,
    TaskSpec,
    UserProfile,
)

# ── Config ─────────────────────────────────────────────────────────────────
from simulator.config import SimulatorConfig, default_config

# ── Backends ───────────────────────────────────────────────────────────────
from simulator.llm_backend import (
    AnthropicBackend,
    BaseBackend,
    OpenAIBackend,
    vLLMBackend,
)
from simulator.mock_backend import RuleBasedBackend


def create_backend(config: SimulatorConfig) -> BaseBackend:
    """
    Factory that instantiates the right backend from a SimulatorConfig.

    Usage:
        config = SimulatorConfig(backend_type="openai", model_name="...")
        backend = create_backend(config)
    """
    bt = config.backend_type
    if bt == "openai":
        from simulator.llm_backend import OpenAIBackend
        return OpenAIBackend(model_name=config.model_name)
    elif bt == "anthropic":
        from simulator.llm_backend import AnthropicBackend
        return AnthropicBackend(api_key=config.anthropic_api_key)
    elif bt == "vllm":
        from simulator.llm_backend import vLLMBackend
        return vLLMBackend(base_url=config.vllm_base_url or "http://localhost:8000")
    else:
        # "mock" and any other unrecognized value
        return RuleBasedBackend(config)

# ── Core engines ────────────────────────────────────────────────────────────
from simulator.bilateral_simulator import BilateralSimulator
from simulator.deliberation import DeliberationEngine
from simulator.importance_ranker import (
    rank_persona_importance,
    select_top_k,
)
from simulator.decision_model import SideDecisionEngine, action_from_probs
from simulator.outcome_simulator import OutcomeSimulator

# ── Reward ──────────────────────────────────────────────────────────────────
from simulator.reward import compute_reward, RewardResult

# ── Persona ─────────────────────────────────────────────────────────────────
from simulator.persona_generator import (
    CandidatePersonaGenerator,
    PersonaGenerator,
    RequesterPersonaGenerator,
)
from simulator.persona_judge import (
    MockPersonaJudge,
    judge_persona_opinion,
)

# ── Utilities ───────────────────────────────────────────────────────────────
from simulator.utils import TraceLogger

__all__ = [
    # Types
    "Action",
    "BilateralDecisionResult",
    "CandidateCard",
    "DecisionResult",
    "DeliberationTrace",
    "JointAction",
    "MatchingContext",
    "OutcomeResult",
    "PersonaImportance",
    "PersonaOpinion",
    "PersonaSpec",
    "RewardResult",
    "RoundUpdate",
    "SideType",
    "TaskSpec",
    "UserProfile",
    # Config
    "SimulatorConfig",
    "default_config",
    # Backends
    "BaseBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "vLLMBackend",
    "RuleBasedBackend",
    "create_backend",
    # Engines
    "BilateralSimulator",
    "DeliberationEngine",
    "SideDecisionEngine",
    "OutcomeSimulator",
    "rank_persona_importance",
    "select_top_k",
    "action_from_probs",
    # Persona
    "PersonaGenerator",
    "RequesterPersonaGenerator",
    "CandidatePersonaGenerator",
    "MockPersonaJudge",
    "judge_persona_opinion",
    # Reward
    "compute_reward",
    # Utils
    "TraceLogger",
]
