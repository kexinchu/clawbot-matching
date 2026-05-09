"""
simulator/persona_judge.py
==========================
Persona-level judgment: given a persona and a matching context,
produce a PersonaOpinion.

Default: MockPersonaJudge (rule-based, no LLM needed).
LLMJudge is a TODO stub for the real LLM path.
"""
from __future__ import annotations

from simulator.config import SimulatorConfig
from simulator.llm_backend import BaseBackend
from simulator.mock_backend import RuleBasedBackend
from simulator.types import MatchingContext, PersonaOpinion, PersonaSpec
from simulator import prompts as prompt_lib


def judge_persona_opinion(
    persona: PersonaSpec,
    context: MatchingContext,
    backend: BaseBackend,
    config: SimulatorConfig,
    round_idx: int = 0,
) -> PersonaOpinion:
    """
    Unified entry point for persona-level judgment.

    Dispatches to rule-based or LLM-based judge based on backend type.
    """
    if isinstance(backend, RuleBasedBackend):
        return backend.generate_opinion(persona, context, round_idx)

    # LLM-based path: delegate to backend.generate_opinion if available
    if hasattr(backend, "generate_opinion"):
        return backend.generate_opinion(persona, context, round_idx)

    # Legacy prompt-based path (for stub backends that don't implement generate_opinion)
    return _llm_judge(persona, context, backend, config, round_idx)


def _llm_judge(
    persona: PersonaSpec,
    context: MatchingContext,
    backend: BaseBackend,
    config: SimulatorConfig,
    round_idx: int,
) -> PersonaOpinion:
    """
    LLM-based persona judgment (stub).

    TODO: wire up once real backends are implemented.
    Falls back to rule-based so the system still runs.
    """
    # Attempt to use structured generation
    sys_prompt = prompt_lib.persona_opinion_system_prompt(
        persona.persona_name,
        persona.short_instruction,
    )
    user_prompt = prompt_lib.persona_opinion_user_prompt(
        _context_to_text(context),
    )

    import json, re
    try:
        raw = backend.generate(sys_prompt + "\n\n" + user_prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            return PersonaOpinion(
                persona_id=persona.persona_id,
                persona_role=persona.persona_role,
                utility_score=float(obj.get("utility_score", 0.0)),
                action_probs={
                    k: float(v)
                    for k, v in obj.get("action_probs", {}).items()
                },
                confidence=float(obj.get("confidence", 0.5)),
                rationale=str(obj.get("rationale", "")),
                extracted_concerns=list(obj.get("extracted_concerns", [])),
            )
    except Exception:
        pass

    # Fallback to rule-based so we always produce a valid opinion
    mock = RuleBasedBackend(config)
    return mock.generate_opinion(persona, context, round_idx)


def _context_to_text(ctx: MatchingContext) -> str:
    return (
        f"Requester: {ctx.requester.user_id} | role={ctx.requester.role}\n"
        f"Candidate: {ctx.candidate.user_id} | role={ctx.candidate.role}\n"
        f"Task: {ctx.task.title} | {ctx.task.description}\n"
        f"Required skills: {ctx.task.required_skills}\n"
        f"Candidate card: {ctx.card.summary}\n"
        f"Strengths: {', '.join(ctx.card.highlighted_strengths)}\n"
        f"Risks: {', '.join(ctx.card.highlighted_risks)}"
    )


class MockPersonaJudge:
    """
    Wrapper class that exposes the rule-based judge.
    Use this directly when you want a class-based interface.
    """

    def __init__(self, config: SimulatorConfig):
        self.mock = RuleBasedBackend(config)

    def judge(
        self,
        persona: PersonaSpec,
        context: MatchingContext,
        round_idx: int = 0,
    ) -> PersonaOpinion:
        return self.mock.generate_opinion(persona, context, round_idx)
