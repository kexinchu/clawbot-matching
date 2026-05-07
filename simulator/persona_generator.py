"""
simulator/persona_generator.py
===============================
Persona generators for requester side and candidate side.

By default, uses RuleBasedBackend.generate_personas() which needs no LLM.
The LLM-based path is wired but gated behind a backend type check.
"""
from __future__ import annotations

from simulator.config import SimulatorConfig
from simulator.llm_backend import BaseBackend
from simulator.mock_backend import RuleBasedBackend
from simulator.types import MatchingContext, PersonaSpec, SideType
from simulator import prompts as prompt_lib


class PersonaGenerator:
    """
    Generates persona lists for one side of the bilateral match.

    Usage:
        gen = PersonaGenerator(backend, config, SideType.REQUESTER)
        personas = gen.generate(context)
    """

    def __init__(
        self,
        backend: BaseBackend,
        config: SimulatorConfig,
        side: SideType,
    ):
        self.backend = backend
        self.config = config
        self.side = side

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, context: MatchingContext) -> list[PersonaSpec]:
        """
        Generate a list of PersonaSpec for the configured side.

        Dispatches to LLM-based or rule-based generation based on backend type.
        """
        num = (
            self.config.num_requester_personas
            if self.side == SideType.REQUESTER
            else self.config.num_candidate_personas
        )

        if isinstance(self.backend, RuleBasedBackend):
            return self.backend.generate_personas(self.side, context, num)

        # LLM-based path (TODO: wire up properly once backend is implemented)
        return self._llm_generate(context, num)

    # ------------------------------------------------------------------
    # LLM-based path (stub)
    # ------------------------------------------------------------------

    def _llm_generate(
        self,
        context: MatchingContext,
        num_personas: int,
    ) -> list[PersonaSpec]:
        """Use LLM to select and instantiate personas."""
        available = self._available_persona_templates()
        context_summary = self._summarize_context(context)

        user_prompt = prompt_lib.persona_selection_user_prompt(
            side=self.side.value,
            num_personas=num_personas,
            available_personas=available,
            context_summary=context_summary,
        )

        raw = self.backend.generate(
            prompt_lib.persona_selection_system_prompt() + "\n\n" + user_prompt
        )

        # Parse JSON list of persona IDs from raw response
        import json, re
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            # Fallback to rule-based
            return self.backend.generate_personas(self.side, context, num_personas)

        try:
            selected_ids = json.loads(match.group())
        except json.JSONDecodeError:
            return self.backend.generate_personas(self.side, context, num_personas)

        templates = {
            p.persona_id: p
            for p in (
                self._requester_templates()
                if self.side == SideType.REQUESTER
                else self._candidate_templates()
            )
        }
        selected = [templates[sid] for sid in selected_ids if sid in templates]
        # Ensure we have at least num_personas
        fallback = self.backend.generate_personas(self.side, context, num_personas)
        for p in fallback:
            if p not in selected:
                selected.append(p)
        return selected[:num_personas]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarize_context(self, ctx: MatchingContext) -> str:
        if self.side == SideType.REQUESTER:
            return (
                f"Task: {ctx.task.title}\n"
                f"Task description: {ctx.task.description}\n"
                f"Required skills: {ctx.task.required_skills}\n"
                f"Candidate: {ctx.candidate.user_id} ({ctx.candidate.role})\n"
                f"Candidate capabilities: {ctx.candidate.capabilities}\n"
                f"Candidate card: {ctx.card.summary}"
            )
        else:
            return (
                f"Task: {ctx.task.title}\n"
                f"Task description: {ctx.task.description}\n"
                f"Required skills: {ctx.task.required_skills}\n"
                f"Requester: {ctx.requester.user_id} ({ctx.requester.role})\n"
                f"Requester needs: {ctx.requester.needs}\n"
                f"Task metadata: {ctx.task.metadata}"
            )

    def _available_persona_templates(self) -> list[str]:
        return [p.persona_id for p in (
            self._requester_templates()
            if self.side == SideType.REQUESTER
            else self._candidate_templates()
        )]

    @staticmethod
    def _requester_templates() -> list[PersonaSpec]:
        # Import from mock_backend to avoid duplication
        from simulator.mock_backend import RuleBasedBackend
        RuleBasedBackend._build_templates()
        return RuleBasedBackend.REQUESTER_PERSONA_TEMPLATES

    @staticmethod
    def _candidate_templates() -> list[PersonaSpec]:
        from simulator.mock_backend import RuleBasedBackend
        RuleBasedBackend._build_templates()
        return RuleBasedBackend.CANDIDATE_PERSONA_TEMPLATES


class RequesterPersonaGenerator(PersonaGenerator):
    def __init__(self, backend: BaseBackend, config: SimulatorConfig):
        super().__init__(backend, config, SideType.REQUESTER)


class CandidatePersonaGenerator(PersonaGenerator):
    def __init__(self, backend: BaseBackend, config: SimulatorConfig):
        super().__init__(backend, config, SideType.CANDIDATE)
