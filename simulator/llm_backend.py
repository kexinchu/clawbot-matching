"""
simulator/llm_backend.py
=======================
Abstract LLM backend interface and real-backend stubs.

BaseBackend defines the contract that both mock and real backends must satisfy.
Real backends (OpenAI, Anthropic, vLLM) are TODO stubs that raise
NotImplementedError — the interface is ready but no API key is required to run.
"""
from __future__ import annotations

import random

from abc import ABC, abstractmethod
from typing import Any

from simulator.types import (
    MatchingContext,
    PersonaOpinion,
    PersonaSpec,
    SideType,
)


# ---------------------------------------------------------------------------
# Shared persona template pool (used by OpenAIBackend.generate_personas)
# ---------------------------------------------------------------------------

_REQUERSTER_TEMPLATES = [
    PersonaSpec(
        persona_id="req_skill_match",
        persona_name="Skill Match Analyst",
        persona_role="skill_match_evaluator",
        focus_dimension="skill coverage of the task",
        short_instruction="Evaluate how well the candidate's skills cover the task requirements.",
    ),
    PersonaSpec(
        persona_id="req_goal_coverage",
        persona_name="Goal Alignment Analyst",
        persona_role="goal_coverage_evaluator",
        focus_dimension="goal alignment between candidate and task",
        short_instruction="Evaluate how well the candidate's goals align with the task objectives.",
    ),
    PersonaSpec(
        persona_id="req_collab_style",
        persona_name="Collaboration Style Analyst",
        persona_role="collaboration_style_evaluator",
        focus_dimension="collaboration style compatibility",
        short_instruction="Assess compatibility of collaboration styles between requester and candidate.",
    ),
    PersonaSpec(
        persona_id="req_time_risk",
        persona_name="Timeline Risk Analyst",
        persona_role="time_risk_evaluator",
        focus_dimension="timeline and risk factors",
        short_instruction="Evaluate timeline feasibility and risk factors.",
    ),
    PersonaSpec(
        persona_id="req_incentive",
        persona_name="Incentive Alignment Analyst",
        persona_role="incentive_evaluator",
        focus_dimension="incentive and motivation alignment",
        short_instruction="Assess alignment of incentives and motivations.",
    ),
    PersonaSpec(
        persona_id="req_trust_safety",
        persona_name="Trust & Safety Analyst",
        persona_role="trust_safety_evaluator",
        focus_dimension="trust and safety considerations",
        short_instruction="Evaluate trust signals and safety considerations.",
    ),
]

_CANDIDATE_TEMPLATES = [
    PersonaSpec(
        persona_id="cand_capability_fit",
        persona_name="Capability Fit Analyst",
        persona_role="capability_fit_evaluator",
        focus_dimension="how well the task matches the candidate's capabilities",
        short_instruction="Evaluate how well this task matches your capabilities and expertise.",
    ),
    PersonaSpec(
        persona_id="cand_task_interest",
        persona_name="Task Interest Analyst",
        persona_role="task_interest_evaluator",
        focus_dimension="interest and engagement level",
        short_instruction="Assess your genuine interest and engagement level in this task.",
    ),
    PersonaSpec(
        persona_id="cand_opportunity_cost",
        persona_name="Opportunity Cost Analyst",
        persona_role="opportunity_cost_evaluator",
        focus_dimension="opportunity cost considerations",
        short_instruction="Evaluate the opportunity cost of accepting vs. passing on this task.",
    ),
    PersonaSpec(
        persona_id="cand_workload",
        persona_name="Workload Balance Analyst",
        persona_role="workload_evaluator",
        focus_dimension="workload and capacity planning",
        short_instruction="Assess whether taking this task fits your current workload.",
    ),
    PersonaSpec(
        persona_id="cand_reciprocity",
        persona_name="Reciprocity Benefit Analyst",
        persona_role="reciprocity_benefit_evaluator",
        focus_dimension="mutual benefit and reciprocity",
        short_instruction="Evaluate the reciprocal benefits of collaborating with this requester.",
    ),
    PersonaSpec(
        persona_id="cand_goal_coverage",
        persona_name="Goal Coverage Analyst",
        persona_role="goal_coverage_evaluator",
        focus_dimension="goal alignment with the task",
        short_instruction="Assess how well this task aligns with your career goals.",
    ),
]


class BaseBackend(ABC):
    """
    Unified interface for all LLM backends.

    Subclasses must implement:
      - generate(prompt) -> str
      - structured_generate(prompt, schema) -> dict | str
      - generate_personas(side, context, num_personas) -> list[PersonaSpec]
      - generate_opinion(persona, context, round_idx) -> PersonaOpinion
      - summarize_discussion(opinions, round_idx) -> str
      - revise_opinion(opinion, others_opinions, summary, round_idx) -> PersonaOpinion
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Free-form text generation from a single prompt string."""
        ...

    def structured_generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any] | str:
        """
        Generation with an optional output schema hint.
        Default implementation just calls generate() and parses as JSON.
        Subclasses can override for structured-output API support.
        """
        raw = self.generate(prompt, **kwargs)
        if schema is None:
            return raw
        # Best-effort JSON parse
        import json, re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return raw

    @abstractmethod
    def generate_personas(
        self,
        side: SideType,
        context: MatchingContext,
        num_personas: int | None = None,
    ) -> list[PersonaSpec]:
        """Generate a list of persona specs for the given side."""
        ...

    @abstractmethod
    def generate_opinion(
        self,
        persona: PersonaSpec,
        context: MatchingContext,
        round_idx: int = 0,
    ) -> PersonaOpinion:
        """Generate a persona's opinion about the match."""
        ...

    @abstractmethod
    def summarize_discussion(
        self,
        opinions: list[PersonaOpinion],
        round_idx: int,
    ) -> str:
        """Summarize the deliberation state at round_idx."""
        ...

    @abstractmethod
    def revise_opinion(
        self,
        opinion: PersonaOpinion,
        others_opinions: list[PersonaOpinion],
        summary: str,
        round_idx: int,
    ) -> PersonaOpinion:
        """Revise a persona's opinion based on group deliberation."""
        ...


class OpenAIBackend(BaseBackend):
    """
    OpenAI-compatible API backend via OpenRouter.

    OpenRouter uses the standard OpenAI chat completions format with:
      - base URL: https://openrouter.ai/api/v1
      - Authorization: Bearer <OPENROUTER_API_KEY>
      - Extra headers: HTTP-Referer, X-Title
    """

    def __init__(self,
        model_name: str = "deepseek/deepseek-chat-v3-0324",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        persona_seed: int = 42,
    ):
        self.model_name = model_name
        self.api_key = api_key or _load_key("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.persona_seed = persona_seed

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai>=1.0.0 is required. Run: pip install openai")

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Support model override via kwargs
        model = kwargs.pop("model", self.model_name)
        temperature = kwargs.pop("temperature", 0.7)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content

    def structured_generate(self, prompt: str, schema: dict | None = None, **kwargs):
        raw = self.generate(prompt, **kwargs)
        if schema is None:
            return raw
        import json, re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return raw

    # ------------------------------------------------------------------------
    # Persona generation
    # ------------------------------------------------------------------------

    def generate_personas(
        self,
        side: SideType,
        context: MatchingContext,
        num_personas: int | None = None,
    ) -> list[PersonaSpec]:
        """Sample persona specs from a fixed template pool deterministically."""
        templates = _REQUERSTER_TEMPLATES if side == SideType.REQUESTER else _CANDIDATE_TEMPLATES
        num = num_personas or (4 if side == SideType.REQUESTER else 4)
        rng = random.Random(self.persona_seed)
        shuffled = rng.sample(templates, k=len(templates))
        return shuffled[: min(num, len(shuffled))]

    # ------------------------------------------------------------------------
    # Persona opinion generation
    # ------------------------------------------------------------------------

    def generate_opinion(
        self,
        persona: PersonaSpec,
        context: MatchingContext,
        round_idx: int = 0,
    ) -> PersonaOpinion:
        """Use LLM to generate a persona's opinion as structured JSON."""
        prompt = _build_opinion_prompt(persona, context, round_idx)
        schema = {
            "type": "object",
            "properties": {
                "utility_score": {
                    "type": "number",
                    "description": "Score from -1.0 (strong reject) to 1.0 (strong accept)",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "accept_prob": {
                    "type": "number",
                    "description": "Probability of accepting (0 to 1)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "skip_prob": {
                    "type": "number",
                    "description": "Probability of skipping (0 to 1)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "reject_prob": {
                    "type": "number",
                    "description": "Probability of rejecting (0 to 1)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in this assessment (0 to 1)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief reasoning for the opinion",
                },
                "extracted_concerns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of specific concerns raised",
                },
            },
            "required": ["utility_score", "accept_prob", "skip_prob", "reject_prob", "confidence", "rationale", "extracted_concerns"],
        }
        raw = self.structured_generate(prompt, schema=schema)

        if isinstance(raw, str):
            # Fallback on parse failure
            return _fallback_opinion(persona)

        accept_prob = max(0.0, min(1.0, float(raw.get("accept_prob", 0.5))))
        skip_prob = max(0.0, min(1.0, float(raw.get("skip_prob", 0.3))))
        reject_prob = max(0.0, min(1.0, float(raw.get("reject_prob", 0.2))))
        total = accept_prob + skip_prob + reject_prob
        if total > 0:
            accept_prob /= total
            skip_prob /= total
            reject_prob /= total

        return PersonaOpinion(
            persona_id=persona.persona_id,
            persona_role=persona.persona_role,
            utility_score=max(-1.0, min(1.0, float(raw.get("utility_score", 0.0)))),
            action_probs={
                "accept": accept_prob,
                "skip": skip_prob,
                "reject": reject_prob,
            },
            confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.5)))),
            rationale=str(raw.get("rationale", "")),
            extracted_concerns=list(raw.get("extracted_concerns", [])),
        )

    # ------------------------------------------------------------------------
    # Deliberation helpers
    # ------------------------------------------------------------------------

    def summarize_discussion(
        self,
        opinions: list[PersonaOpinion],
        round_idx: int,
    ) -> str:
        """Summarize deliberation opinions using the LLM."""
        if not opinions:
            return "No opinions expressed yet."

        prompt = _build_summary_prompt(opinions, round_idx)
        return self.generate(prompt, temperature=0.4)

    def revise_opinion(
        self,
        opinion: PersonaOpinion,
        others_opinions: list[PersonaOpinion],
        summary: str,
        round_idx: int,
    ) -> PersonaOpinion:
        """Revise a persona's opinion using the LLM."""
        prompt = _build_revision_prompt(opinion, others_opinions, summary, round_idx)
        schema = {
            "type": "object",
            "properties": {
                "utility_score": {
                    "type": "number",
                    "description": "Revised score from -1.0 to 1.0",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "accept_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "skip_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reject_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale": {"type": "string"},
                "extracted_concerns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["utility_score", "accept_prob", "skip_prob", "reject_prob", "confidence", "rationale", "extracted_concerns"],
        }
        raw = self.structured_generate(prompt, schema=schema)

        if isinstance(raw, str):
            return _fallback_revised_opinion(opinion, others_opinions, round_idx)

        accept_prob = max(0.0, min(1.0, float(raw.get("accept_prob", 0.5))))
        skip_prob = max(0.0, min(1.0, float(raw.get("skip_prob", 0.3))))
        reject_prob = max(0.0, min(1.0, float(raw.get("reject_prob", 0.2))))
        total = accept_prob + skip_prob + reject_prob
        if total > 0:
            accept_prob /= total
            skip_prob /= total
            reject_prob /= total

        return PersonaOpinion(
            persona_id=opinion.persona_id,
            persona_role=opinion.persona_role,
            utility_score=max(-1.0, min(1.0, float(raw.get("utility_score", opinion.utility_score)))),
            action_probs={
                "accept": accept_prob,
                "skip": skip_prob,
                "reject": reject_prob,
            },
            confidence=max(0.0, min(1.0, float(raw.get("confidence", opinion.confidence)))),
            rationale=f"[Revised r{round_idx}] {raw.get('rationale', opinion.rationale)}",
            extracted_concerns=list(raw.get("extracted_concerns", opinion.extracted_concerns)),
        )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_opinion_prompt(persona: PersonaSpec, context: MatchingContext, round_idx: int) -> str:
    req = context.requester
    cand = context.candidate
    task = context.task
    card = context.card

    return f"""You are evaluating a bilateral match between a requester and a candidate.

## Your Persona
- ID: {persona.persona_id}
- Name: {persona.persona_name}
- Role: {persona.persona_role}
- Focus: {persona.focus_dimension}
- Instruction: {persona.short_instruction}

## Context
### Requester ({req.user_id})
- Role: {req.role}
- Capabilities: {req.capabilities}
- Needs: {req.needs}
- Preferences: {req.preferences}
- History: {req.history_summary}

### Candidate ({cand.user_id})
- Role: {cand.role}
- Capabilities: {cand.capabilities}
- Needs: {cand.needs}
- Preferences: {cand.preferences}
- Constraints: {cand.constraints}
- History: {cand.history_summary}

### Task ({task.task_id})
- Title: {task.title}
- Description: {task.description}
- Required skills: {task.required_skills}
- Metadata: {task.metadata}

### Candidate Card Summary
{card.summary}
Highlighted strengths: {card.highlighted_strengths}
Highlighted risks: {card.highlighted_risks}
Explanation: {card.explanation}

### Match History
{context.history}

## Your Task
Think deeply from the perspective of "{persona.persona_name}". Consider:
- {persona.short_instruction}
- Any risks or red flags
- Any positive signals

Provide your assessment as a JSON object with the fields:
- utility_score: float (-1.0 = strong reject, 0.0 = neutral, 1.0 = strong accept)
- accept_prob, skip_prob, reject_prob: probabilities summing to ~1.0
- confidence: how certain you are (0.0-1.0)
- rationale: brief reasoning (1-3 sentences)
- extracted_concerns: list of specific concerns (e.g. "Skill gaps in X", "Timeline too tight")
"""


def _build_summary_prompt(opinions: list[PersonaOpinion], round_idx: int) -> str:
    opinion_lines = []
    for op in opinions:
        concerns_str = ", ".join(op.extracted_concerns) if op.extracted_concerns else "none"
        opinion_lines.append(
            f"- {op.persona_id} ({op.persona_role}): "
            f"utility={op.utility_score:.2f}, confidence={op.confidence:.2f}, "
            f"concerns=[{concerns_str}]"
        )

    return f"""You are summarizing a multi-persona deliberation round.

## Round {round_idx} Opinions
{chr(10).join(opinion_lines)}

## Task
Write a concise 2-3 sentence summary of the deliberation so far, highlighting:
- The overall倾向 (leaning toward accept, skip, or reject)
- Key concerns that emerged
- Any points of agreement or disagreement

Respond with only the summary text (no JSON).
"""


def _build_revision_prompt(
    opinion: PersonaOpinion,
    others_opinions: list[PersonaOpinion],
    summary: str,
    round_idx: int,
) -> str:
    others_lines = []
    for op in others_opinions:
        concerns_str = ", ".join(op.extracted_concerns) if op.extracted_concerns else "none"
        others_lines.append(
            f"- {op.persona_id}: utility={op.utility_score:.2f}, confidence={op.confidence:.2f}, concerns=[{concerns_str}]"
        )

    return f"""You are revising your opinion after a group deliberation.

## Your Original Opinion
- Persona: {opinion.persona_id} ({opinion.persona_role})
- Utility score: {opinion.utility_score:.3f}
- Confidence: {opinion.confidence:.2f}
- Concerns: {opinion.extracted_concerns}
- Rationale: {opinion.rationale}

## Other Personas' Opinions
{chr(10).join(others_lines)}

## Leader Summary
{summary}

## Task
Based on the group deliberation, revise your opinion. Consider:
- Did the group raise points you hadn't considered?
- Are there concerns that appeared in multiple opinions?
- Has your view shifted, or do you still hold your original position?

Provide a revised opinion as JSON with fields:
- utility_score: float (-1.0 to 1.0)
- accept_prob, skip_prob, reject_prob: probabilities summing to ~1.0
- confidence: float (0.0-1.0)
- rationale: brief revised reasoning
- extracted_concerns: list of concerns (including any newly raised ones)
"""


# ---------------------------------------------------------------------------
# Fallback helpers (when LLM parsing fails)
# ---------------------------------------------------------------------------

def _fallback_opinion(persona: PersonaSpec) -> PersonaOpinion:
    """Return a neutral fallback opinion when LLM parsing fails."""
    return PersonaOpinion(
        persona_id=persona.persona_id,
        persona_role=persona.persona_role,
        utility_score=0.0,
        action_probs={"accept": 0.4, "skip": 0.3, "reject": 0.3},
        confidence=0.4,
        rationale="LLM opinion generation failed; using neutral fallback.",
        extracted_concerns=[],
    )


def _fallback_revised_opinion(
    opinion: PersonaOpinion,
    others_opinions: list[PersonaOpinion],
    round_idx: int,
) -> PersonaOpinion:
    """Simple rule-based revision when LLM fails."""
    if not others_opinions:
        return opinion
    group_mean = sum(o.utility_score for o in others_opinions) / len(others_opinions)
    pull = 0.15
    new_score = opinion.utility_score + pull * (group_mean - opinion.utility_score)
    new_score = max(-1.0, min(1.0, new_score))
    alignment = abs(group_mean - opinion.utility_score)
    delta_conf = -0.1 * alignment if alignment > 0.2 else 0.05 * (1 - alignment)
    new_conf = max(0.1, min(0.95, opinion.confidence + delta_conf))
    new_probs = {
        "accept": max(0.0, min(1.0, opinion.action_probs.get("accept", 0.4) + 0.1 * (group_mean - opinion.utility_score))),
        "skip": opinion.action_probs.get("skip", 0.3),
        "reject": max(0.0, min(1.0, opinion.action_probs.get("reject", 0.3) - 0.1 * (group_mean - opinion.utility_score))),
    }
    total = sum(new_probs.values())
    if total > 0:
        for k in new_probs:
            new_probs[k] /= total
    all_concerns = []
    for o in others_opinions:
        all_concerns.extend(o.extracted_concerns)
    concern_counts: dict[str, int] = {}
    for c in all_concerns:
        key = c.lower()
        concern_counts[key] = concern_counts.get(key, 0) + 1
    new_concerns = list(opinion.extracted_concerns)
    for c, cnt in concern_counts.items():
        if cnt >= 2 and c not in [x.lower() for x in new_concerns]:
            new_concerns.append(c.title())
    return PersonaOpinion(
        persona_id=opinion.persona_id,
        persona_role=opinion.persona_role,
        utility_score=new_score,
        action_probs=new_probs,
        confidence=new_conf,
        rationale=f"[Revised r{round_idx}] {opinion.rationale} → group_mean={group_mean:.2f}",
        extracted_concerns=new_concerns,
    )


# ---------------------------------------------------------------------------
# Key loader
# ---------------------------------------------------------------------------

def _load_key(env_var: str) -> str:
    import os
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    # Fallback: scan ~/.bashrc for an export line of the form
    #   export VAR_NAME=...
    # This handles non-interactive invocations where bashrc isn't sourced.
    bashrc = os.path.expanduser("~/.bashrc")
    try:
        prefix = f"export {env_var}="
        for line in open(bashrc):
            if line.lstrip().startswith(prefix):
                _, _, raw_val = line.partition("=")
                val = raw_val.strip().strip("\"'")
                if val:
                    return val
    except OSError:
        pass
    raise RuntimeError(
        f"{env_var} is not set. "
        f"Please add 'export {env_var}=<your-key>' to ~/.bashrc "
        f"and run 'source ~/.bashrc'."
    )


# ---------------------------------------------------------------------------
# Stub backends
# ---------------------------------------------------------------------------

class AnthropicBackend(BaseBackend):
    """
    Anthropic API backend stub.

    TODO: implement with anthropic library.
    Requires: ANTHROPIC_API_KEY in environment or config.anthropic_api_key.
    """

    def __init__(self, model_name: str = "claude-3-5-haiku-20241022",
                 api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            "AnthropicBackend.generate() is not implemented yet. "
            "Set backend_type='mock' to run without an API key."
        )

    def generate_personas(self, side, context, num_personas=None):
        raise NotImplementedError("AnthropicBackend.generate_personas() is not implemented.")

    def generate_opinion(self, persona, context, round_idx=0):
        raise NotImplementedError("AnthropicBackend.generate_opinion() is not implemented.")

    def summarize_discussion(self, opinions, round_idx):
        raise NotImplementedError("AnthropicBackend.summarize_discussion() is not implemented.")

    def revise_opinion(self, opinion, others_opinions, summary, round_idx):
        raise NotImplementedError("AnthropicBackend.revise_opinion() is not implemented.")


class vLLMBackend(BaseBackend):
    """
    vLLM backend stub.

    TODO: implement with openai-python pointing to a vLLM server.
    Requires: config.vllm_base_url pointing to a running vLLM instance.
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 model_name: str = "meta-llama/Llama-3-8B-Instruct"):
        self.base_url = base_url
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            "vLLMBackend.generate() is not implemented yet. "
            "Set backend_type='mock' to run without an API key."
        )

    def generate_personas(self, side, context, num_personas=None):
        raise NotImplementedError("vLLMBackend.generate_personas() is not implemented.")

    def generate_opinion(self, persona, context, round_idx=0):
        raise NotImplementedError("vLLMBackend.generate_opinion() is not implemented.")

    def summarize_discussion(self, opinions, round_idx):
        raise NotImplementedError("vLLMBackend.summarize_discussion() is not implemented.")

    def revise_opinion(self, opinion, others_opinions, summary, round_idx):
        raise NotImplementedError("vLLMBackend.revise_opinion() is not implemented.")
