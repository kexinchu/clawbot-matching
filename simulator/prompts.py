"""
simulator/prompts.py
====================
Prompt templates used by the persona generator (for LLM-based generation).
These are only consumed when backend_type != 'mock'.
"""
from __future__ import annotations


def persona_selection_system_prompt() -> str:
    return (
        "You are an expert at selecting the most relevant evaluative personas "
        "for a bilateral matching decision. Given a matching context, you should "
        "choose 3-5 personas that are most informative for each side. "
        "Prefer diversity: pick personas that evaluate different dimensions."
    )


def persona_selection_user_prompt(
    side: str,
    num_personas: int,
    available_personas: list[str],
    context_summary: str,
) -> str:
    return (
        f"Side: {side}\n"
        f"Select {num_personas} personas from the following pool:\n"
        + "\n".join(f"  - {p}" for p in available_personas)
        + f"\n\nMatching context:\n{context_summary}\n\n"
        f"Return a JSON list of persona IDs."
    )


def persona_opinion_system_prompt(persona_name: str, persona_instruction: str) -> str:
    return (
        f"You are the persona: {persona_name}.\n"
        f"Your evaluation focus: {persona_instruction}\n"
        "Given the matching context, output your opinion as a JSON object with fields:\n"
        "  utility_score: float in [-1, 1] (positive = favor accept, negative = favor reject)\n"
        "  action_probs: {{'accept': float, 'skip': float, 'reject': float}}\n"
        "  confidence: float in [0, 1]\n"
        "  rationale: str (2-3 sentences explaining your reasoning)\n"
        "  extracted_concerns: list[str] (1-3 specific concerns, empty if none)"
    )


def persona_opinion_user_prompt(context_summary: str) -> str:
    return f"Matching context:\n{context_summary}"


def deliberation_summary_system_prompt() -> str:
    return (
        "You are a synthesis agent leading a multi-persona deliberation. "
        "Your job is to summarize disagreements and identify the key conflict points "
        "that personas should address in the next round. "
        "Be concise and point out the most consequential conflicts."
    )


def deliberation_summary_user_prompt(
    round_idx: int,
    opinions: list[dict],
) -> str:
    lines = [f"Round {round_idx} opinions:"]
    for o in opinions:
        lines.append(
            f"  - {o['persona_name']} (role={o['persona_role']}): "
            f"utility={o['utility_score']:.2f}, "
            f"confidence={o['confidence']:.2f}, "
            f"concerns={o['extracted_concerns']}"
        )
    lines.append("\nProvide a 2-3 sentence synthesis highlighting major disagreements.")
    return "\n".join(lines)
