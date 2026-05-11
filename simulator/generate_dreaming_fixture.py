from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from simulator.config import SimulatorConfig
from simulator.mock_backend import RuleBasedBackend
from simulator.persona_generator import (
    CandidatePersonaGenerator,
    RequesterPersonaGenerator,
)
from simulator.types import CandidateCard, MatchingContext, TaskSpec, UserProfile


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "LLM_Dreaming" / "dreaming_fixture.json"


def _requester_record() -> dict:
    return {
        "user_id": "alice",
        "role": "requester",
        "capabilities": [
            {"description": "bayesian statistics", "mu": 0.3, "sigma": 0.2, "source": "explicit"},
            {"description": "python programming", "mu": 0.8, "sigma": 0.1, "source": "explicit"},
            {"description": "academic paper writing", "mu": 0.4, "sigma": 0.3, "source": "implicit"},
        ],
        "needs": [
            {"description": "bayesian statistics mentorship", "intensity": 0.8},
            {"description": "python programming", "intensity": 0.1},
            {"description": "academic paper writing", "intensity": 0.7},
        ],
        "soft_profile": {
            "availability": "30h/week",
            "timezone": "UTC-5 (EST)",
            "deadline_pressure": "tight - submission in 3 months",
            "collab_style": "sync-heavy, likes daily standups",
            "communication": "concise, action-oriented",
            "personality_notes": (
                "Startup CTO, fast-paced, direct communicator, values efficiency over perfection"
            ),
            "priorities": [
                "hit NeurIPS deadline",
                "solid Bayesian modeling",
                "clean reproducible code",
            ],
        },
    }


def _candidate_records() -> list[dict]:
    return [
        {
            "user_id": "bob",
            "role": "candidate",
            "capabilities": [
                {"description": "bayesian statistics", "mu": 0.9, "sigma": 0.1, "source": "explicit"},
                {"description": "python programming", "mu": 0.5, "sigma": 0.2, "source": "explicit"},
                {"description": "academic paper writing", "mu": 0.8, "sigma": 0.1, "source": "explicit"},
            ],
            "needs": [
                {"description": "bayesian statistics", "intensity": 0.2},
                {"description": "python programming", "intensity": 0.7},
                {"description": "academic paper writing", "intensity": 0.9},
            ],
            "soft_profile": {
                "availability": "20h/week",
                "timezone": "UTC-5 (EST)",
                "deadline_pressure": "moderate - also has coursework",
                "collab_style": "mixed, prefers structured weekly meetings",
                "communication": "detailed, likes to explain reasoning",
                "personality_notes": (
                    "Stats PhD student, methodical, thorough, sometimes slow but high quality output"
                ),
                "priorities": [
                    "first-author NeurIPS paper",
                    "learn industry data pipelines",
                    "build portfolio for job search",
                ],
            },
        },
        {
            "user_id": "carol",
            "role": "candidate",
            "capabilities": [
                {"description": "bayesian statistics", "mu": 0.6, "sigma": 0.4, "source": "meta"},
                {"description": "python programming", "mu": 0.7, "sigma": 0.35, "source": "meta"},
                {"description": "academic paper writing", "mu": 0.5, "sigma": 0.4, "source": "meta"},
            ],
            "needs": [
                {"description": "bayesian statistics", "intensity": 0.5},
                {"description": "python programming", "intensity": 0.3},
                {"description": "academic paper writing", "intensity": 0.8},
            ],
            "soft_profile": {
                "availability": "15h/week - also freelancing",
                "timezone": "UTC+1 (CET)",
                "deadline_pressure": "relaxed - no hard deadlines personally",
                "collab_style": "async-first, replies in batches",
                "communication": "visual, loves diagrams and notebooks",
                "personality_notes": (
                    "Creative ML researcher, lots of ideas, jumps between projects, inconsistent follow-through"
                ),
                "priorities": [
                    "explore novel Bayesian methods",
                    "add to publications list",
                    "flexible schedule",
                ],
            },
        },
        {
            "user_id": "dave",
            "role": "candidate",
            "capabilities": [
                {"description": "bayesian statistics", "mu": 0.85, "sigma": 0.15, "source": "explicit"},
                {"description": "python programming", "mu": 0.75, "sigma": 0.1, "source": "explicit"},
                {"description": "academic paper writing", "mu": 0.6, "sigma": 0.2, "source": "explicit"},
            ],
            "needs": [
                {"description": "bayesian statistics", "intensity": 0.3},
                {"description": "python programming", "intensity": 0.2},
                {"description": "academic paper writing", "intensity": 0.7},
            ],
            "soft_profile": {
                "availability": "25h/week",
                "timezone": "UTC+8 (SGT)",
                "deadline_pressure": "moderate",
                "collab_style": "async-first, very responsive on Slack",
                "communication": "concise, code-speaks-louder",
                "personality_notes": (
                    "Senior ML engineer at a Singapore startup, pragmatic, ships fast, prefers working code over theory"
                ),
                "priorities": [
                    "get a top-venue publication",
                    "transition to research role",
                    "learn academic writing conventions",
                ],
            },
        },
        {
            "user_id": "eve",
            "role": "candidate",
            "capabilities": [
                {"description": "bayesian statistics", "mu": 0.95, "sigma": 0.05, "source": "explicit"},
                {"description": "python programming", "mu": 0.9, "sigma": 0.05, "source": "explicit"},
                {"description": "academic paper writing", "mu": 0.85, "sigma": 0.1, "source": "explicit"},
            ],
            "needs": [
                {"description": "bayesian statistics", "intensity": 0.1},
                {"description": "python programming", "intensity": 0.1},
                {"description": "academic paper writing", "intensity": 0.3},
            ],
            "soft_profile": {
                "availability": "5h/week - leading 2 other projects",
                "timezone": "UTC-5 (EST)",
                "deadline_pressure": "very tight - own deadlines competing",
                "collab_style": "async only, slow to respond",
                "communication": "terse, bullet points",
                "personality_notes": (
                    "Tenured professor, brilliant but stretched thin, delegates heavily, hard to get time with"
                ),
                "priorities": [
                    "add another publication to lab output",
                    "mentor junior researchers",
                    "minimal time commitment",
                ],
            },
        },
        {
            "user_id": "frank",
            "role": "candidate",
            "capabilities": [
                {"description": "bayesian statistics", "mu": 0.4, "sigma": 0.3, "source": "implicit"},
                {"description": "python programming", "mu": 0.6, "sigma": 0.25, "source": "explicit"},
                {"description": "academic paper writing", "mu": 0.3, "sigma": 0.35, "source": "implicit"},
            ],
            "needs": [
                {"description": "bayesian statistics", "intensity": 0.9},
                {"description": "python programming", "intensity": 0.5},
                {"description": "academic paper writing", "intensity": 0.9},
            ],
            "soft_profile": {
                "availability": "40h/week - dedicated to this",
                "timezone": "UTC-5 (EST)",
                "deadline_pressure": "none - gap year, fully flexible",
                "collab_style": "sync-heavy, loves pair programming",
                "communication": "detailed, asks lots of questions",
                "personality_notes": (
                    "Recent CS masters grad, eager to learn, high energy, very responsive, needs mentoring"
                ),
                "priorities": [
                    "learn Bayesian methods hands-on",
                    "get first research publication",
                    "build relationship with experienced researchers",
                ],
            },
        },
    ]


def _task_record() -> dict:
    return {
        "task_id": "task_001",
        "goal": "Build Bayesian churn model, target NeurIPS",
        "requirements": [
            {"description": "bayesian statistics", "level": 0.8, "constraint_type": "soft"},
            {"description": "python programming", "level": 0.6, "constraint_type": "soft"},
            {"description": "academic paper writing", "level": 0.7, "constraint_type": "soft"},
        ],
        "offers": [
            {"description": "research collaboration", "strength": 0.8, "source": "explicit"},
            {"description": "co-authorship", "strength": 0.7, "source": "explicit"},
        ],
        "data_clearance": 0,
    }


def _build_context(requester_record: dict, candidate_record: dict, task_record: dict) -> MatchingContext:
    requester = UserProfile(
        user_id=requester_record["user_id"],
        role=requester_record["role"],
        capabilities={item["description"]: item["mu"] for item in requester_record["capabilities"]},
        needs={item["description"]: item["intensity"] for item in requester_record["needs"]},
        preferences={"soft_profile": requester_record["soft_profile"]},
    )
    candidate = UserProfile(
        user_id=candidate_record["user_id"],
        role=candidate_record["role"],
        capabilities={item["description"]: item["mu"] for item in candidate_record["capabilities"]},
        needs={item["description"]: item["intensity"] for item in candidate_record["needs"]},
        preferences={"soft_profile": candidate_record["soft_profile"]},
    )
    task = TaskSpec(
        task_id=task_record["task_id"],
        title=task_record["goal"],
        description=task_record["goal"],
        required_skills={item["description"]: item["level"] for item in task_record["requirements"]},
        metadata={"offers": task_record["offers"]},
    )
    card = CandidateCard(
        candidate_id=candidate.user_id,
        summary=(
            f"{candidate.user_id} offers "
            f"{', '.join(skill for skill in candidate.capabilities.keys())}"
        ),
        highlighted_strengths=list(candidate.capabilities.keys())[:2],
        highlighted_risks=[],
        explanation="Generated for Layer 3 dream fixture export.",
    )
    return MatchingContext(
        requester=requester,
        candidate=candidate,
        task=task,
        card=card,
    )


def build_fixture() -> dict:
    requester = _requester_record()
    candidates = _candidate_records()
    task = _task_record()

    config = SimulatorConfig(
        backend_type="mock",
        num_requester_personas=1,
        num_candidate_personas=5,
    )
    config.apply_seed()
    backend = RuleBasedBackend(config)
    context = _build_context(requester, candidates[0], task)

    requester_personas = RequesterPersonaGenerator(backend, config).generate(context)
    candidate_personas = CandidatePersonaGenerator(backend, config).generate(context)

    return {
        "metadata": {
            "generated_by": "simulator/generate_dreaming_fixture.py",
            "num_requester_personas": config.num_requester_personas,
            "num_candidate_personas": config.num_candidate_personas,
        },
        "task": task,
        "requester": requester,
        "candidates": candidates,
        "requester_personas": [asdict(persona) for persona in requester_personas],
        "candidate_personas": [asdict(persona) for persona in candidate_personas],
    }


def main() -> None:
    payload = build_fixture()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote dreaming fixture to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
