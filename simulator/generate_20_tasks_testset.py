"""Generate a 20-task simulator dataset with persona bundles.

Output:
    simulator/20_Tasks_Testset.json

Each task contains:
  - one proposer/requester profile
  - task specification
  - 20 candidate profiles
  - candidate cards
  - requester/candidate persona bundles generated from PersonaGenerator
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from simulator.config import SimulatorConfig
from simulator.mock_backend import RuleBasedBackend
from simulator.persona_generator import (
    CandidatePersonaGenerator,
    RequesterPersonaGenerator,
)
from simulator.types import CandidateCard, MatchingContext, TaskSpec, UserProfile


OUTPUT_PATH = Path(__file__).resolve().parent / "20_Tasks_Testset.json"

SKILL_POOL = [
    "python",
    "ml_systems",
    "data_engineering",
    "bayesian_modeling",
    "paper_writing",
    "frontend",
    "backend",
    "product_design",
    "project_management",
    "communication",
    "recommender_systems",
    "evaluation",
    "statistics",
    "distributed_systems",
    "data_analysis",
]

ROLE_POOL = [
    "researcher",
    "engineer",
    "data_scientist",
    "ml_engineer",
    "project_manager",
    "designer",
]

TASK_TITLES = [
    "Build Bayesian Churn Model",
    "Design Recommendation API",
    "Launch RAG Evaluation Dashboard",
    "Create ML Data Pipeline",
    "Prototype Hiring Assistant",
    "Improve Matching Quality Monitor",
    "Ship Research Collaboration Portal",
    "Develop Demand Forecasting Workflow",
    "Build Knowledge Graph Explorer",
    "Optimize Retrieval Stack",
    "Create Project Risk Scorer",
    "Develop User Segmentation Toolkit",
    "Deploy Online Learning Service",
    "Design Task Allocation Engine",
    "Implement Experiment Tracking UI",
    "Build Collaboration Outcome Predictor",
    "Create Portfolio Ranking Service",
    "Prototype Marketplace Search",
    "Develop Causal Analysis Notebook",
    "Ship Candidate Vetting Tool",
]

TASK_DOMAINS = [
    "research",
    "matching",
    "marketplace",
    "ml_platform",
    "analytics",
]


def _round_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 2)


def _sample_skill_scores(rng: random.Random, skills: list[str], min_n: int, max_n: int) -> dict[str, float]:
    chosen = rng.sample(skills, k=rng.randint(min_n, max_n))
    return {skill: _round_score(rng.uniform(0.35, 0.98)) for skill in chosen}


def _make_proposer(task_idx: int, rng: random.Random, required_skills: dict[str, float]) -> UserProfile:
    proposer_skills = set(required_skills.keys())
    proposer_skills.update(rng.sample(SKILL_POOL, k=2))
    capabilities = {
        skill: _round_score(rng.uniform(0.45, 0.95))
        for skill in proposer_skills
    }
    needs = {
        skill: _round_score(rng.uniform(0.5, 0.95))
        for skill in required_skills
    }
    timezone = rng.choice(["US_Eastern", "US_Pacific", "UTC", "CET", "SGT"])
    return UserProfile(
        user_id=f"task_{task_idx:02d}_proposer",
        role="requester",
        capabilities=capabilities,
        needs=needs,
        preferences={
            "communication_style": rng.choice(["async_first", "structured_sync", "mixed"]),
            "timezone": timezone,
            "urgency_preference": rng.choice(["high", "medium", "low"]),
        },
        constraints={
            "meeting_load": _round_score(rng.uniform(0.1, 0.8)),
        },
        history_summary=(
            f"Requester for task {task_idx:02d}; previous project delivery record is "
            f"{rng.choice(['strong', 'mixed', 'solid'])}."
        ),
    )


def _make_candidate(task_idx: int, cand_idx: int, rng: random.Random, required_skills: dict[str, float]) -> UserProfile:
    chosen = set(required_skills.keys())
    chosen.update(rng.sample(SKILL_POOL, k=2))
    capabilities = {}
    for skill in chosen:
        base = required_skills.get(skill, rng.uniform(0.35, 0.9))
        capabilities[skill] = _round_score(base + rng.uniform(-0.25, 0.2))

    needs = _sample_skill_scores(rng, SKILL_POOL, 2, 4)
    role = rng.choice(ROLE_POOL)
    availability = rng.choice(["high", "medium", "low"])
    current_load = _round_score(rng.uniform(0.1, 0.95))
    return UserProfile(
        user_id=f"task_{task_idx:02d}_candidate_{cand_idx:02d}",
        role=role,
        capabilities=capabilities,
        needs=needs,
        preferences={
            "availability": availability,
            "interests": " ".join(sorted(chosen)[:4]),
            "timezone": rng.choice(["US_Eastern", "US_Pacific", "UTC", "CET", "SGT"]),
            "current_load": current_load,
        },
        constraints={
            "workload": current_load,
            "schedule_flexibility": rng.choice(["high", "medium", "low"]),
        },
        history_summary=(
            f"{role} with {rng.randint(1, 9)} completed projects and "
            f"{rng.choice(['few', 'some', 'many'])} prior collaborations."
        ),
    )


def _make_task(task_idx: int, rng: random.Random) -> TaskSpec:
    title = TASK_TITLES[task_idx % len(TASK_TITLES)]
    required_skills = _sample_skill_scores(rng, SKILL_POOL, 3, 5)
    return TaskSpec(
        task_id=f"task_{task_idx:02d}",
        title=title,
        description=(
            f"{title} for a {rng.choice(TASK_DOMAINS)} initiative. "
            f"Needs strong execution across {', '.join(required_skills.keys())}."
        ),
        required_skills=required_skills,
        metadata={
            "urgency": rng.choice(["high", "medium", "low"]),
            "budget": rng.choice(["lean", "competitive", "premium"]),
            "visibility": rng.choice(["high", "medium"]),
            "duration_weeks": rng.randint(2, 16),
            "domain": rng.choice(TASK_DOMAINS),
        },
    )


def _make_card(candidate: UserProfile, task: TaskSpec, rng: random.Random) -> CandidateCard:
    strengths = sorted(candidate.capabilities, key=candidate.capabilities.get, reverse=True)[:3]
    risks = []
    current_load = float(candidate.preferences.get("current_load", 0.5))
    if current_load > 0.75:
        risks.append("High current workload may limit responsiveness")
    if candidate.preferences.get("availability") == "low":
        risks.append("Low availability may increase scheduling risk")
    if not risks:
        risks.append(rng.choice([
            "Needs clearer project scope before committing",
            "Would benefit from more structured onboarding",
            "Potential timezone coordination overhead",
        ]))
    return CandidateCard(
        candidate_id=candidate.user_id,
        summary=(
            f"{candidate.role} with strengths in {', '.join(strengths)} "
            f"for task '{task.title}'."
        ),
        highlighted_strengths=strengths,
        highlighted_risks=risks,
        explanation=(
            f"Generated profile for {candidate.user_id} against {task.task_id}. "
            f"Availability={candidate.preferences.get('availability')}."
        ),
    )


def _make_context(
    requester: UserProfile,
    candidate: UserProfile,
    task: TaskSpec,
    card: CandidateCard,
    rng: random.Random,
) -> MatchingContext:
    return MatchingContext(
        requester=requester,
        candidate=candidate,
        task=task,
        card=card,
        history={
            "prior_collaboration": rng.choice([True, False]),
            "prior_skipped": rng.choice([True, False]),
            "dispute_rate": _round_score(rng.uniform(0.0, 0.2)),
        },
        latent_requester_preferences={
            "structured": _round_score(rng.uniform(0.0, 1.0)),
            "fastpaced": _round_score(rng.uniform(0.0, 1.0)),
        },
        latent_candidate_preferences={
            "creative": _round_score(rng.uniform(0.0, 1.0)),
            "structured": _round_score(rng.uniform(0.0, 1.0)),
        },
        latent_interpersonal_affinity=round(rng.uniform(-1.0, 1.0), 2),
        latent_risk_tolerance=_round_score(rng.uniform(0.0, 1.0)),
        latent_opportunity_bias=round(rng.uniform(-1.0, 1.0), 2),
    )


def build_testset(num_tasks: int = 20, candidates_per_task: int = 20) -> dict:
    cfg = SimulatorConfig(
        backend_type="mock",
        random_seed=42,
        persona_selection_seed=42,
        num_requester_personas=4,
        num_candidate_personas=4,
        trace_verbose=False,
    )
    cfg.apply_seed()
    backend = RuleBasedBackend(cfg)
    requester_generator = RequesterPersonaGenerator(backend, cfg)
    candidate_generator = CandidatePersonaGenerator(backend, cfg)
    rng = random.Random(cfg.random_seed)

    tasks = []
    for task_idx in range(1, num_tasks + 1):
        task = _make_task(task_idx, rng)
        proposer = _make_proposer(task_idx, rng, task.required_skills)
        candidates = []

        for cand_idx in range(1, candidates_per_task + 1):
            candidate = _make_candidate(task_idx, cand_idx, rng, task.required_skills)
            card = _make_card(candidate, task, rng)
            context = _make_context(proposer, candidate, task, card, rng)
            requester_personas = requester_generator.generate(context)
            candidate_personas = candidate_generator.generate(context)

            candidates.append({
                "candidate_profile": asdict(candidate),
                "candidate_card": asdict(card),
                "requester_personas": [asdict(persona) for persona in requester_personas],
                "candidate_personas": [asdict(persona) for persona in candidate_personas],
                "context_latents": {
                    "history": context.history,
                    "latent_requester_preferences": context.latent_requester_preferences,
                    "latent_candidate_preferences": context.latent_candidate_preferences,
                    "latent_interpersonal_affinity": context.latent_interpersonal_affinity,
                    "latent_risk_tolerance": context.latent_risk_tolerance,
                    "latent_opportunity_bias": context.latent_opportunity_bias,
                },
            })

        tasks.append({
            "task": asdict(task),
            "proposer_profile": asdict(proposer),
            "candidates": candidates,
        })

    return {
        "metadata": {
            "generated_by": "simulator/generate_20_tasks_testset.py",
            "num_tasks": num_tasks,
            "candidates_per_task": candidates_per_task,
            "num_requester_personas_per_pair": cfg.num_requester_personas,
            "num_candidate_personas_per_pair": cfg.num_candidate_personas,
            "backend_type": cfg.backend_type,
            "random_seed": cfg.random_seed,
        },
        "tasks": tasks,
    }


def main() -> None:
    payload = build_testset()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote 20-task simulator testset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
