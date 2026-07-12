"""Generate a 20-task simulator dataset with persona bundles.

Output:
    simulator/20_Tasks_Testset.json
    simulator/20_Tasks_Testset_v2.json (with --version v2)

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
import argparse
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
V2_OUTPUT_PATH = Path(__file__).resolve().parent / "20_Tasks_Testset_v2.json"
V3_OUTPUT_PATH = Path(__file__).resolve().parent / "20_Tasks_Testset_v3.json"

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

OFFER_POOL_BY_SKILL = {
    "python": ["python mentoring", "backend automation experience"],
    "ml_systems": ["ml systems leadership", "model deployment experience"],
    "data_engineering": ["data pipeline ownership", "analytics infrastructure practice"],
    "bayesian_modeling": ["bayesian modeling mentorship", "statistical inference practice"],
    "paper_writing": ["research writing collaboration", "publication strategy support"],
    "frontend": ["frontend product polish", "dashboard design experience"],
    "backend": ["backend architecture practice", "api design ownership"],
    "product_design": ["product design portfolio growth", "user research exposure"],
    "project_management": ["project leadership experience", "cross functional coordination"],
    "communication": ["stakeholder communication practice", "structured collaboration"],
    "recommender_systems": ["recommender systems experience", "ranking evaluation practice"],
    "evaluation": ["evaluation methodology exposure", "benchmark design experience"],
    "statistics": ["statistical analysis mentorship", "experiment design practice"],
    "distributed_systems": ["distributed systems experience", "scalable service design"],
    "data_analysis": ["data analysis practice", "insight generation ownership"],
}


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


def _make_task_offers(
    required_skills: dict[str, float],
    rng: random.Random,
    task_idx: int,
    version: str = "v2",
) -> dict[str, float]:
    """Create task benefits that can semantically match candidate needs.

    Offer labels intentionally reuse skill/benefit words from candidate need
    descriptions so MapScore's S_need has real variance across tasks instead
    of collapsing to the same zero value for every candidate.
    """
    if version == "v3":
        offers: dict[str, float] = {}
        ranked_skills = sorted(required_skills, key=required_skills.get, reverse=True)
        for skill in ranked_skills[:3]:
            offers[skill] = _round_score(0.65 + 0.3 * required_skills[skill] + rng.uniform(-0.04, 0.04))

        decoy_pool = [skill for skill in SKILL_POOL if skill not in required_skills]
        for skill in rng.sample(decoy_pool, k=min(4, len(decoy_pool))):
            offers[skill] = _round_score(rng.uniform(0.05, 0.25))
        return offers

    offers: dict[str, float] = {}
    ranked_skills = sorted(required_skills, key=required_skills.get, reverse=True)
    for skill in ranked_skills[:3]:
        label = rng.choice(OFFER_POOL_BY_SKILL.get(skill, [f"{skill} experience"]))
        offers[label] = _round_score(0.55 + 0.4 * required_skills[skill] + rng.uniform(-0.08, 0.08))

    domain_offer = {
        "research": "research collaboration",
        "matching": "matching system experience",
        "marketplace": "marketplace product exposure",
        "ml_platform": "ml platform deployment practice",
        "analytics": "analytics decision support",
    }[TASK_DOMAINS[task_idx % len(TASK_DOMAINS)]]
    offers[domain_offer] = _round_score(rng.uniform(0.45, 0.9))
    return offers


def _make_task(
    task_idx: int,
    rng: random.Random,
    include_offers: bool = False,
    benchmark_version: str = "v1",
) -> TaskSpec:
    title = TASK_TITLES[task_idx % len(TASK_TITLES)]
    required_skills = _sample_skill_scores(rng, SKILL_POOL, 3, 5)
    offers = (
        _make_task_offers(required_skills, rng, task_idx, version=benchmark_version)
        if include_offers else {}
    )
    return TaskSpec(
        task_id=f"task_{task_idx:02d}",
        title=title,
        description=(
            f"{title} for a {rng.choice(TASK_DOMAINS)} initiative. "
            f"Needs strong execution across {', '.join(required_skills.keys())}."
        ),
        required_skills=required_skills,
        offers=offers,
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


def build_testset(
    num_tasks: int = 20,
    candidates_per_task: int = 20,
    include_offers: bool = False,
    benchmark_version: str = "v1",
) -> dict:
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
        task = _make_task(
            task_idx,
            rng,
            include_offers=include_offers,
            benchmark_version=benchmark_version,
        )
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
            "benchmark_version": benchmark_version,
            "has_task_offers": include_offers,
            "outcome_oracle": (
                "v3_offer_need_fit_workload_visibility_budget_hidden_latents"
                if benchmark_version == "v3" else "legacy"
            ),
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=("v1", "v2", "v3"),
        default="v1",
        help="v2 adds explicit task offers; v3 fixes the benchmark version for the offer-aware outcome oracle.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    args = parser.parse_args()

    include_offers = args.version in {"v2", "v3"}
    default_path = {"v1": OUTPUT_PATH, "v2": V2_OUTPUT_PATH, "v3": V3_OUTPUT_PATH}[args.version]
    output_path = args.output or default_path
    payload = build_testset(include_offers=include_offers, benchmark_version=args.version)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote 20-task simulator testset to {output_path}")


if __name__ == "__main__":
    main()
