"""Generate a batch matching benchmark with shared candidate pools.

Unlike ``20_Tasks_Testset_v3`` (task-local candidates, no capacity conflict),
each batch here has multiple requester-task pairs competing for the same
shared candidates under finite capacity.

Profiles reuse the v3 schema and generation rules from
``generate_20_tasks_testset.py`` (offers, skills, cards, latents).
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from simulator.config import SimulatorConfig
from simulator.generate_20_tasks_testset import (
    ROLE_POOL,
    SKILL_POOL,
    TASK_DOMAINS,
    TASK_TITLES,
    _make_card,
    _make_context,
    _make_proposer,
    _make_task,
    _round_score,
    _sample_skill_scores,
)
from simulator.mock_backend import RuleBasedBackend
from simulator.persona_generator import (
    CandidatePersonaGenerator,
    RequesterPersonaGenerator,
)
from simulator.types import UserProfile

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "Batch_Matching_Testset_v1.json"
DEFAULT_SEEDS = (42, 123, 456)


def _discrete_coverage(required: dict[str, float], caps: dict[str, float]) -> float:
    if not required:
        return 1.0
    total = sum(max(0.0, float(v)) for v in required.values())
    if total <= 0.0:
        return 0.0
    covered = 0.0
    for skill, level in required.items():
        if float(caps.get(skill, -1.0)) >= float(level):
            covered += max(0.0, float(level))
    return covered / total


def _offer_need_token_fit(offers: dict[str, float], needs: dict[str, float]) -> float:
    """Lightweight need-fit used only inside the generator for competition shaping."""
    if not offers or not needs:
        return 0.0
    total = sum(max(0.0, float(v)) for v in needs.values())
    if total <= 0.0:
        return 0.0
    score = 0.0
    for need, intensity in needs.items():
        best = 0.0
        for offer, strength in offers.items():
            if need == offer or need in offer or offer in need:
                best = max(best, float(strength))
        score += max(0.0, float(intensity)) * best
    return score / total


def _make_shared_candidate(
    batch_idx: int,
    cand_idx: int,
    rng: random.Random,
    *,
    specialist_skills: list[str] | None = None,
    broad: bool = False,
    star: bool = False,
) -> UserProfile:
    """Task-agnostic candidate from SKILL_POOL (not conditioned on one task)."""
    if star and specialist_skills:
        chosen = set(specialist_skills)
        chosen.update(rng.sample(SKILL_POOL, k=min(4, len(SKILL_POOL))))
        capabilities = {
            skill: _round_score(rng.uniform(0.78, 0.98))
            for skill in chosen
        }
    elif broad:
        chosen = set(rng.sample(SKILL_POOL, k=rng.randint(6, 9)))
        capabilities = {
            skill: _round_score(rng.uniform(0.45, 0.85))
            for skill in chosen
        }
    elif specialist_skills:
        chosen = set(specialist_skills)
        chosen.update(rng.sample(SKILL_POOL, k=2))
        capabilities = {
            skill: _round_score(rng.uniform(0.55, 0.95))
            for skill in chosen
        }
    else:
        capabilities = _sample_skill_scores(rng, SKILL_POOL, 3, 6)

    needs = _sample_skill_scores(rng, SKILL_POOL, 2, 4)
    role = rng.choice(ROLE_POOL)
    availability = rng.choice(["high", "medium", "low"])
    current_load = _round_score(rng.uniform(0.1, 0.95))
    return UserProfile(
        user_id=f"batch_{batch_idx:02d}_candidate_{cand_idx:02d}",
        role=role,
        capabilities=capabilities,
        needs=needs,
        preferences={
            "availability": availability,
            "interests": " ".join(sorted(capabilities.keys())[:4]),
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


def _pair_is_feasible_public(
    task: dict,
    candidate_profile: dict,
    *,
    min_coverage: float = 0.0,
) -> bool:
    """Public feasibility used as the shared edge filter.

    A pair is infeasible when discrete weighted skill coverage is zero
    (or below ``min_coverage``), or when an explicit planted hard conflict
    is present in public preferences/constraints.
    """
    required = task.get("required_skills", {}) or {}
    caps = candidate_profile.get("capabilities", {}) or {}
    coverage = _discrete_coverage(required, caps)
    if coverage <= min_coverage:
        return False

    # Planted public hard conflict: low availability + high urgency + low flexibility.
    prefs = candidate_profile.get("preferences", {}) or {}
    constraints = candidate_profile.get("constraints", {}) or {}
    meta = task.get("metadata", {}) or {}
    if (
        prefs.get("availability") == "low"
        and constraints.get("schedule_flexibility") == "low"
        and meta.get("urgency") == "high"
        and candidate_profile.get("_force_infeasible")
    ):
        return False
    if candidate_profile.get("_force_infeasible_tasks"):
        if task.get("task_id") in candidate_profile["_force_infeasible_tasks"]:
            return False
    return True


def _ensure_competition(
    rng: random.Random,
    tasks: list,
    candidates: list[UserProfile],
) -> None:
    """Shape profiles so each batch has realistic capacity competition."""
    if len(tasks) < 2 or len(candidates) < 2:
        return

    star = candidates[0]
    backup_poor = candidates[-1]

    # Two tasks both have high coverage of the star candidate.
    for task_obj in tasks[:2]:
        for skill, level in list(task_obj.required_skills.items())[:3]:
            star.capabilities[skill] = _round_score(
                max(float(star.capabilities.get(skill, 0.0)), float(level) + rng.uniform(0.05, 0.15))
            )

    # One candidate has a markedly better need fit for task 0 offers.
    focus = tasks[0]
    need_lover = candidates[1]
    top_offers = sorted(focus.offers, key=focus.offers.get, reverse=True)[:2]
    for offer in top_offers:
        need_lover.needs[offer] = _round_score(rng.uniform(0.85, 0.98))
    # Dilute other candidates' need fit for those offers a bit.
    for other in candidates[2:]:
        for offer in top_offers:
            if offer in other.needs:
                other.needs[offer] = _round_score(min(float(other.needs[offer]), 0.35))

    # One task has limited feasible backup: make backup_poor miss most of task 2's skills.
    scarce = tasks[min(2, len(tasks) - 1)]
    for skill, level in scarce.required_skills.items():
        if skill in backup_poor.capabilities and float(backup_poor.capabilities[skill]) >= float(level):
            backup_poor.capabilities[skill] = _round_score(float(level) - rng.uniform(0.15, 0.35))

    # Ensure most tasks have >= 2 feasible candidates by boosting two random candidates.
    for task_obj in tasks:
        feasible = [
            c for c in candidates
            if _discrete_coverage(task_obj.required_skills, c.capabilities) > 0.0
        ]
        if len(feasible) >= 2:
            continue
        boosters = rng.sample(candidates, k=min(2, len(candidates)))
        for c in boosters:
            for skill, level in list(task_obj.required_skills.items())[:2]:
                c.capabilities[skill] = _round_score(
                    max(float(c.capabilities.get(skill, 0.0)), float(level) + 0.02)
                )


def _plant_infeasible_pairs(
    rng: random.Random,
    tasks: list,
    candidates: list[UserProfile],
) -> dict[str, list[str]]:
    """Return candidate_id -> [task_ids] that are forced infeasible via public tags."""
    forced: dict[str, list[str]] = {}
    if not tasks or not candidates:
        return forced
    # Zero out coverage for one (candidate, task) pair and tag it.
    c = candidates[-1]
    t = tasks[-1]
    for skill in list(t.required_skills.keys()):
        if skill in c.capabilities:
            c.capabilities[skill] = _round_score(
                min(float(c.capabilities[skill]), float(t.required_skills[skill]) - 0.2)
            )
            if c.capabilities[skill] < 0:
                c.capabilities[skill] = 0.0
    # Extra planted conflict on another pair using public flag stored temporarily.
    if len(candidates) >= 2 and len(tasks) >= 2:
        forced[candidates[-2].user_id] = [tasks[-2].task_id]
    return forced


def build_batch(
    batch_idx: int,
    rng: random.Random,
    *,
    tasks_per_batch: int,
    candidates_per_batch: int,
    candidate_capacity: int,
    requester_generator: RequesterPersonaGenerator,
    candidate_generator: CandidatePersonaGenerator,
) -> dict:
    # Shared candidates generated once (task-agnostic).
    specialist_focus = rng.sample(SKILL_POOL, k=min(4, len(SKILL_POOL)))
    candidates: list[UserProfile] = []
    for cand_idx in range(1, candidates_per_batch + 1):
        if cand_idx == 1:
            cand = _make_shared_candidate(
                batch_idx, cand_idx, rng, specialist_skills=specialist_focus, star=True
            )
        elif cand_idx == 2:
            cand = _make_shared_candidate(batch_idx, cand_idx, rng, broad=True)
        else:
            focus = rng.sample(SKILL_POOL, k=rng.randint(2, 4))
            cand = _make_shared_candidate(
                batch_idx, cand_idx, rng, specialist_skills=focus
            )
        candidates.append(cand)

    # Independent tasks + requesters (v3 offer rule).
    task_objs = []
    proposers = []
    for local_idx in range(1, tasks_per_batch + 1):
        global_idx = (batch_idx - 1) * tasks_per_batch + local_idx
        task = _make_task(
            global_idx,
            rng,
            include_offers=True,
            benchmark_version="v3",
        )
        # Stable batch-local task id.
        task.task_id = f"batch_{batch_idx:02d}_task_{local_idx:02d}"
        task.title = TASK_TITLES[(global_idx - 1) % len(TASK_TITLES)]
        task.description = (
            f"{task.title} for a {rng.choice(TASK_DOMAINS)} initiative. "
            f"Needs strong execution across {', '.join(task.required_skills.keys())}."
        )
        proposer = _make_proposer(global_idx, rng, task.required_skills)
        proposer.user_id = f"batch_{batch_idx:02d}_requester_{local_idx:02d}"
        task_objs.append(task)
        proposers.append(proposer)

    _ensure_competition(rng, task_objs, candidates)
    forced_infeasible = _plant_infeasible_pairs(rng, task_objs, candidates)

    shared_candidates = []
    for cand in candidates:
        profile = asdict(cand)
        # Strip any private generator tags from the published profile.
        profile.pop("_force_infeasible", None)
        profile.pop("_force_infeasible_tasks", None)
        shared_candidates.append({"candidate_profile": profile})

    # Attach forced infeasible as public metadata on pairs only (not MapScore/oracle).
    task_entries = []
    for task, proposer in zip(task_objs, proposers):
        pair_entries = []
        for cand in candidates:
            card = _make_card(cand, task, rng)
            context = _make_context(proposer, cand, task, card, rng)
            requester_personas = requester_generator.generate(context)
            candidate_personas = candidate_generator.generate(context)

            cand_profile = asdict(cand)
            cand_profile.pop("_force_infeasible", None)
            cand_profile.pop("_force_infeasible_tasks", None)

            feasible = _pair_is_feasible_public(asdict(task), cand_profile)
            if cand.user_id in forced_infeasible and task.task_id in forced_infeasible[cand.user_id]:
                feasible = False

            pair_entries.append({
                "candidate_id": cand.user_id,
                # Snapshot of the shared profile (identical across tasks in the batch).
                "candidate_profile": deepcopy(cand_profile),
                "candidate_card": asdict(card),
                "requester_personas": [asdict(p) for p in requester_personas],
                "candidate_personas": [asdict(p) for p in candidate_personas],
                "context_latents": {
                    "history": context.history,
                    "latent_requester_preferences": context.latent_requester_preferences,
                    "latent_candidate_preferences": context.latent_candidate_preferences,
                    "latent_interpersonal_affinity": context.latent_interpersonal_affinity,
                    "latent_risk_tolerance": context.latent_risk_tolerance,
                    "latent_opportunity_bias": context.latent_opportunity_bias,
                },
                "public_feasible": feasible,
            })

        # Sanity: most tasks should have >= 2 feasible candidates.
        n_feas = sum(1 for p in pair_entries if p["public_feasible"])
        if n_feas < 2:
            # Emergency boost: mark top-2 coverage pairs feasible by lifting caps
            # already done in _ensure_competition; recompute.
            for p in sorted(
                pair_entries,
                key=lambda x: _discrete_coverage(
                    task.required_skills, x["candidate_profile"]["capabilities"]
                ),
                reverse=True,
            )[:2]:
                # Lift capabilities so coverage > 0 and mark feasible unless forced.
                cid = p["candidate_id"]
                if cid in forced_infeasible and task.task_id in forced_infeasible.get(cid, []):
                    continue
                for skill, level in list(task.required_skills.items())[:2]:
                    p["candidate_profile"]["capabilities"][skill] = _round_score(
                        max(
                            float(p["candidate_profile"]["capabilities"].get(skill, 0.0)),
                            float(level) + 0.05,
                        )
                    )
                    # Keep shared candidate profile in sync.
                    for shared in shared_candidates:
                        if shared["candidate_profile"]["user_id"] == cid:
                            shared["candidate_profile"]["capabilities"][skill] = (
                                p["candidate_profile"]["capabilities"][skill]
                            )
                    for other_task in task_entries:
                        for other_pair in other_task["pair_entries"]:
                            if other_pair["candidate_id"] == cid:
                                other_pair["candidate_profile"]["capabilities"][skill] = (
                                    p["candidate_profile"]["capabilities"][skill]
                                )
                p["public_feasible"] = _pair_is_feasible_public(
                    asdict(task), p["candidate_profile"]
                )

        task_entries.append({
            "task": asdict(task),
            "proposer_profile": asdict(proposer),
            "pair_entries": pair_entries,
        })

    # After possible emergency boosts, re-sync shared profiles from first task snapshot
    # and recompute feasibility consistently for every pair.
    profile_by_id = {
        c["candidate_profile"]["user_id"]: deepcopy(c["candidate_profile"])
        for c in shared_candidates
    }
    # Prefer the latest synced capabilities from any pair_entry.
    for te in task_entries:
        for pe in te["pair_entries"]:
            profile_by_id[pe["candidate_id"]] = deepcopy(pe["candidate_profile"])

    for shared in shared_candidates:
        cid = shared["candidate_profile"]["user_id"]
        shared["candidate_profile"] = deepcopy(profile_by_id[cid])

    for te in task_entries:
        for pe in te["pair_entries"]:
            pe["candidate_profile"] = deepcopy(profile_by_id[pe["candidate_id"]])
            feasible = _pair_is_feasible_public(te["task"], pe["candidate_profile"])
            cid = pe["candidate_id"]
            if cid in forced_infeasible and te["task"]["task_id"] in forced_infeasible[cid]:
                feasible = False
            pe["public_feasible"] = feasible

    return {
        "batch_id": f"batch_{batch_idx:02d}",
        "candidate_capacity": candidate_capacity,
        "shared_candidates": shared_candidates,
        "tasks": task_entries,
    }


def build_batch_testset(
    *,
    seed: int = 42,
    num_batches: int = 20,
    tasks_per_batch: int = 5,
    candidates_per_batch: int = 7,
    candidate_capacity: int = 1,
) -> dict:
    cfg = SimulatorConfig(
        backend_type="mock",
        random_seed=seed,
        persona_selection_seed=seed,
        num_requester_personas=4,
        num_candidate_personas=4,
        trace_verbose=False,
    )
    cfg.apply_seed()
    backend = RuleBasedBackend(cfg)
    requester_generator = RequesterPersonaGenerator(backend, cfg)
    candidate_generator = CandidatePersonaGenerator(backend, cfg)
    rng = random.Random(seed)

    batches = [
        build_batch(
            batch_idx,
            rng,
            tasks_per_batch=tasks_per_batch,
            candidates_per_batch=candidates_per_batch,
            candidate_capacity=candidate_capacity,
            requester_generator=requester_generator,
            candidate_generator=candidate_generator,
        )
        for batch_idx in range(1, num_batches + 1)
    ]

    return {
        "metadata": {
            "generated_by": "simulator/generate_batch_matching_testset.py",
            "benchmark_name": "Batch_Matching_Testset_v1",
            "benchmark_version": "v3_batch_shared_pool",
            "num_batches": num_batches,
            "tasks_per_batch": tasks_per_batch,
            "candidates_per_batch": candidates_per_batch,
            "candidate_capacity": candidate_capacity,
            "task_capacity": 1,
            "num_requester_personas_per_pair": cfg.num_requester_personas,
            "num_candidate_personas_per_pair": cfg.num_candidate_personas,
            "backend_type": cfg.backend_type,
            "random_seed": seed,
            "has_task_offers": True,
            "shared_candidate_pool": True,
            "outcome_oracle": "v3_offer_need_fit_workload_visibility_budget_hidden_latents",
            "notes": (
                "Candidates are shared within each batch. Public fields only for "
                "matching preferences; context_latents are hidden oracle signals."
            ),
        },
        "batches": batches,
    }


def write_seeded_datasets(
    output: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    **kwargs,
) -> list[Path]:
    """Write one JSON per seed next to ``output`` (or ``output`` if single)."""
    output = Path(output)
    written: list[Path] = []
    for seed in seeds:
        payload = build_batch_testset(seed=seed, **kwargs)
        if len(seeds) == 1:
            path = output
        else:
            stem = output.stem
            path = output.with_name(f"{stem}_seed{seed}{output.suffix}")
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(path)
        print(f"Wrote batch matching testset (seed={seed}) to {path}")
    # Also write the first seed to the canonical path for convenience.
    if len(seeds) > 1:
        primary = build_batch_testset(seed=seeds[0], **kwargs)
        output.write_text(json.dumps(primary, indent=2), encoding="utf-8")
        print(f"Wrote canonical batch matching testset (seed={seeds[0]}) to {output}")
        if output not in written:
            written.insert(0, output)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate batch matching benchmark with shared candidate pools."
    )
    parser.add_argument("--seed", type=int, default=None, help="Single seed (overrides --seeds).")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated seeds for multi-seed generation (default: 42,123,456).",
    )
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--tasks-per-batch", type=int, default=5)
    parser.add_argument("--candidates-per-batch", type=int, default=7)
    parser.add_argument("--candidate-capacity", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.seed is not None:
        seeds = (args.seed,)
    else:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    write_seeded_datasets(
        args.output,
        seeds=seeds,
        num_batches=args.num_batches,
        tasks_per_batch=args.tasks_per_batch,
        candidates_per_batch=args.candidates_per_batch,
        candidate_capacity=args.candidate_capacity,
    )


if __name__ == "__main__":
    main()
