"""V3 LLM Dreaming ablation runner.

Compares:
  - scap_greedy: static S_cap-only selection
  - mapscore_greedy: static full analytical MapScore selection
  - mapscore_dreaming: analytical top-k shortlist followed by existing
    LLM_Dreaming PlanningLayer reranking

The runner reuses the existing DreamSimulator, bilateral simulator, outcome
simulator, and reward code. Dreaming receives only public structured profiles
and task contents; hidden simulator latents are used only by the outcome oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


TESTS_DIR = Path(__file__).resolve().parent
ONLINE_LEARNING_DIR = TESTS_DIR.parent
REPO_ROOT = ONLINE_LEARNING_DIR.parent
MAPPING_ALGO_DIR = REPO_ROOT / "mapping-algo"
LOGS_DIR = REPO_ROOT / "logs"
EXPERIMENTS_DIR = REPO_ROOT / "Experiments"
DEFAULT_TESTSET = REPO_ROOT / "simulator" / "20_Tasks_Testset_v3_tiered.json"

for path in (TESTS_DIR, ONLINE_LEARNING_DIR, MAPPING_ALGO_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from WorldModel import WorldModel  # noqa: E402
from config import MatchConfig  # noqa: E402
from datatypes import CapabilityEntry, MatchResult, NeedEntry, Task, UserState  # noqa: E402
from LLM_Dreaming.LLM_dreaming import (  # noqa: E402
    DreamSimulator,
    ExtendedProfile,
    PlanningLayer,
    SoftProfile,
)
from simulator.config import SimulatorConfig  # noqa: E402
from simulator.mock_backend import RuleBasedBackend  # noqa: E402
from simulator.outcome_simulator import OutcomeSimulator  # noqa: E402
from simulator.reward import compute_reward  # noqa: E402
from simulator.bilateral_simulator import BilateralSimulator  # noqa: E402

from run_20_tasks_evaluation import (  # noqa: E402
    CFG,
    ENC,
    EVAL_THETA_C,
    EVAL_THETA_N,
    SCAP_THETA_C,
    SCAP_THETA_N,
    TRUE_SIGMA,
    build_learning_state_from_priors,
    build_matching_context,
    build_requester_for_match,
    build_task,
    build_user_state,
    compute_ground_truth_M,
)


OUTCOME_FIELDS = [
    "mutual_accept_probability",
    "completion_probability",
    "requester_satisfaction",
    "candidate_satisfaction",
    "joint_reward",
    "total_reward",
]
PROCESS_FIELDS = [
    "time_feasibility",
    "collaboration_style_compatibility",
    "candidate_usability",
    "average_messages_per_task",
    "api_calls_per_task",
    "tokens_per_task",
    "wall_clock_time_per_task",
]


class CountingDreamSimulator(DreamSimulator):
    """DreamSimulator wrapper that records calls, approximate tokens, and time."""

    def __init__(self, *args, seed: int = 0, **kwargs):
        self.seed = seed
        self.api_calls = 0
        self.approx_tokens = 0
        self.wall_clock_seconds = 0.0
        self.api_errors: List[str] = []
        super().__init__(*args, **kwargs)

    def reset_counters(self) -> None:
        self.api_calls = 0
        self.approx_tokens = 0
        self.wall_clock_seconds = 0.0
        self.api_errors = []

    def _call_llm(self, system: str, messages: list) -> str:
        seeded_system = f"{system}\n\nExperiment seed: {self.seed}."
        start = time.perf_counter()
        try:
            output = super()._call_llm(seeded_system, messages)
        except Exception as exc:
            self.api_errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
            if "matching quality evaluator" in seeded_system:
                output = json.dumps({
                    "time_energy": {"score": 0.5, "reason": "API error fallback."},
                    "priority_alignment": {"score": 0.5, "reason": "API error fallback."},
                    "collab_style": {"score": 0.5, "reason": "API error fallback."},
                    "personality_fit": {"score": 0.5, "reason": "API error fallback."},
                    "overall_compatibility": 0.5,
                    "top_risk": "api_error_fallback",
                    "top_synergy": "api_error_fallback",
                    "recommendation": "risky_match",
                    "api_error": True,
                })
            elif "posted the task" in seeded_system:
                output = (
                    "Thanks for the context. I would like to understand your availability, "
                    "working style, and any constraints before deciding whether this collaboration is feasible."
                )
            else:
                output = (
                    "I can discuss the task based on my public profile. My availability, workload, "
                    "needs, and the task offers will determine whether this is a practical fit."
                )
        elapsed = time.perf_counter() - start
        self.api_calls += 1
        self.wall_clock_seconds += elapsed
        text_in = seeded_system + "\n" + "\n".join(str(m.get("content", "")) for m in messages)
        self.approx_tokens += _approx_tokens(text_in) + _approx_tokens(output)
        return output


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _capabilities_from_state(state: UserState) -> List[CapabilityEntry]:
    return [
        CapabilityEntry(
            cap.embedding,
            mu=float(cap.mu),
            sigma=float(cap.sigma),
            source=cap.source,
            description=cap.description,
        )
        for cap in state.capabilities
    ]


def _needs_from_state(state: UserState) -> List[NeedEntry]:
    return [
        NeedEntry(need.embedding, intensity=float(need.intensity), description=need.description)
        for need in state.needs
    ]


def public_soft_profile(profile: dict, role: str) -> SoftProfile:
    prefs = profile.get("preferences", {}) or {}
    constraints = profile.get("constraints", {}) or {}
    availability = str(prefs.get("availability", "medium"))
    current_load = prefs.get("current_load", constraints.get("workload", "unknown"))
    timezone = str(prefs.get("timezone", "unknown"))
    return SoftProfile(
        availability=f"{availability}; current_load={current_load}",
        timezone=timezone,
        deadline_pressure=str(prefs.get("urgency_preference", "unknown")),
        collab_style=str(prefs.get("communication_style", "mixed")),
        communication=str(prefs.get("communication_style", "mixed")),
        personality_notes=profile.get("history_summary") or f"Public {role} profile only.",
        priorities=[
            "complete the task successfully",
            "manage availability and workload honestly",
            "seek task benefits matching own needs",
        ],
    )


def public_extended_profile(
    profile: dict,
    state: UserState,
    role: str,
) -> ExtendedProfile:
    public_state = UserState(
        user_id=state.user_id,
        capabilities=_capabilities_from_state(state),
        needs=_needs_from_state(state),
        clearance_level=state.clearance_level,
        soft_profile=None,
    )
    return ExtendedProfile(public_state, public_soft_profile(profile, role))


def assert_public_dream_inputs(task_entry: dict, public_payload: dict) -> None:
    """Guardrail: hidden latents must not enter Dreaming prompts/profiles."""
    serialized = json.dumps(public_payload, sort_keys=True)
    forbidden = [
        "context_latents",
        "latent_requester_preferences",
        "latent_candidate_preferences",
        "latent_interpersonal_affinity",
        "latent_risk_tolerance",
        "latent_opportunity_bias",
    ]
    for key in forbidden:
        if key in serialized:
            raise AssertionError(f"Hidden latent key leaked into Dreaming input: {key}")


def analytical_matches(
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
    mode: str,
) -> Dict[str, MatchResult]:
    if mode == "scap":
        world_model = WorldModel(config=CFG, theta_c=SCAP_THETA_C, theta_n=SCAP_THETA_N)
    elif mode == "mapscore":
        world_model = WorldModel(config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N)
    else:
        raise ValueError(mode)
    return {
        candidate.user_id: world_model.compute_match(
            requester,
            candidate,
            task,
            use_ucb=False,
            round_t=1,
        )
        for candidate in candidate_pool
    }


def select_top1(matches: Dict[str, MatchResult]) -> str:
    return max(matches, key=lambda cid: (matches[cid].match_score, cid))


def analytical_top_k(matches: Dict[str, MatchResult], k: int) -> List[MatchResult]:
    return sorted(matches.values(), key=lambda m: (m.match_score, m.candidate_id), reverse=True)[:k]


def selected_match_metrics(match: MatchResult) -> dict:
    return {
        "S_cap": round(float(match.s_cap), 4),
        "S_need": round(float(match.s_need), 4),
        "MapScore": round(float(match.match_score), 4),
    }


def compute_bilateral_outcome(task_entry: dict, candidate_id: str) -> Tuple[dict, dict]:
    cfg = SimulatorConfig(
        backend_type="mock",
        random_seed=42,
        persona_selection_seed=42,
        num_requester_personas=6,
        num_candidate_personas=6,
        decision_mode="threshold",
        trace_verbose=False,
    )
    cfg.apply_seed()
    context = build_matching_context(task_entry, candidate_id)
    backend = RuleBasedBackend(cfg)
    bilateral = BilateralSimulator(backend, cfg).run(context)
    outcome = OutcomeSimulator(cfg).simulate(context, bilateral)
    reward = compute_reward(bilateral, outcome, cfg)
    outcome_payload = {
        "mutual_accept_probability": round(float(bilateral.joint_accept_prob), 4),
        "completion_probability": round(float(outcome.completion_probability), 4),
        "requester_satisfaction": round(float(outcome.requester_satisfaction), 4),
        "candidate_satisfaction": round(float(outcome.candidate_satisfaction), 4),
        "joint_reward": round(float(reward.feedback_reward), 4),
        "total_reward": round(float(reward.total_reward), 4),
        "joint_action": bilateral.joint_action.value,
    }
    process = process_metrics_from_bilateral(bilateral)
    return outcome_payload, process


def _opinion_by_role(bilateral, role: str) -> Optional[float]:
    vals = []
    for decision in (bilateral.requester_decision, bilateral.candidate_decision):
        for opinion in decision.all_persona_opinions.values():
            if opinion.persona_role == role:
                vals.append(float(opinion.utility_score))
    if not vals:
        return None
    return float(np.mean(vals))


def _mean_existing(values: List[Optional[float]], default: float = 0.5) -> float:
    xs = [v for v in values if v is not None]
    return float(np.mean(xs)) if xs else default


def process_metrics_from_bilateral(bilateral) -> dict:
    time_feasibility = _mean_existing([
        _opinion_by_role(bilateral, "time_risk_evaluator"),
        _opinion_by_role(bilateral, "workload_evaluator"),
    ])
    style = _mean_existing([
        _opinion_by_role(bilateral, "collaboration_style_evaluator"),
    ])
    usability = _mean_existing([
        _opinion_by_role(bilateral, "capability_fit_evaluator"),
        _opinion_by_role(bilateral, "task_interest_evaluator"),
        _opinion_by_role(bilateral, "reciprocity_benefit_evaluator"),
    ])
    return {
        "time_feasibility": round(time_feasibility, 4),
        "collaboration_style_compatibility": round(style, 4),
        "candidate_usability": round(usability, 4),
    }


def evaluate_static_method(
    method: str,
    task_entry: dict,
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
    optimum_id: str,
    m_true_per_cand: Dict[str, float],
    m_optimum: float,
) -> dict:
    mode = "scap" if method == "scap_greedy" else "mapscore"
    matches = analytical_matches(requester, candidate_pool, task, mode)
    selected_id = select_top1(matches)
    outcome, process = compute_bilateral_outcome(task_entry, selected_id)
    rho = m_true_per_cand[selected_id] / m_optimum if m_optimum > 0 else 0.0
    return {
        "method": method,
        "selected_candidate_id": selected_id,
        "rho_last": round(float(rho), 4),
        "rho_mode": round(float(rho), 4),
        "selected_match": selected_match_metrics(matches[selected_id]),
        "outcome": outcome,
        "process": {
            **process,
            "average_messages_per_task": 0.0,
            "api_calls_per_task": 0.0,
            "tokens_per_task": 0.0,
            "wall_clock_time_per_task": 0.0,
        },
    }


def build_public_dream_profiles(
    task_entry: dict,
    requester_state: UserState,
    candidate_pool: List[UserState],
) -> Tuple[ExtendedProfile, Dict[str, ExtendedProfile], dict]:
    requester_ext = public_extended_profile(
        task_entry["proposer_profile"],
        requester_state,
        "requester",
    )
    candidate_entries_by_id = {
        c["candidate_profile"]["user_id"]: c for c in task_entry["candidates"]
    }
    candidate_exts = {
        state.user_id: public_extended_profile(
            candidate_entries_by_id[state.user_id]["candidate_profile"],
            state,
            "candidate",
        )
        for state in candidate_pool
    }
    public_payload = {
        "task": task_entry["task"],
        "requester_profile": task_entry["proposer_profile"],
        "candidate_profiles": [
            candidate_entries_by_id[state.user_id]["candidate_profile"]
            for state in candidate_pool
        ],
    }
    assert_public_dream_inputs(task_entry, public_payload)
    return requester_ext, candidate_exts, public_payload


def evaluate_mapscore_dreaming(
    task_entry: dict,
    requester: UserState,
    candidate_pool: List[UserState],
    task: Task,
    optimum_id: str,
    m_true_per_cand: Dict[str, float],
    m_optimum: float,
    *,
    top_k: int,
    seed: int,
    dreamer: CountingDreamSimulator,
    dreaming_enabled: bool = True,
) -> dict:
    matches = analytical_matches(requester, candidate_pool, task, "mapscore")
    analytical_selected_id = select_top1(matches)
    shortlist = analytical_top_k(matches, top_k)
    shortlist_ids = [m.candidate_id for m in shortlist]

    if not dreaming_enabled:
        selected_id = analytical_selected_id
        outcome, process = compute_bilateral_outcome(task_entry, selected_id)
        rho = m_true_per_cand[selected_id] / m_optimum if m_optimum > 0 else 0.0
        return {
            "method": "mapscore_dreaming",
            "seed": seed,
            "selected_candidate_id": selected_id,
            "analytical_top1_candidate_id": analytical_selected_id,
            "shortlist_candidate_ids": shortlist_ids,
            "reranked": False,
            "rho_last": round(float(rho), 4),
            "rho_mode": round(float(rho), 4),
            "selected_match": selected_match_metrics(matches[selected_id]),
            "outcome": outcome,
            "process": {
                **process,
                "average_messages_per_task": 0.0,
                "api_calls_per_task": 0.0,
                "tokens_per_task": 0.0,
                "wall_clock_time_per_task": 0.0,
            },
            "dreaming": {"enabled": False, "refined": []},
        }

    requester_ext, candidate_exts, _ = build_public_dream_profiles(
        task_entry,
        requester,
        candidate_pool,
    )
    dreamer.reset_counters()
    started = time.perf_counter()
    planner = PlanningLayer(
        world_model=WorldModel(config=CFG, theta_c=EVAL_THETA_C, theta_n=EVAL_THETA_N),
        dream_simulator=dreamer,
        top_k=top_k,
        top_n=top_k,
        analytical_weight=0.5,
        dream_weight=0.5,
    )
    refined = planner.run_from_layer3_output(
        requester=requester_ext,
        task=task,
        candidate_profiles=candidate_exts,
        layer3_output=shortlist,
        verbose=False,
    )
    wall = time.perf_counter() - started
    selected_id = refined[0].candidate_id
    outcome, process = compute_bilateral_outcome(task_entry, selected_id)
    rho = m_true_per_cand[selected_id] / m_optimum if m_optimum > 0 else 0.0
    messages = sum(len(r.transcript) for r in refined)
    analytical_top = matches[analytical_selected_id]
    dream_top = matches[selected_id]
    refined_payload = [
        {
            "candidate_id": r.candidate_id,
            "analytical_score": round(float(r.analytical_score), 4),
            "dream_score": round(float(r.dream_score), 4),
            "combined_score": round(float(r.combined_score), 4),
            "recommendation": r.recommendation,
            "top_risk": r.top_risk,
            "top_synergy": r.top_synergy,
            "compatibility": r.compatibility,
        }
        for r in refined
    ]
    return {
        "method": "mapscore_dreaming",
        "seed": seed,
        "selected_candidate_id": selected_id,
        "analytical_top1_candidate_id": analytical_selected_id,
        "shortlist_candidate_ids": shortlist_ids,
        "reranked": selected_id != analytical_selected_id,
        "rerank_reason": refined_payload[0]["top_synergy"] or refined_payload[0]["recommendation"],
        "delta_top1_s_cap": round(float(dream_top.s_cap - analytical_top.s_cap), 4),
        "delta_top1_s_need": round(float(dream_top.s_need - analytical_top.s_need), 4),
        "rho_last": round(float(rho), 4),
        "rho_mode": round(float(rho), 4),
        "selected_match": selected_match_metrics(matches[selected_id]),
        "outcome": outcome,
        "process": {
            **process,
            "average_messages_per_task": round(float(messages), 4),
            "api_calls_per_task": round(float(dreamer.api_calls), 4),
            "tokens_per_task": round(float(dreamer.approx_tokens), 4),
            "wall_clock_time_per_task": round(float(wall), 4),
            "api_errors_per_task": round(float(len(dreamer.api_errors)), 4),
        },
        "dreaming": {
            "enabled": True,
            "refined": refined_payload,
            "mock_mode": dreamer.mock_mode,
            "api_errors": list(dreamer.api_errors),
        },
    }


def evaluate_task(
    task_entry: dict,
    *,
    seeds: List[int],
    top_k: int,
    dreamer_factory,
    dreaming_enabled: bool,
) -> dict:
    task_dict = task_entry["task"]
    task = build_task(task_dict)
    requester = build_requester_for_match(task_entry["proposer_profile"])
    candidate_entries = task_entry["candidates"]
    candidate_pool = [build_learning_state_from_priors(c) for c in candidate_entries]
    m_true_per_cand, optimum_id, m_optimum = compute_ground_truth_M(
        requester,
        candidate_entries,
        task,
    )
    results = {
        "task_id": task_dict["task_id"],
        "title": task_dict.get("title", ""),
        "optimum_candidate_id": optimum_id,
        "M_optimum": round(float(m_optimum), 4),
        "conditions": {},
    }
    for method in ("scap_greedy", "mapscore_greedy"):
        results["conditions"][method] = evaluate_static_method(
            method,
            task_entry,
            requester,
            candidate_pool,
            task,
            optimum_id,
            m_true_per_cand,
            m_optimum,
        )
    dreaming_runs = []
    for seed in seeds:
        dreamer = dreamer_factory(seed)
        dreaming_runs.append(evaluate_mapscore_dreaming(
            task_entry,
            requester,
            candidate_pool,
            task,
            optimum_id,
            m_true_per_cand,
            m_optimum,
            top_k=top_k,
            seed=seed,
            dreamer=dreamer,
            dreaming_enabled=dreaming_enabled,
        ))
    results["conditions"]["mapscore_dreaming"] = dreaming_runs
    return results


def _values_for_condition(per_task: List[dict], method: str, field_path: List[str]) -> List[float]:
    vals = []
    for task_result in per_task:
        cond = task_result["conditions"][method]
        if method == "mapscore_dreaming":
            for run in cond:
                node = run
                for key in field_path:
                    node = node[key]
                vals.append(float(node))
        else:
            node = cond
            for key in field_path:
                node = node[key]
            vals.append(float(node))
    return vals


def _summary_stats(vals: List[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(vals)), 4),
        "std": round(float(np.std(vals)), 4),
        "median": round(float(np.median(vals)), 4),
        "min": round(float(np.min(vals)), 4),
        "max": round(float(np.max(vals)), 4),
    }


def _rank_corr(xs: List[float], ys: List[float]) -> float:
    def ranks(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            r = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = r
            i = j + 1
        return out
    return _pearson(ranks(xs), ranks(ys))


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return 0.0
    return round(float(np.corrcoef(np.array(xs), np.array(ys))[0, 1]), 4)


def aggregate_results(per_task: List[dict]) -> dict:
    summary: Dict[str, Any] = {}
    for method in ("scap_greedy", "mapscore_greedy", "mapscore_dreaming"):
        method_summary = {
            "rho_last": _summary_stats(_values_for_condition(per_task, method, ["rho_last"])),
            "rho_mode": _summary_stats(_values_for_condition(per_task, method, ["rho_mode"])),
        }
        for field in ("S_cap", "S_need", "MapScore"):
            method_summary[field] = _summary_stats(
                _values_for_condition(per_task, method, ["selected_match", field])
            )
        method_summary["outcome"] = {
            field: _summary_stats(_values_for_condition(per_task, method, ["outcome", field]))
            for field in OUTCOME_FIELDS
        }
        method_summary["process"] = {
            field: _summary_stats(_values_for_condition(per_task, method, ["process", field]))
            for field in PROCESS_FIELDS
        }
        if method == "mapscore_dreaming":
            reranks = _values_for_condition(per_task, method, ["reranked"])
            method_summary["rerank_rate"] = _summary_stats(reranks)
            method_summary["delta_top1_s_cap"] = _summary_stats(
                _values_for_condition(per_task, method, ["delta_top1_s_cap"])
            )
            method_summary["delta_top1_s_need"] = _summary_stats(
                _values_for_condition(per_task, method, ["delta_top1_s_need"])
            )
        summary[method] = method_summary

    dream_scores = []
    corr_targets = {field: [] for field in OUTCOME_FIELDS}
    for task_result in per_task:
        for run in task_result["conditions"]["mapscore_dreaming"]:
            refined = run.get("dreaming", {}).get("refined", [])
            if refined:
                dream_scores.append(float(refined[0]["dream_score"]))
                for field in OUTCOME_FIELDS:
                    corr_targets[field].append(float(run["outcome"][field]))
    summary["dreaming_correlations"] = {
        field: {
            "pearson": _pearson(dream_scores, vals),
            "spearman": _rank_corr(dream_scores, vals),
        }
        for field, vals in corr_targets.items()
    }
    return summary


def write_markdown_summary(payload: dict, out_path: Path) -> None:
    summary = payload["summary"]
    baseline = summary["mapscore_greedy"]
    lines = [
        "# V3 Dreaming Ablation Summary",
        "",
        f"Generated at: `{payload['metadata']['generated_at']}`",
        f"Testset: `{payload['metadata']['testset']}`",
        f"Dreaming mode: `{payload['metadata']['dreaming_mode']}`",
        "",
        "## Matching Metrics",
        "",
        "| method | rho_last | delta vs MapScore | S_cap | S_need | MapScore |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("scap_greedy", "mapscore_greedy", "mapscore_dreaming"):
        row = summary[method]
        delta = row["rho_last"]["mean"] - baseline["rho_last"]["mean"]
        lines.append(
            f"| {method} | {row['rho_last']['mean']:.4f} ± {row['rho_last']['std']:.4f} "
            f"| {delta:+.4f} | {row['S_cap']['mean']:.4f} | "
            f"{row['S_need']['mean']:.4f} | {row['MapScore']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Outcome Metrics",
        "",
        "| method | mutual | completion | req_sat | cand_sat | joint_reward | total_reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for method in ("scap_greedy", "mapscore_greedy", "mapscore_dreaming"):
        out = summary[method]["outcome"]
        base_out = baseline["outcome"]
        def fmt_out(field: str) -> str:
            mean = out[field]["mean"]
            std = out[field]["std"]
            delta = mean - base_out[field]["mean"]
            return f"{mean:.4f} ± {std:.4f} ({delta:+.4f})"
        lines.append(
            f"| {method} | {fmt_out('mutual_accept_probability')} | "
            f"{fmt_out('completion_probability')} | "
            f"{fmt_out('requester_satisfaction')} | "
            f"{fmt_out('candidate_satisfaction')} | "
            f"{fmt_out('joint_reward')} | {fmt_out('total_reward')} |"
        )
    lines.extend([
        "",
        "Values are mean ± std; parentheses show delta relative to `mapscore_greedy`.",
        "",
        "## Process Metrics",
        "",
        "| method | time feasibility | style compatibility | candidate usability | messages/task | API calls/task | tokens/task | wall-clock/task |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in ("scap_greedy", "mapscore_greedy", "mapscore_dreaming"):
        proc = summary[method]["process"]
        base_proc = baseline["process"]
        def fmt_proc(field: str) -> str:
            mean = proc[field]["mean"]
            std = proc[field]["std"]
            delta = mean - base_proc[field]["mean"]
            return f"{mean:.4f} ± {std:.4f} ({delta:+.4f})"
        lines.append(
            f"| {method} | {fmt_proc('time_feasibility')} | "
            f"{fmt_proc('collaboration_style_compatibility')} | "
            f"{fmt_proc('candidate_usability')} | "
            f"{fmt_proc('average_messages_per_task')} | "
            f"{fmt_proc('api_calls_per_task')} | "
            f"{fmt_proc('tokens_per_task')} | "
            f"{fmt_proc('wall_clock_time_per_task')} |"
        )
    lines.extend([
        "",
        "## Dreaming Diagnostics",
        "",
        f"- rerank_rate: {summary['mapscore_dreaming'].get('rerank_rate', {}).get('mean', 0.0):.4f}",
        f"- delta_top1_s_cap: {summary['mapscore_dreaming'].get('delta_top1_s_cap', {}).get('mean', 0.0):+.4f}",
        f"- delta_top1_s_need: {summary['mapscore_dreaming'].get('delta_top1_s_need', {}).get('mean', 0.0):+.4f}",
        "",
        "Dream score correlations with outcomes:",
    ])
    for field, stats in summary["dreaming_correlations"].items():
        lines.append(f"- {field}: Pearson={stats['pearson']:.4f}, Spearman={stats['spearman']:.4f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--dreaming-mode", choices=("mock", "api", "disabled"), default="mock")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--n-turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument(
        "--partial-output",
        type=Path,
        default=None,
        help="Optional JSON path updated after each task for long API runs.",
    )
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="If --partial-output exists, load completed tasks and continue from the next task.",
    )
    args = parser.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    data = json.loads(args.testset.read_text(encoding="utf-8"))
    tasks = data["tasks"][: args.max_tasks] if args.max_tasks and args.max_tasks > 0 else data["tasks"]
    dreaming_enabled = args.dreaming_mode != "disabled"

    import LLM_Dreaming.LLM_dreaming as dreaming_module
    original_api_key = dreaming_module.API_KEY
    if args.dreaming_mode in {"mock", "disabled"}:
        dreaming_module.API_KEY = ""

    def dreamer_factory(seed: int) -> CountingDreamSimulator:
        return CountingDreamSimulator(
            n_turns=args.n_turns,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            timeout=args.api_timeout,
            seed=seed,
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    per_task = []
    completed_task_ids = set()
    if args.resume_partial and args.partial_output is not None and args.partial_output.exists():
        partial = json.loads(args.partial_output.read_text(encoding="utf-8"))
        per_task = list(partial.get("per_task", []))
        completed_task_ids = {r["task_id"] for r in per_task}
        print(
            f"Resuming from {args.partial_output}: {len(completed_task_ids)} completed tasks.",
            flush=True,
        )
    try:
        for idx, task_entry in enumerate(tasks, start=1):
            if task_entry["task"]["task_id"] in completed_task_ids:
                continue
            result = evaluate_task(
                task_entry,
                seeds=seeds,
                top_k=args.top_k,
                dreamer_factory=dreamer_factory,
                dreaming_enabled=dreaming_enabled,
            )
            per_task.append(result)
            ms = result["conditions"]["mapscore_greedy"]["selected_candidate_id"]
            dr = [r["selected_candidate_id"] for r in result["conditions"]["mapscore_dreaming"]]
            print(f"[{idx:>2}/{len(tasks)}] {result['task_id']} mapscore={ms} dreaming={dr}", flush=True)
            if args.partial_output is not None:
                args.partial_output.parent.mkdir(parents=True, exist_ok=True)
                partial_payload = {
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "testset": str(args.testset),
                        "dreaming_mode": args.dreaming_mode,
                        "model": args.model,
                        "base_url": args.base_url,
                        "top_k": args.top_k,
                        "seeds": seeds,
                        "n_turns": args.n_turns,
                        "temperature": args.temperature,
                        "num_tasks_completed": len(per_task),
                        "num_tasks_target": len(tasks),
                        "partial": True,
                        "api_key_logged": False,
                    },
                    "per_task": per_task,
                }
                args.partial_output.write_text(
                    json.dumps(partial_payload, indent=2),
                    encoding="utf-8",
                )
    finally:
        dreaming_module.API_KEY = original_api_key

    summary = aggregate_results(per_task)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"v3_dreaming_ablation_{args.dreaming_mode}_n{len(tasks)}"
    out_json = LOGS_DIR / f"{tag}_{timestamp}.json"
    out_md = EXPERIMENTS_DIR / f"{tag}_{timestamp}.md"
    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "testset": str(args.testset),
            "dreaming_mode": args.dreaming_mode,
            "model": args.model,
            "base_url": args.base_url,
            "top_k": args.top_k,
            "seeds": seeds,
            "n_turns": args.n_turns,
            "temperature": args.temperature,
            "num_tasks": len(tasks),
            "api_key_logged": False,
        },
        "per_task": per_task,
        "summary": summary,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_summary(payload, out_md)

    print()
    print(f"Wrote JSON log to {out_json}")
    print(f"Wrote Markdown summary to {out_md}")
    for method in ("scap_greedy", "mapscore_greedy", "mapscore_dreaming"):
        s = summary[method]
        print(
            f"{method:<18} rho={s['rho_last']['mean']:.3f}±{s['rho_last']['std']:.3f} "
            f"mutual={s['outcome']['mutual_accept_probability']['mean']:.3f} "
            f"completion={s['outcome']['completion_probability']['mean']:.3f} "
            f"total={s['outcome']['total_reward']['mean']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
