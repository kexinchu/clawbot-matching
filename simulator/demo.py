"""
simulator/demo.py
================
End-to-end demo showing the full bilateral matching simulator pipeline.

Run with:
    python -m simulator.demo
"""
from __future__ import annotations

import pprint
import sys

from simulator import (
    BilateralSimulator,
    MatchingContext,
    OutcomeSimulator,
    RuleBasedBackend,
    SimulatorConfig,
    TaskSpec,
    UserProfile,
    CandidateCard,
    compute_reward,
    TraceLogger,
    create_backend,
)


def make_demo_context() -> MatchingContext:
    """Build a realistic demo matching context."""
    requester = UserProfile(
        user_id="req_alpha",
        role="project_manager",
        capabilities={
            "project_management": 0.8,
            "communication": 0.9,
            "agile": 0.7,
        },
        needs={
            "technical_execution": 0.9,
            "deadline_reliability": 0.8,
            "collaboration": 0.7,
        },
        preferences={
            "communication_style": "async_first",
            "timezone": "US_Eastern",
        },
        constraints={},
        history_summary="Has run 12 successful projects; no disputes.",
    )

    candidate = UserProfile(
        user_id="cand_beta",
        role="software_engineer",
        capabilities={
            "python": 0.95,
            "ml_systems": 0.85,
            "data_engineering": 0.80,
            "communication": 0.65,
        },
        needs={
            "career_growth": 0.6,
            "interesting_project": 0.7,
        },
        preferences={
            "interests": "ml_systems python data pipeline optimization",
            "availability": "high",
            "current_load": 0.3,
        },
        constraints={"workload": 0.3},
        history_summary="5 years experience; 8 completed freelance projects.",
    )

    task = TaskSpec(
        task_id="task_001",
        title="Build ML Data Pipeline for Recommendation Engine",
        description=(
            "Design and implement an end-to-end ML pipeline that processes user "
            "interaction logs, trains a recommendation model, and deploys it as "
            "an API. Requires Python, ML systems knowledge, and data engineering."
        ),
        required_skills={
            "python": 0.8,
            "ml_systems": 0.7,
            "data_engineering": 0.7,
        },
        metadata={
            "urgency": "high",
            "budget": "competitive",
            "visibility": "high",
            "portfolio_boost": True,
            "duration_weeks": 6,
        },
    )

    card = CandidateCard(
        candidate_id="cand_beta",
        summary=(
            "Experienced software engineer specializing in ML systems and "
            "data pipelines. Delivered 8 successful projects with strong Python skills."
        ),
        highlighted_strengths=[
            "Strong Python and ML systems background",
            "Demonstrated experience with recommendation systems",
            "Portfolio of open-source data pipeline projects",
        ],
        highlighted_risks=[
            "Currently has 3 active projects — bandwidth may be constrained",
            "Availability for urgent meetings is limited",
        ],
        explanation=(
            "Cand_beta has relevant technical skills and past experience "
            "with similar systems. However, their current workload and limited "
            "availability for meetings introduces some risk to the timeline."
        ),
    )

    return MatchingContext(
        requester=requester,
        candidate=candidate,
        task=task,
        card=card,
        history={
            "prior_collaboration": False,
            "prior_skipped": False,
            "dispute_rate": 0.0,
        },
        # ── Latent signals: these are Layer-2-invisible but affect soft personas ──
        latent_requester_preferences={"structured": 0.9, "fastpaced": 0.3},
        latent_candidate_preferences={"creative": 0.8, "ml_systems": 0.9},
        latent_interpersonal_affinity=0.4,   # positive hidden chemistry
        latent_risk_tolerance=0.6,          # moderate risk appetite
        latent_opportunity_bias=0.2,         # slight optimistic bias
    )


def main() -> None:
    print("=" * 70)
    print("  BILATERAL MATCHING SIMULATOR — DEMO")
    print("=" * 70)
    print()

    # ── Setup ──────────────────────────────────────────────────────────────
    # backend_type options:
    #   "mock"    — rule-based, no API key needed
    #   "openai"  — real LLM via OpenRouter (requires OPENROUTER_API_KEY in ~/.bashrc)
    config = SimulatorConfig(
        backend_type="openai",
        model_name="deepseek/deepseek-chat-v3-0324",
        random_seed=42,
        trace_verbose=True,
        num_requester_personas=4,
        num_candidate_personas=4,
        max_deliberation_rounds=3,
        top_k_personas=3,
        early_stop_threshold=0.70,
    )
    config.apply_seed()

    backend = create_backend(config)
    logger = TraceLogger(verbose=True)
    sim = BilateralSimulator(backend, config, logger)

    # ── Build context ───────────────────────────────────────────────────────
    context = make_demo_context()

    print("── Matching Context ──")
    print(f"  Requester: {context.requester.user_id} ({context.requester.role})")
    print(f"  Candidate:  {context.candidate.user_id} ({context.candidate.role})")
    print(f"  Task:       {context.task.title}")
    print()

    # ── Run bilateral simulation ─────────────────────────────────────────────
    bilateral = sim.run(context)

    print()
    print("── Side Decisions ──")
    print(f"  Requester → action={bilateral.requester_decision.action.value}, "
          f"utility={bilateral.requester_decision.utility:.3f}, "
          f"confidence={bilateral.requester_decision.confidence:.3f}")
    print(f"  Candidate → action={bilateral.candidate_decision.action.value}, "
          f"utility={bilateral.candidate_decision.utility:.3f}, "
          f"confidence={bilateral.candidate_decision.confidence:.3f}")
    print()
    print(f"  Joint Action: {bilateral.joint_action.value}")
    print(f"  Joint Accept Prob: {bilateral.joint_accept_prob:.3f}")
    print()

    # ── Outcome simulation ───────────────────────────────────────────────────
    outcome_sim = OutcomeSimulator(config)
    outcome = outcome_sim.simulate(context, bilateral)

    print("── Outcome Simulation ──")
    print(f"  Agreement probability:  {outcome.agreement_probability:.1%}")
    print(f"  Expected rounds:        ~{outcome.expected_rounds:.1f}")
    print(f"  Completion probability:  {outcome.completion_probability:.1%}")
    print(f"  Requester satisfaction: {outcome.requester_satisfaction:.1%}")
    print(f"  Candidate satisfaction:  {outcome.candidate_satisfaction:.1%}")
    print()

    # ── Reward construction ───────────────────────────────────────────────────
    reward = compute_reward(bilateral, outcome, config)

    print("── Reward Result ──")
    print(f"  Feedback reward:   {reward.feedback_reward:+.3f}")
    print(f"  Efficiency reward:{reward.efficiency_reward:+.3f}")
    print(f"  Quality reward:    {reward.quality_reward:+.3f}")
    print(f"  ──────────────────────────────")
    print(f"  TOTAL REWARD:      {reward.total_reward:+.3f}")
    print()

    # ── Full trace dump ──────────────────────────────────────────────────────
    print("── Full Trace (JSON) ──")
    pprint.pprint(logger.entries, width=100)

    print()
    print("=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
