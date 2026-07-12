from simulator.bilateral_simulator import BilateralSimulator
from simulator.config import SimulatorConfig
from simulator.mock_backend import RuleBasedBackend
from simulator.outcome_simulator import OutcomeSimulator
from simulator.types import CandidateCard, MatchingContext, TaskSpec, UserProfile


def _context(offer_strength: float) -> MatchingContext:
    requester = UserProfile(
        user_id="requester",
        role="requester",
        capabilities={},
        needs={},
        preferences={"communication_style": "mixed"},
        constraints={},
    )
    candidate = UserProfile(
        user_id="candidate",
        role="engineer",
        capabilities={"python": 0.9},
        needs={"publication mentorship": 1.0},
        preferences={"availability": "high", "current_load": 0.1, "interests": "python research"},
        constraints={"workload": 0.1},
    )
    task = TaskSpec(
        task_id="offer_need_monotonic",
        title="Offer Need Monotonic",
        description="Python research project",
        required_skills={"python": 0.7},
        offers={
            "publication mentorship": offer_strength,
            "low visibility maintenance": 0.1,
        },
        metadata={"visibility": "high", "budget": "premium", "urgency": "medium"},
    )
    return MatchingContext(
        requester=requester,
        candidate=candidate,
        task=task,
        card=CandidateCard(
            candidate_id="candidate",
            summary="strong candidate",
            highlighted_strengths=["python"],
            highlighted_risks=[],
        ),
        history={"prior_collaboration": False, "prior_skipped": False, "dispute_rate": 0.0},
        latent_requester_preferences={"structured": 0.5},
        latent_candidate_preferences={"creative": 0.5},
        latent_interpersonal_affinity=0.2,
        latent_risk_tolerance=0.7,
        latent_opportunity_bias=0.1,
    )


def _run(context: MatchingContext):
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
    backend = RuleBasedBackend(cfg)
    bilateral = BilateralSimulator(backend, cfg).run(context)
    outcome = OutcomeSimulator(cfg).simulate(context, bilateral)
    return bilateral, outcome


def test_offer_need_fit_monotonically_increases_candidate_outcomes():
    low_bilateral, low_outcome = _run(_context(offer_strength=0.1))
    high_bilateral, high_outcome = _run(_context(offer_strength=0.9))

    assert high_outcome.candidate_satisfaction > low_outcome.candidate_satisfaction
    assert high_bilateral.joint_accept_prob > low_bilateral.joint_accept_prob
