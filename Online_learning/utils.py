from UserProfile import UserProfile
from Capability import Capability
from Task import Task
from MatchResult import MatchResult
import numpy as np

def dummy_create_profile(user_id: str, role: str) -> UserProfile:
    """
    In production: LLM extracts (Cap, Need) from dialogue.
    Here: generate synthetic profiles for testing.
    """
    profiles = {
        "alice": UserProfile(
            user_id="alice",
            capability={
                "bayesian":     Capability(mu=0.3, sigma=0.2),
                "python":       Capability(mu=0.8, sigma=0.1),
                "paper_writing": Capability(mu=0.4, sigma=0.3),
            },
            need={"bayesian": 0.8, "python": 0.1, "paper_writing": 0.7}
        ),
        "bob": UserProfile(
            user_id="bob",
            capability={
                "bayesian":     Capability(mu=0.9, sigma=0.1),
                "python":       Capability(mu=0.5, sigma=0.2),
                "paper_writing": Capability(mu=0.8, sigma=0.1),
            },
            need={"bayesian": 0.2, "python": 0.7, "paper_writing": 0.9}
        ),
        "carol": UserProfile(
            user_id="carol",
            capability={
                "bayesian":     Capability(mu=0.6, sigma=0.4),   # new user, high uncertainty
                "python":       Capability(mu=0.7, sigma=0.35),
                "paper_writing": Capability(mu=0.5, sigma=0.4),
            },
            need={"bayesian": 0.5, "python": 0.3, "paper_writing": 0.8}
        ),
    }
    return profiles.get(user_id)
 
 
def dummy_create_task() -> Task:
    """Synthetic task for testing."""
    return Task(
        task_id="task_001",
        goal="Build Bayesian churn model, target NeurIPS",
        Q_T={"bayesian": 0.8, "python": 0.6, "paper_writing": 0.7}
    )

def dummy_user_feedback(match_result: MatchResult) -> dict:
    """
    Simulate user actions and collaboration outcome.
    In production: collected from real user interactions.
 
    Returns raw data that the Reward Function will process.
    """
    # Higher match score → higher chance of acceptance
    noise = np.random.normal(0, 0.05)
    acceptance_prob = np.clip(match_result.M + noise, 0, 1)
 
    # r_u, r_v: simulate both sides
    r_u = 1.0 if acceptance_prob > 0.6 else (0.7 if acceptance_prob > 0.4 else 0.3)
    r_v = 1.0 if acceptance_prob > 0.5 else (0.7 if acceptance_prob > 0.3 else 0.3)
 
    # n_rounds: better matches negotiate faster
    n_rounds = max(3, int(3 + (1 - match_result.M) * 25 + np.random.randint(-2, 3)))
 
    # completion: probabilistic based on match quality
    completion_prob = match_result.M * 0.9 + 0.1
    if np.random.random() < completion_prob:
        f_completion = 1.0
    elif np.random.random() < 0.5:
        f_completion = 0.5
    else:
        f_completion = 0.0
 
    # mutual rating: correlated with match quality + noise
    stars_u = np.clip(int(match_result.M * 5 + np.random.normal(0, 0.5) + 0.5), 1, 5)
    stars_v = np.clip(int(match_result.M * 5 + np.random.normal(0, 0.5) + 0.5), 1, 5)
 
    return {
        "r_u": r_u,
        "r_v": r_v,
        "n_rounds": n_rounds,
        "f_completion": f_completion,
        "stars_u": stars_u,
        "stars_v": stars_v,
    }