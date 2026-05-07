"""
Test script: track μ, σ, weights, and reward convergence with wandb.
Usage: python Test.py
"""

import sys
import os
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mapping-algo"); sys.path.append(_p) if _p not in sys.path else None  # os.path.join(os.path.dirname(__file__), '..', 'mapping-algo'))

import numpy as np
import wandb
from Online_learning import OnlineLearning
from WorldModel import WorldModel
from ol_utils import dummy_user_feedback, dummy_create_profile, dummy_create_task, find_cap

np.random.seed(42)

key = "600e5cca820a9fbb7580d052801b3acfd5c92da2"
wandb.login(key=key)

wandb.init(
    project="mbrl-matching",
    name="layer5-convergence-test",
    config={
        "n_rounds": 50,
        "lambda1": 0.5,
        "lambda2": 1.0,
        "lr": 0.01,
        "sigma_base": 0.3,
        "theta_init": [0.4, -0.1],
    },
)

# --- Setup ---
alice = dummy_create_profile("alice", "requester")
bob   = dummy_create_profile("bob",   "candidate")
task  = dummy_create_task()
world_model = WorldModel(theta_c=0.4, theta_n=-0.1)
engine = OnlineLearning(world_model)

# --- Run 50 rounds ---
for i in range(wandb.config.n_rounds):
    report = engine.run_one_round(alice, bob, task)

    bob_bayesian     = find_cap(bob, "bayesian")
    bob_python       = find_cap(bob, "python")
    bob_paper        = find_cap(bob, "paper_writing")

    wandb.log({
        "round": report["round"],

        # Reward signals
        "reward/R":            report["reward"]["R"],
        "reward/r_feedback":   report["reward"]["r_feedback"],
        "reward/r_efficiency": report["reward"]["r_efficiency"],
        "reward/r_quality":    report["reward"]["r_quality"],

        # Match score
        "match/M":      report["match_score"]["M (no UCB)"],
        "match/M_ucb":  report["match_score"]["M (with UCB)"],
        "match/S_cap":  report["match_score"]["S_cap"],
        "match/S_need": report["match_score"]["S_need"],

        # Bob capability μ
        "mu/bayesian":      bob_bayesian.mu      if bob_bayesian  else None,
        "mu/python":        bob_python.mu        if bob_python    else None,
        "mu/paper_writing": bob_paper.mu         if bob_paper     else None,

        # Bob capability σ
        "sigma/bayesian":      bob_bayesian.sigma   if bob_bayesian else None,
        "sigma/python":        bob_python.sigma     if bob_python   else None,
        "sigma/paper_writing": bob_paper.sigma      if bob_paper    else None,

        # Weights
        "weights/w_c": world_model.w_c,
        "weights/w_n": world_model.w_n,

        # UCB
        "ucb/beta_t": report["beta_t"],
        "ucb/bonus_bayesian":      engine.ucb_explorer.beta * (bob_bayesian.sigma  if bob_bayesian else 0),
        "ucb/bonus_python":        engine.ucb_explorer.beta * (bob_python.sigma    if bob_python   else 0),
        "ucb/bonus_paper_writing": engine.ucb_explorer.beta * (bob_paper.sigma     if bob_paper    else 0),

        # SGD loss
        "loss": report["path2_weights"]["loss"],
    })

wandb.finish()
print("\nDone. Run `wandb sync wandb/latest-run` to upload, or view locally.")
