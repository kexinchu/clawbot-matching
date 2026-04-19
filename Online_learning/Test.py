"""
Test script: track μ, σ, weights, and reward convergence with wandb.
Usage: python test_layer5_wandb.py
"""

import numpy as np
import wandb
from Online_learning import OnlineLearning
from UserProfile import UserProfile
from Task import Task
from WorldModel import WorldModel
from Reward_function import RewardFunction
from Parameter_update import BayesianUpdater, WeightUpdater, UCBExplorer
from typing import List, Optional
from utils import dummy_user_feedback, dummy_create_profile, dummy_create_task

np.random.seed(42)

key = "600e5cca820a9fbb7580d052801b3acfd5c92da2"
wandb.login(key=key)
# --- Init wandb (offline mode, no API key needed) ---
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
bob = dummy_create_profile("bob", "candidate")
task = dummy_create_task()
world_model = WorldModel(theta_c=0.4, theta_n=-0.1)
engine = OnlineLearning(world_model)

# --- Run 50 rounds ---
for i in range(wandb.config.n_rounds):
    report = engine.run_one_round(alice, bob, task)

    # Log all tracked metrics
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
        "mu/bayesian":      bob.capability["bayesian"].mu,
        "mu/python":        bob.capability["python"].mu,
        "mu/paper_writing": bob.capability["paper_writing"].mu,

        # Bob capability σ
        "sigma/bayesian":      bob.capability["bayesian"].sigma,
        "sigma/python":        bob.capability["python"].sigma,
        "sigma/paper_writing": bob.capability["paper_writing"].sigma,

        # Weights
        "weights/w_c": world_model.w_c,
        "weights/w_n": world_model.w_n,

        # UCB
        "ucb/beta_t": report["beta_t"],
        "ucb/bonus_bayesian":      engine.ucb_explorer.beta * bob.capability["bayesian"].sigma,
        "ucb/bonus_python":        engine.ucb_explorer.beta * bob.capability["python"].sigma,
        "ucb/bonus_paper_writing": engine.ucb_explorer.beta * bob.capability["paper_writing"].sigma,

        # SGD loss
        "loss": report["path2_weights"]["loss"],
    })

wandb.finish()
print("\nDone. Run `wandb sync wandb/latest-run` to upload, or view locally.")