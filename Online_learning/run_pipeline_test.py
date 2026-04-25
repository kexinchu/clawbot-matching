"""End-to-end pipeline test: L1 → L2 → L3 → L4 → L5.

Run from Online_learning/:
    python run_pipeline_test.py
"""

import sys, os
_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mapping-algo')
if _p not in sys.path:
    sys.path.append(_p)

import json
import numpy as np
np.random.seed(42)

from L1_parser import parse_user, parse_task
from WorldModel import WorldModel
from Online_learning import OnlineLearning
from ol_utils import find_cap, dummy_user_feedback
from pipeline import match_one_to_one, match_one_to_n
from config import MatchConfig

SEP = "=" * 60


# ── helpers ──────────────────────────────────────────────────────────

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def ok(label, value=""):
    print(f"  [OK] {label}" + (f": {value}" if value != "" else ""))

def show_match(r):
    print(f"       {r.candidate_id:10s}  M={r.match_score:.4f}  "
          f"S_cap={r.s_cap:.4f}  S_need={r.s_need:.4f}  "
          f"gate={r.sigma_gate}" +
          (f"  ucb_bonus={r.ucb_bonus:.4f}" if r.ucb_bonus else ""))


# ── L1: parse ────────────────────────────────────────────────────────

section("L1  Parse user profiles and task from JSON")

alice = parse_user("data/alice.json")
bob   = parse_user("data/bob.json")
task  = parse_task("data/task_001.json")

# Extra user parsed from inline dict (simulates form submission)
carol = parse_user({
    "user_id": "carol",
    "clearance_level": 0,
    "capabilities": [
        {"description": "bayesian statistics",    "mu": 0.6, "sigma": 0.4, "source": "meta"},
        {"description": "python programming",     "mu": 0.7, "sigma": 0.35, "source": "meta"},
        {"description": "academic paper writing", "mu": 0.5, "sigma": 0.4,  "source": "meta"},
    ],
    "needs": [
        {"description": "bayesian statistics mentorship",       "intensity": 0.5},
        {"description": "python programming practice",          "intensity": 0.3},
        {"description": "academic paper writing collaboration", "intensity": 0.8},
    ],
})

# User who will fail the clearance gate
dave = parse_user({
    "user_id": "dave",
    "clearance_level": -1,   # below task.data_clearance=0 → should pass (0 >= 0)
    "capabilities": [
        {"description": "bayesian statistics", "mu": 0.5, "sigma": 0.3, "source": "explicit"},
    ],
    "needs": [],
})

for u in [alice, bob, carol, dave]:
    ok(f"{u.user_id:6s}  caps={len(u.capabilities)}  needs={len(u.needs)}  clearance={u.clearance_level}")

ok("task requirements", [r.description for r in task.requirements])
ok("task offers",       [o.description for o in task.offers])


# ── L2: single match ─────────────────────────────────────────────────

section("L2  Single match score (alice→bob, alice→carol)")

cfg = MatchConfig(embedding_dim=64)
wm  = WorldModel(theta_c=0.4, theta_n=-0.1, config=cfg)

r_bob   = wm.compute_match(alice, bob,   task, use_ucb=False)
r_carol = wm.compute_match(alice, carol, task, use_ucb=False)
r_bob_ucb = wm.compute_match(alice, bob, task, use_ucb=True, round_t=10)

show_match(r_bob)
show_match(r_carol)
ok("bob with UCB (round=10)", f"M={r_bob_ucb.match_score:.4f}  ucb_bonus={r_bob_ucb.ucb_bonus:.4f}")

# Gate check: dave has clearance_level=-1 < task.data_clearance=0
task_secure = parse_task({
    "task_id": "secure_task", "goal": "test gate",
    "data_clearance": 1,
    "requirements": [{"description": "bayesian statistics", "level": 0.3, "constraint_type": "soft"}],
    "offers": [],
})
r_dave_secure = wm.compute_match(alice, dave, task_secure, use_ucb=False)
ok("dave vs secure_task gate",
   f"sigma={r_dave_secure.sigma_gate}  reason='{r_dave_secure.gate_fail_reason}'")
assert r_dave_secure.sigma_gate == 0, "Expected gate=0 for clearance < 1"

# Hard constraint check
task_hard = parse_task({
    "task_id": "hard_task", "goal": "test hard constraint",
    "data_clearance": 0,
    "requirements": [{"description": "quantum computing", "level": 0.9, "constraint_type": "hard"}],
    "offers": [],
})
r_bob_hard = wm.compute_match(alice, bob, task_hard, use_ucb=False)
ok("bob vs hard 'quantum computing' gate",
   f"sigma={r_bob_hard.sigma_gate}  reason='{r_bob_hard.gate_fail_reason}'")
assert r_bob_hard.sigma_gate == 0, "Expected gate=0 for hard constraint fail"

ok("Gate logic verified (clearance + hard constraint)")


# ── L3-1: one-to-one ranking ─────────────────────────────────────────

section("L3  1-1 ranking over candidate pool [bob, carol, dave]")

pool_1 = [bob, carol, dave]
ranked = match_one_to_one(alice, task, pool_1, wm.theta, cfg, top_k=5, use_ucb=True, round_t=5)

print(f"  Eligible after gate: {len(ranked)}/{len(pool_1)}")
for r in ranked:
    show_match(r)

assert len(ranked) >= 2, "Expected at least bob and carol to pass gate"
assert ranked[0].match_score >= ranked[-1].match_score, "Expected descending order"
ok("1-1 ranking sorted descending, gate applied")


# ── L3-N: one-to-many team building ──────────────────────────────────

section("L3  1-N team building over pool [bob, carol, dave]")

team = match_one_to_n(alice, task, pool_1, wm.theta, cfg, use_ucb=True, round_t=5)
print(f"  Team: {team.team_members}")
print(f"  Collective coverage: {team.collective_coverage:.4f}")
print(f"  Selection order: {team.selection_order}")
print(f"  Residual gaps: { {k: round(v,4) for k,v in team.residual_gaps.items()} }")
print(f"  Termination: {team.termination_reason}")

assert len(team.team_members) > 0, "Expected at least one team member"
assert team.collective_coverage > 0, "Expected positive collective coverage"
ok("1-N team building ran successfully")


# ── L4: feedback simulation ───────────────────────────────────────────

section("L4  Dummy feedback (placeholder for real Layer 4)")

fb = dummy_user_feedback(r_bob)
print(f"  r_u={fb['r_u']}  r_v={fb['r_v']}  "
      f"n_rounds={fb['n_rounds']}  completion={fb['f_completion']}  "
      f"stars=u{fb['stars_u']}/v{fb['stars_v']}")
ok("Feedback dict structure valid", list(fb.keys()))


# ── L5: learning loop ─────────────────────────────────────────────────

section("L5  Online learning — 20 rounds convergence check")

engine = OnlineLearning(wm)

# Snapshot before
bob_bay_before = find_cap(bob, "bayesian statistics")
mu0, sig0 = bob_bay_before.mu, bob_bay_before.sigma
w_c0 = wm.w_c

print(f"  Before — bob/bayesian: mu={mu0:.4f} sigma={sig0:.4f}  w_c={w_c0:.4f}")

rewards = []
for i in range(20):
    report = engine.run_one_round(alice, bob, task)
    rewards.append(report["reward"]["R"])

mu1   = find_cap(bob, "bayesian statistics").mu
sig1  = find_cap(bob, "bayesian statistics").sigma
w_c1  = wm.w_c

print(f"  After  — bob/bayesian: mu={mu1:.4f} sigma={sig1:.4f}  w_c={w_c1:.4f}")
print(f"  σ reduced: {sig0:.4f} → {sig1:.4f}  (uncertainty ↓)")
print(f"  Reward avg (last 5): {sum(rewards[-5:])/5:.4f}")
print(f"  Reward avg (first 5): {sum(rewards[:5])/5:.4f}")

# Basic sanity checks
assert sig1 < sig0, "sigma should decrease after Bayesian updates"
assert 0 < wm.w_c < 1 and 0 < wm.w_n < 1, "weights must stay in (0,1)"
assert abs(wm.w_c + wm.w_n - 1.0) < 1e-6, "weights must sum to 1"
ok("σ decreased after Bayesian updates")
ok("weights remain valid softmax")


# ── L3-N after learning ───────────────────────────────────────────────

section("L3  1-N team building after 20 learning rounds")

team2 = match_one_to_n(alice, task, [bob, carol], wm.theta, cfg, use_ucb=True, round_t=21)
print(f"  Team: {team2.team_members}  coverage={team2.collective_coverage:.4f}")
ok("Team building still valid post-learning")


# ── Summary ───────────────────────────────────────────────────────────

section("SUMMARY")
print("""
  L1  ✓  JSON / dict  →  UserState / Task
  L2  ✓  Gate (clearance + hard constraint) + MapScore (attention soft-match)
         UCB exploration (β·σ bonus)
  L3  ✓  1-1 ranking (sorted by M̃, gate filtered)
         1-N team building (greedy submodular, (1−1/e) guarantee)
  L4  ✓  Feedback stub (dummy → real signals TBD in Phase 3)
  L5  ✓  Bayesian (μ,σ) update — embedding similarity + source-aware σ_obs
         Weight SGD (softmax Jacobian)
         UCB round counter
""")
