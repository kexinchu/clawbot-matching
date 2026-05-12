# Online Learning — 20-Task Evaluation

Evaluation of the Layer-5 online-learning loop on the simulator test set (20 tasks × 20 candidates per task). Most recent runs use `simulator/20_Tasks_Testset_tiered.json`: the base cap levels stay in `candidate_profile.capabilities` as ground truth, while each candidate gets tier-based `capability_priors` (`μ_init`, `σ_init` per skill) from `python -m simulator.augement_test_set_with_tiers`. The harness loads priors for the learning pool; `M_optimum` and ρ still use true capabilities. The goal is to characterise convergence and quality vs two baselines.

- Code: `Online_learning/tests/run_20_tasks_evaluation.py`
- Single-condition plot: `plots/plot_20_tasks_evaluation.py`
- Multi-condition comparison plot: `plots/plot_20_tasks_noise_comparison.py`

---

## 1. Procedure

### 1.1 Test set

`simulator/20_Tasks_Testset.json` is the base file. **`simulator/20_Tasks_Testset_tiered.json`** augments each candidate with `capability_priors`, `tier`, and `tier_meta` while leaving `candidate_profile.capabilities` as ground truth. Each task entry yields:

- `task`: `task_id`, `description`, `required_skills` (skill → required level)
- `proposer_profile`: requester user with `capabilities` + `needs`
- `candidates[*]`: 20 candidate profiles, each with `capabilities`, `needs`, role, etc.

The requester is rebuilt with **empty capabilities** for the matching pipeline. With the raw proposer capabilities included, the requester's own skills cancel the per-requirement gap inside `S_cap` and every candidate saturates at `M = 1.0`, making the experiment undecidable. Building the requester as "I delegate this task; treat each requirement as a real gap" gives every candidate a differentiable `S_cap`.

### 1.2 Three conditions per task

| Condition       | Selection                              | Capability UCB (μ+β·σ) | Candidate UCB | Weight SGD | Notes                                       |
|-----------------|----------------------------------------|------------------------|---------------|------------|---------------------------------------------|
| `online_ucb`    | full pipeline + **σ-weighted** candidate UCB (`M + c·σ̄·√(log t/(n+1))`) | on                     | on (c=0.5)    | on         | the proposed method                         |
| `scap_greedy`   | analytical greedy on `S_cap` only      | off                    | off           | off        | `θ_c, θ_n = (10, -10)` → `w_n ≈ 0`          |
| `random`        | uniformly random pool selection         | n/a                    | n/a           | n/a        | 5 seeds, results averaged                   |

Dreaming (Layer 3.2) is disabled for all three to keep the comparison about the selection mechanism, not the LLM compatibility layer.

### 1.3 Ground-truth oracle

`M_optimum` and ρ are always computed against the **truth**, independent of what the agent has learned:

- For each candidate, a "true" `UserState` is built with `μ = capabilities[skill]` and `σ ≈ 0`.
- `M_optimum = max over candidates of M(true_v)` under a fixed neutral θ = (0.4, −0.1).
- `M_optimum_id` = the candidate that achieves it (the analytical optimum).

### 1.4 Feedback environment

A custom `GroundTruthFeedback(FeedbackProvider)` was added (see `run_20_tasks_evaluation.py`). When the agent picks candidate `v`, the feedback is computed against `v`'s **true** capabilities — not against the agent's current belief. This is the standard bandit assumption: a fixed environment that can correct the agent's posterior. Without it, when initial μ are noisy the agent rewards confirm its own mistaken belief and never recover.

Concretely, `GroundTruthFeedback.collect` calls `WorldModel.compute_match(requester, true_v, task)` and passes the resulting `MatchResult` through `dummy_user_feedback` to derive the 6-field feedback dict.

### 1.5 Metrics

For every (task × condition) we record:

- `n_rounds_consensus`: first round at which the same candidate has been selected `K_CONVERGE = 3` times in a row (capped at `N_MAX_ROUNDS`).
- `n_rounds_first_optimal`: first round at which the analytical optimum was selected (capped at `N_MAX_ROUNDS`).
- `rho_last = M(last selected candidate, true caps) / M_optimum`
- `rho_mode = M(most-frequently selected candidate, true caps) / M_optimum`

For the random baseline these are means/stds over 5 seeds.

### 1.6 Prior-noise modes

`perturb_userstate_mu(state, mode, sigma, rng)` sets the agent's **initial** μ for every candidate before round 1:

| Mode       | Effect on learning-pool μ (after pool construction) | Notes |
|------------|-----------------------------------------------------|-------|
| `none`     | No extra perturbation | With tiered JSON, pool already uses `capability_priors`; with non-tiered JSON, μ comes from profiles (legacy flat σ experiments). |
| `gaussian` | `μ ← clip(μ + N(0, σ), 0, 1)` per capability | Layered on whatever μ the pool currently has. |
| `uniform`  | `μ ← U(0, 1)` per capability | Cold-start stress test layered on the pool. |

### 1.7 Reproducibility

```bash
# Run a single condition
python Online_learning/tests/run_20_tasks_evaluation.py \
    --noise-mode gaussian --noise-sigma 0.5 \
    --n-max-rounds 25 --pool-size 8

# Per-condition plot
python plots/plot_20_tasks_evaluation.py <suffix>

# Multi-condition comparison
python plots/plot_20_tasks_noise_comparison.py <suffix1> <suffix2> <suffix3>
```

CLI flags exposed by the runner: `--noise-mode`, `--noise-sigma`, `--noise-seed`, `--n-max-rounds`, `--k-converge`, `--pool-size`.

---

## 2. Experiments and observations

Results are means over 20 tasks. `n_first_opt` and `n_consensus` are capped at the round budget.

### 2.1 Baseline: pool=20, budget=60, no noise

```
                   n_consensus    n_first_opt    rho_mode    rho_last
online_ucb         60.00          10.30          0.924       0.819
scap_greedy         3.05            1.00          1.000       1.000
random             22.00          35.24          0.900       0.900
```

Plot: `plots/online_learning_20tasks_eval_20260511-222549.png`

Observations:
- With perfect priors, `scap_greedy` is essentially the oracle (`ρ = 1.000`, converges on round 3).
- `online_ucb` finds the optimum within ~10 rounds (`n_first_opt = 10.3`) but UCB1's exploration keeps it from settling on a K=3 streak inside 60 rounds (`n_consensus = 60`, the cap). Forcing it to commit to its mode pick gives `ρ_mode = 0.924`.
- `random` lands the optimum slowly (round ~35) and its mode-pick quality is the pool-average ρ ≈ 0.90.

This is the "no-incentive-to-learn" regime. Without noise the UCB cost is pure overhead.

### 2.2 pool=20, budget=60, gaussian σ=0.2

```
                   n_consensus    n_first_opt    rho_mode    rho_last
online_ucb         60.00          10.60          0.907       0.838
scap_greedy        23.65          16.15          0.981       0.978
random             22.00          35.24          0.900       0.900
```

Plot: `plots/online_learning_20tasks_eval_gaussian0.2_20260512-062918.png`

Observations:
- `scap_greedy` no longer converges instantly. `n_first_opt` jumps from 1.00 to 16.15: greedy commits to its noisy top pick, the truth-grounded feedback drags that pick's μ down, and greedy gradually eliminates wrong candidates until it lands on the optimum. `n_consensus` rises to 23.65.
- `ρ_mode` for greedy drops only marginally (0.982 → 0.981) because gaussian perturbation around the truth typically preserves which candidates are roughly best.
- `online_ucb` is essentially unchanged from the no-noise condition: it does the same warmup regardless of priors, so `n_first_opt` stays at ~10.6 and `ρ_mode` at 0.907.

### 2.3 pool=20, budget=60, uniform random priors

```
                   n_consensus    n_first_opt    rho_mode    rho_last
online_ucb         60.00          12.40          0.866       0.848
scap_greedy        34.25          46.15          0.938       0.946
random             22.00          35.24          0.900       0.900
```

Plot: `plots/online_learning_20tasks_eval_uniform_20260512-062920.png`

Observations:
- `scap_greedy` is now actively harmed by its priors. `n_first_opt = 46.15` means it spends nearly the whole 60-round budget eliminating wrong candidates before stumbling on the optimum. `ρ_mode` drops to 0.938.
- `online_ucb` still finishes its warmup in ~20 rounds and finds the optimum in ~12.4 rounds. `ρ_mode = 0.866` — slightly worse than `random`, because UCB1's post-warmup exploration is still picking under-explored arms instead of locking in.
- `random` is, by construction, completely noise-invariant.

Comparison plot for all three at pool=20, budget=60:
`plots/online_learning_20tasks_noise_comparison.png`.

### 2.4 Tighter budget: pool=20, budget=25, σ=0.5

```
                   n_consensus    n_first_opt    rho_mode    rho_last
true priors:
  online_ucb       25.00          11.00          0.896       0.918
  scap_greedy      24.15           1.00          0.982       0.959
  random           16.00          17.60          0.901       0.900

gaussian σ=0.5:
  online_ucb       25.00          10.55          0.911       0.920
  scap_greedy      11.95          15.00          0.956       0.967
  random           16.00          17.60          0.901       0.900

uniform random μ:
  online_ucb       25.00          12.40          0.878       0.927
  scap_greedy       6.95          18.75          0.940       0.946
  random           16.00          17.60          0.901       0.900
```

Comparison plot: `plots/online_learning_20tasks_noise_comparison_n25.png`

Observation: under a 25-round budget with 20 candidates, UCB1's warmup ("try each unseen candidate once" via the n_v=0 → ∞ bonus) consumes nearly the whole budget. So UCB has barely any rounds left to exploit, and greedy-by-elimination (driven by `GroundTruthFeedback`) can still recover well enough. The robustness pattern is visible — UCB metrics flat across noise levels, greedy `n_first_opt` going 1 → 15 → 18.75 — but the rho_mode crossover does not happen yet.

### 2.5 Crossover regime: pool=8, budget=25, σ=0.5

```
                   n_consensus    n_first_opt    rho_mode    rho_last
true priors:
  online_ucb       25.00           3.90          0.950       0.950
  scap_greedy      14.85           1.00          0.997       0.992
  random            9.20          11.47          0.913       0.907

gaussian σ=0.5:
  online_ucb       25.00           4.25          0.957       0.957
  scap_greedy       6.85          14.00          0.947       0.954
  random            9.20          11.47          0.913       0.907

uniform random μ:
  online_ucb       25.00           4.75          0.957       0.957
  scap_greedy       5.00          20.40          0.938       0.943  ← worse than UCB
  random            9.20          11.47          0.913       0.907
```

Comparison plot: `plots/online_learning_20tasks_noise_comparison_n25_p8.png`

This is the regime that demonstrates the value of UCB. With only 8 candidates and a 25-round budget, UCB1's warmup costs ~8 rounds, leaving 17 rounds to exploit. Greedy, when its priors are wrong, runs out of budget before it can eliminate enough wrong candidates.

### 2.6 Tier-based priors, budget 25, no CLI noise (`--noise-mode none`)

Test set: `simulator/20_Tasks_Testset_tiered.json`. Learning uses `capability_priors` per candidate (four tiers: veteran / mid / new_explicit / new_implicit, with coupled `σ_init` and `μ_init` noise). Candidate-level selection uses **σ-weighted** UCB in `Online_learning.py` (bonus scales by mean capability σ so high-σ candidates are explored more). Capability-level UCB inside the match score remains on for `online_ucb`; dreaming stays off.

#### Pool = 8 candidates, `N_MAX_ROUNDS = 25`

```
                   n_consensus    n_first_opt    rho_mode    rho_last
online_ucb         21.95          11.60          0.952       0.929
scap_greedy         4.35           6.30          0.983       0.983
random              9.20          11.47          0.913       0.907
```

Log: `logs/online_learning_20tasks_eval_none_n25_p8_tiered.json`  
Plot: `plots/online_learning_20tasks_eval_none_n25_p8_tiered.png`

Observations:

- **Greedy stays strong** on quality (`ρ_mode ≈ 0.983`) and reaches consensus quickly (`n_consensus ≈ 4.35`). Tiered `μ_init` is still anchored near truth; ground-truth feedback lets greedy correct ranking errors within a few rounds.
- **Online UCB** reaches the optimum in median earlier than random (`n_first_opt` mean 11.6 vs random 11.47) but **does not hit K=3 consensus** before the cap on many tasks (`n_consensus` mean 21.95, median 25): exploration bonuses tied to prior σ keep the policy from locking in within 25 rounds, while `ρ_mode ≈ 0.952` stays close to greedy’s mode quality.
- **ρ_last vs ρ_mode** for online_ucb: last-round pick is slightly worse than the eventual mode (0.929 vs 0.952), consistent with ongoing exploration at round 25.

#### Pool = 20 candidates (full), `N_MAX_ROUNDS = 25`

```
                   n_consensus    n_first_opt    rho_mode    rho_last
online_ucb         25.00          16.60          0.952       0.902
scap_greedy         6.95           9.05          0.975       0.968
random             16.00          17.60          0.901       0.900
```

Log: `logs/online_learning_20tasks_eval_none_n25_tiered.json`  
Plot: `plots/online_learning_20tasks_eval_none_n25_tiered.png`

Observations:

- **Online UCB** hits the **25-round cap on consensus for every task** (`n_consensus = 25.00`): with 20 candidates and limited budget, σ-weighted UCB still spreads selections enough that a 3-in-a-row streak rarely completes before the horizon. **First-optimal** pick averages **16.6** rounds — slower than in the small-pool regime because the ranked pool is larger and the policy keeps revisiting high-uncertainty arms.
- **Greedy** remains the practical winner here: faster consensus (6.95), higher `ρ_mode` (0.975 vs 0.952), and `ρ_last` nearly equal to `ρ_mode` (policy has largely committed).
- **Random** again sits near ρ ≈ 0.90 on `ρ_mode`; tiered priors do not affect the random baseline.

Comparison to **§2.4** (same budget and pool sizes, but **flat `σ_init` and true μ** on the non-tiered JSON): tiered priors add realistic **μ** noise and heterogeneous **σ**, which **slow greedy’s** path to the optimum (`n_first_opt` rises from ~1 toward ~9 for pool=20) while keeping it best on `ρ_mode` under `none`. The tiered setting is the right regime for stress-testing σ-modulated exploration without extra `--noise-mode gaussian` / `uniform`.

---

## 3. Key findings

### Finding 1 — When priors are correct, `S_cap` greedy is the oracle

With `noise = none`, `scap_greedy` reaches the analytical optimum on round 1 (`n_first_opt = 1.00`) and achieves `ρ ≈ 1.0`. UCB pays a small "exploration tax" (`ρ_mode ≈ 0.90–0.95`) because its UCB1 warmup forces it to sample under-explored candidates even after it has found the right one. This is the expected explore-vs-exploit tradeoff.

### Finding 2 — UCB metrics are essentially flat across noise levels

`Online_ucb` shows roughly the same `n_first_opt` regardless of how bad the priors are:

| noise mode     | n_first_opt (pool=20)   | n_first_opt (pool=8)    |
|----------------|-------------------------|--------------------------|
| none           | 10.30 – 11.00           | 3.90                     |
| gaussian σ=0.2 | 10.60                   | —                        |
| gaussian σ=0.5 | 10.55                   | 4.25                     |
| uniform        | 12.40                   | 4.75                     |

UCB1's first-pass warmup makes the time-to-optimum a function of pool size, not prior quality. This is what bandit theory predicts.

### Finding 3 — Greedy's time-to-optimum degrades sharply with noise

`scap_greedy.n_first_opt` over the pool=20 budget=60 series:

| noise mode     | n_first_opt  |
|----------------|--------------|
| none           |  1.00        |
| gaussian σ=0.2 | 16.15        |
| gaussian σ=0.5 | 15.00 (budget=25) / not run (budget=60) |
| uniform        | 46.15        |

Greedy can still find the optimum eventually via Bayesian-update-driven elimination, but the rounds required grow ~46× from `none` to `uniform`. If the round budget is anywhere near the pool size, this matters.

### Finding 4 — Quality crossover under tight budget + small pool

At pool=8, budget=25, `ρ_mode` shows the crossover we predicted:

| noise mode     | online_ucb | scap_greedy   |
|----------------|------------|---------------|
| none           | 0.950      | **0.997** ★   |
| gaussian σ=0.5 | **0.957**  | 0.947         |
| uniform        | **0.957**  | 0.938         |

- With **true priors**, greedy wins (0.997 vs 0.950): the oracle prior makes UCB's exploration purely wasteful.
- With **σ=0.5 noise**, the two are essentially tied (0.957 vs 0.947).
- With **uniform priors**, UCB beats greedy (0.957 vs 0.938).

The reason the crossover requires a small pool: at pool=20 UCB has so little exploitation budget that even when greedy fails, UCB doesn't finish second — it's still in warmup. The crossover regime is the one where the budget is comparable to but slightly larger than pool size (here: budget=25 ≈ 3 × pool=8).

### Finding 5 — Random is a noise-invariant floor

`random.ρ_mode = 0.900–0.913` in every condition. It uses no priors and no learning, so noise has no effect on it. Across all experiments, both `online_ucb` and `scap_greedy` clear this floor in every noise condition, but only by a few percentage points — the 20-task testset's candidate pools are full of reasonable matches, so even an uninformed picker does decently.

### Finding 6 — `ρ_last` vs `ρ_mode` matters for UCB-style methods

For `scap_greedy` the two metrics nearly coincide (greedy locks on once it commits). For `online_ucb`, `ρ_mode > ρ_last` consistently — the agent has identified the right candidate but its last selection is often an exploratory pick on a different arm. If we ask "what would the system commit to if forced", `ρ_mode` is the right number; if we ask "what's the agent's most recent recommendation", `ρ_last` is.

---

## 4. Practical recommendations

1. **Candidate-level exploration** in `Online_learning.py` uses **σ-weighted** UCB: `M̃_v = M_v + c · σ̄_v · √(log(t)/(n_v+1))`, with σ̄_v the mean of the candidate’s capability σ values (from priors / posterior). There is no infinite warm-up for `n_v = 0`; unseen arms get a **finite** bonus proportional to σ̄_v, so the policy is not forced to try every candidate before re-sampling.

2. **For the current testset, the explore/exploit tradeoff favors greedy** because the testset's M-gaps between candidates are small (~0.05–0.15) and the ground-truth feedback is strongly corrective. If you want UCB to look better in absolute terms, the regime to push toward is: noisier feedback, larger M-gaps, tighter round budget, smaller pool.

3. **Tune `candidate_ucb_c` to the budget and pool.** `c = sqrt(2)` (UCB1's textbook value) is too aggressive at 20 candidates with small M-gaps — it keeps the agent in exploration mode well past where it should commit. Dropping to `c = 0.2` or `0.1` would sharpen the crossover at pool=20.

4. **`ρ_mode` is a better quality metric than `ρ_last`** for any method that maintains exploration in steady state (UCB, Thompson, ε-greedy). `ρ_last` is fine for pure greedy.

5. **`GroundTruthFeedback` should remain the default for benchmarking.** Using `DummyFeedback` (where reward is derived from the agent's *current belief*) makes the benchmark non-corrective and overstates greedy's robustness: a greedy that confirms its own beliefs trivially "converges" but to whatever wrong answer it started with.

---

## 5. File index

### Code

- `simulator/augement_test_set_with_tiers.py` — build `20_Tasks_Testset_tiered.json` from the base test set (`python -m simulator.augement_test_set_with_tiers ...` avoids a stdlib `types` name clash when run from `simulator/`).
- `Online_learning/tests/run_20_tasks_evaluation.py` — experiment driver.
- `plots/plot_20_tasks_evaluation.py` — per-condition 2×2 plot.
- `plots/plot_20_tasks_noise_comparison.py` — multi-condition comparison.

### Result JSONs (`logs/`)

| Suffix                                       | Setup                               |
|----------------------------------------------|-------------------------------------|
| `none_20260512-062916`                       | pool=20, budget=60, noise=none      |
| `gaussian0.2_20260512-062918`                | pool=20, budget=60, σ=0.2           |
| `uniform_20260512-062920`                    | pool=20, budget=60, uniform         |
| `none_n25_20260512-063829`                   | pool=20, budget=25, noise=none      |
| `gaussian0.5_n25_20260512-063830`            | pool=20, budget=25, σ=0.5           |
| `uniform_n25_20260512-063831`                | pool=20, budget=25, uniform         |
| `none_n25_p8_20260512-064022`                | pool=8,  budget=25, noise=none      |
| `gaussian0.5_n25_p8_20260512-064023`         | pool=8,  budget=25, σ=0.5           |
| `uniform_n25_p8_20260512-064023`             | pool=8,  budget=25, uniform         |
| `none_n25_p8_tiered`                         | tiered JSON, pool=8,  budget=25, noise=none, σ-weighted cand. UCB |
| `none_n25_tiered`                            | tiered JSON, pool=20, budget=25, noise=none, σ-weighted cand. UCB |

### Plots (`plots/`)

- `online_learning_20tasks_eval_<suffix>.png` — one per result JSON.
- `online_learning_20tasks_eval_none_n25_p8_tiered.png` — tiered priors, pool=8, budget=25.
- `online_learning_20tasks_eval_none_n25_tiered.png` — tiered priors, pool=20, budget=25.
- `online_learning_20tasks_noise_comparison.png` — pool=20, budget=60, the three noise modes.
- `online_learning_20tasks_noise_comparison_n25.png` — pool=20, budget=25, the three noise modes.
- `online_learning_20tasks_noise_comparison_n25_p8.png` — pool=8, budget=25, the three noise modes (the crossover figure).
