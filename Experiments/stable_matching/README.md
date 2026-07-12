# Gale–Shapley Stable Matching for Batch Collaboration Allocation

This experiment is the final **external baseline** on CoWeaver’s `pd` branch.
It compares two preference constructions under an **identical** many-to-one
deferred-acceptance (DA) allocator on a **batch** benchmark with shared
candidates and capacity conflict.

| Method | Requester→candidate preference | Candidate→task preference |
|--------|--------------------------------|---------------------------|
| `GaleShapley-SkillCoverage` | Discrete weighted skill coverage | Discrete offer–need fit |
| `CoWeaver-DA` | Feasibility-gated analytical MapScore \(M\) (no UCB / Dreaming / feedback) | \(S_{\mathrm{need}}\) |

**Oracle Max-Weight Assignment** is reported only as a hidden-outcome **upper
bound**. It is not a fair method comparison target.

---

## Why a new batch benchmark?

The individual v3 suite (`20_Tasks_Testset_v3_tiered.json`) gives every task its
own candidate pool generated from that task’s requirements. Candidates are
**not shared**, so there is **no capacity competition** and classical stable
matching is vacuous (each task can independently pick its favourite).

The batch benchmark (`simulator/Batch_Matching_Testset_v1*.json`) instead has:

- 20 batches (default)
- 5 requester–task pairs per batch
- 7 **shared** candidates per batch
- candidate capacity = 1 (code supports \(k > 1\))
- task capacity = 1
- unmatched tasks / candidates allowed

All profiles follow the **v3 schema** (capabilities, needs, preferences,
constraints, history; task offers; per-pair cards, personas, and hidden
`context_latents`).

---

## Mapping from classical Gale–Shapley

| Classical GS | This work |
|--------------|-----------|
| Men / proposers | Requester–task pairs (propose) |
| Women / acceptors | Candidates (capacity \(k\)) |
| Acceptable partners | Public feasible edges |
| Preference lists | Method-specific scores (below) |
| Stable matching | No blocking pair under stated prefs |

DA implementation: `Experiments/stable_matching/deferred_acceptance.py`
(requester-proposing, many-to-one).

---

## Preference formulas

### A. GaleShapley-SkillCoverage

\[
P^{GS}_{u_i}(v_j)=\mathrm{DiscreteWeightedSkillCoverage}(T_i,v_j)
\]

- Uses only `task.required_skills` and `candidate.capabilities`
- Skill \(s\) counts iff \(\mathrm{cap}(s)\ge \mathrm{req}(s)\)
- Aggregated by requirement weights
- **No** residual gap, \(S_{\mathrm{need}}\), MapScore, UCB, Dreaming, or outcome oracle

\[
P^{GS}_{v_j}(T_i)=\mathrm{DiscreteOfferNeedFit}(O_{T_i},\mathrm{Need}_{v_j})
\]

- Uses only public `task.offers` and `candidate.needs`
- Implemented via simulator `offer_need_fit` (token overlap × strength)

### B. CoWeaver-DA

\[
P^{CW}_{u_i}(v_j)=M(u_i,v_j,T_i)=\sigma\cdot(w_c S_{\mathrm{cap}}+w_n S_{\mathrm{need}})
\]

- Analytical MapScore with evaluation \(\theta\) (\(w_c,w_n\) via softmax)
- UCB **off**, feedback update **off**, Dreaming **off**
- Hidden latents / outcomes **not** read for preferences

\[
P^{CW}_{v_j}(T_i)=S_{\mathrm{need}}(v_j,T_i)
\]

- Candidates refuse gate failures / non-positive scores

**Sole algorithmic difference:** preference construction. Same feasible edges,
same DA, same capacities, same tie-break seed.

---

## Public signal vs hidden oracle

**Public (may enter preferences):** profiles, task specs/offers, cards,
`public_feasible`.

**Hidden (evaluation only):** `context_latents`, BilateralSimulator /
OutcomeSimulator / `compute_reward` outputs, MapScore diagnostics as labels,
Oracle Max-Weight utilities \(U^\star_{ij}=\mathrm{total\_reward}_{ij}\).

---

## Benchmark generation

```bash
PYTHONPATH=. python simulator/generate_batch_matching_testset.py \
  --seeds 42,123,456 \
  --num-batches 20 \
  --tasks-per-batch 5 \
  --candidates-per-batch 7 \
  --candidate-capacity 1 \
  --output simulator/Batch_Matching_Testset_v1.json
```

CLI also supports `--seed` (single). Fixed seeds **42, 123, 456** are the
default cross-benchmark set.

Tie-breaking: within equal scores, a seeded shuffle
(`--tie-break-seed`, default 0) orders preference lists. DA uses those lists
directly.

---

## Running the experiment

Smoke (2 batches/seed, skip oracle for speed):

```bash
python Online_learning/tests/run_batch_stable_matching.py \
  --seeds 42 --max-batches 2 --skip-oracle --tag smoke
```

Full 3-seed run:

```bash
python Online_learning/tests/run_batch_stable_matching.py \
  --seeds 42,123,456 --tag full
```

Outputs:

- `logs/batch_stable_matching_<tag>.json` — per-batch + aggregate
- `Experiments/stable_matching/results_<tag>.md` — table + commands

---

## Tests

```bash
pytest Online_learning/tests/test_stable_matching.py \
       Online_learning/tests/test_batch_generator.py -q
```

Invariants covered: shared profiles, capacity, DA stability, gate exclusion,
GS≠MapScore, CoWeaver≠hidden latents, shared edges/allocator, competition,
oracle isolation, seed reproducibility.

---

## File layout

```
simulator/generate_batch_matching_testset.py
simulator/Batch_Matching_Testset_v1.json
simulator/Batch_Matching_Testset_v1_seed{42,123,456}.json
Experiments/stable_matching/
  README.md
  batch_generator_utils.py
  preferences.py
  deferred_acceptance.py
  evaluation.py
  plot_batch_results.py
Online_learning/tests/
  run_batch_stable_matching.py
  test_stable_matching.py
  test_batch_generator.py
```
