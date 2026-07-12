# Batch Stable Matching Results

- Benchmark seeds: `[42, 123, 456]`
- Batches per seed: `20`
- Tasks / batch: `5`
- Candidates / batch: `7`
- Candidate capacity: `1`
- Tie-break seed: `0`

## Aggregate (mean ± std across batches × seeds)

| method | total batch reward | mean total reward | matched task rate | mutual accept | completion | req sat | cand sat | min pair reward |
|---|---|---|---|---|---|---|---|---|
| GaleShapley-SkillCoverage | 3.8378 ± 0.3361 | 0.8055 ± 0.0188 | 0.9533 ± 0.0846 | 0.5426 ± 0.0296 | 0.6198 ± 0.0373 | 0.6557 ± 0.0342 | 0.5671 ± 0.0358 | 0.7477 ± 0.0477 |
| CoWeaver-DA | 3.7868 ± 0.3768 | 0.7944 ± 0.0178 | 0.9533 ± 0.0921 | 0.5236 ± 0.0320 | 0.5905 ± 0.0381 | 0.6266 ± 0.0352 | 0.5744 ± 0.0362 | 0.7506 ± 0.0344 |
| Oracle-MaxWeight (oracle upper bound) | 4.0751 ± 0.0622 | 0.8150 ± 0.0124 | 1.0000 ± 0.0000 | 0.5552 ± 0.0258 | 0.6375 ± 0.0279 | 0.6715 ± 0.0260 | 0.5723 ± 0.0344 | 0.7620 ± 0.0140 |

Oracle-MaxWeight is an **oracle upper bound** using hidden `total_reward`; it is not a fair method comparison target.

## Preference diagnostics

- Mean requester ranking Kendall τ (GS vs CoWeaver-DA): `0.1888 ± 0.2416`
- Mean #tasks with different assignment: `3.5500 ± 1.1608`
- CoWeaver matched mean S_need vs GS: `0.7407 ± 0.0858` vs `0.6364 ± 0.0850`
- GS matched mean coverage vs CoWeaver: `0.4731 ± 0.0877` vs `0.3948 ± 0.0936`

## Reproducible commands

```bash
python simulator/generate_batch_matching_testset.py --seeds 42,123,456 --output simulator/Batch_Matching_Testset_v1.json
```

```bash
python Online_learning/tests/run_batch_stable_matching.py --seeds 42,123,456 --tie-break-seed 0 --tag formal
```

```bash
pytest Online_learning/tests/test_stable_matching.py Online_learning/tests/test_batch_generator.py -q
```

