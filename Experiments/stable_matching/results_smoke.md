# Batch Stable Matching Results

- Benchmark seeds: `[42]`
- Batches per seed: `2`
- Tasks / batch: `5`
- Candidates / batch: `7`
- Candidate capacity: `1`
- Tie-break seed: `0`

## Aggregate (mean ± std across batches × seeds)

| method | total batch reward | mean total reward | matched task rate | mutual accept | completion | req sat | cand sat | min pair reward |
|---|---|---|---|---|---|---|---|---|
| GaleShapley-SkillCoverage | 3.9065 ± 0.0412 | 0.7813 ± 0.0082 | 1.0000 ± 0.0000 | 0.4793 ± 0.0112 | 0.5582 ± 0.0272 | 0.6040 ± 0.0363 | 0.5718 ± 0.0206 | 0.7510 ± 0.0059 |
| CoWeaver-DA | 3.5304 ± 0.4101 | 0.7841 ± 0.0040 | 0.9000 ± 0.1000 | 0.4765 ± 0.0194 | 0.5627 ± 0.0137 | 0.5912 ± 0.0302 | 0.6023 ± 0.0358 | 0.7530 ± 0.0039 |

Oracle-MaxWeight is an **oracle upper bound** using hidden `total_reward`; it is not a fair method comparison target.

## Preference diagnostics

- Mean requester ranking Kendall τ (GS vs CoWeaver-DA): `0.2867 ± 0.0067`
- Mean #tasks with different assignment: `3.0000 ± 1.0000`
- CoWeaver matched mean S_need vs GS: `0.7683 ± 0.0485` vs `0.6504 ± 0.0764`
- GS matched mean coverage vs CoWeaver: `0.2983 ± 0.0817` vs `0.2860 ± 0.0582`

## Reproducible commands

```bash
python simulator/generate_batch_matching_testset.py --seeds 42,123,456 --output simulator/Batch_Matching_Testset_v1.json
```

```bash
python Online_learning/tests/run_batch_stable_matching.py --seeds 42 --tie-break-seed 0 --tag smoke
```

```bash
pytest Online_learning/tests/test_stable_matching.py Online_learning/tests/test_batch_generator.py -q
```

