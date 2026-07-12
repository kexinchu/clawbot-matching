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
| GaleShapley-SkillCoverage | 3.8158 ± 0.3320 | 0.8010 ± 0.0223 | 0.9533 ± 0.0846 | 0.5255 ± 0.0308 | 0.6170 ± 0.0378 | 0.6482 ± 0.0345 | 0.5648 ± 0.0359 | 0.7323 ± 0.0741 |
| CoWeaver-DA | 3.7729 ± 0.3772 | 0.7916 ± 0.0209 | 0.9533 ± 0.0921 | 0.5056 ± 0.0316 | 0.5883 ± 0.0388 | 0.6198 ± 0.0350 | 0.5722 ± 0.0364 | 0.7422 ± 0.0527 |
| Oracle-MaxWeight (oracle upper bound) | 4.0662 ± 0.0627 | 0.8132 ± 0.0125 | 1.0000 ± 0.0000 | 0.5399 ± 0.0266 | 0.6356 ± 0.0283 | 0.6642 ± 0.0268 | 0.5711 ± 0.0346 | 0.7622 ± 0.0136 |

Oracle-MaxWeight is an **oracle upper bound** using hidden `total_reward`; it is not a fair method comparison target.

## Preference diagnostics

- Mean requester ranking Kendall τ (GS vs CoWeaver-DA): `0.1888 ± 0.2416`
- Mean #tasks with different assignment: `3.5500 ± 1.1608`
- CoWeaver matched mean S_need vs GS: `0.7407 ± 0.0858` vs `0.6364 ± 0.0850`
- GS matched mean coverage vs CoWeaver: `0.4731 ± 0.0877` vs `0.3948 ± 0.0936`

Preference diagnostics match the intended method biases: **CoWeaver-DA** selects higher \(S_{\mathrm{need}}\) matches; **GaleShapley-SkillCoverage** selects higher discrete skill coverage. On this batch oracle, GS is slightly ahead on mean batch `total_reward`, while CoWeaver has a slightly higher minimum matched-pair reward (worst-case fairness diagnostic).

## Unit tests

```text
19 passed in 0.29s
```

(`test_stable_matching.py` + `test_batch_generator.py`)

## Reproducible commands

```bash
PYTHONPATH=. python simulator/generate_batch_matching_testset.py --seeds 42,123,456 --output simulator/Batch_Matching_Testset_v1.json
```

```bash
PYTHONPATH=. python Online_learning/tests/run_batch_stable_matching.py --seeds 42,123,456 --tie-break-seed 0 --tag full
```

```bash
pytest Online_learning/tests/test_stable_matching.py Online_learning/tests/test_batch_generator.py -q
```

## Artifacts

- JSON log: `logs/batch_stable_matching_full.json` (per-batch + aggregate over seeds 42/123/456)
- Datasets: `simulator/Batch_Matching_Testset_v1.json` and `*_seed{42,123,456}.json`

