# Interpretability CSV Audit

Generated from real log JSON under `logs/`. No hand-filled numbers.

## Source logs

| condition | expected log pattern | resolved | status |
|---|---|---|---|
| tiered_original | include: simple_tiered; exclude: multi_seed, 5seed, contrast_v1 | `/root/autodl-tmp/clawbot-matching/logs/faithfulness_suite_simple_tiered_20260524-205758.json` | OK |
| contrast_benchmark | include: simple_contrast_v2; exclude: multi_seed, 5seed, contrast_v1, sbert | `/root/autodl-tmp/clawbot-matching/logs/faithfulness_suite_simple_contrast_v2_20260524-205802.json` | OK |
| contrast_benchmark_sbert | include: sbert_contrast_v2; exclude: multi_seed, 5seed | `/root/autodl-tmp/clawbot-matching/logs/faithfulness_suite_sbert_contrast_v2_20260524-205804.json` | OK |
| failure_case | include: failure_case; exclude: (none) | `/root/autodl-tmp/clawbot-matching/logs/faithfulness_suite_simple_failure_case_20260524-205826.json` | OK |
| contrast_benchmark_5seed | include: any of: contrast_v2_multi_seed, 5seed; exclude: contrast_v1 | `/root/autodl-tmp/clawbot-matching/logs/faithfulness_suite_simple_contrast_v2_multi_seed_20260524-205827.json` | OK |

## Extracted summary

| condition | num_tasks | encoder | oracle_ratio | oracle_match | status |
|---|---:|---|---:|---:|---|
| tiered_original | 20 | simple | 0.9808 | 0.9500 | OK |
| contrast_benchmark | 20 | simple | 1.0000 | 1.0000 | OK |
| contrast_benchmark_sbert | 20 | sbert | 0.9999 | 0.9500 | OK |
| failure_case | 10 | simple | 0.2914 | 0.1000 | OK |
| contrast_benchmark_5seed | 20 | simple | 1.0000 | 1.0000 | OK |

## Validation checks

- [PASS] All source logs resolved: At least one log file found per condition
- [PASS] tiered_original num_tasks == 20: expected 20, got 20
- [PASS] contrast_benchmark num_tasks == 20: expected 20, got 20
- [PASS] contrast_benchmark_sbert num_tasks == 20: expected 20, got 20
- [PASS] failure_case num_tasks == 10: expected 10, got 10
- [PASS] contrast_benchmark_5seed num_tasks == 20: expected 20, got 20
- [PASS] contrast_benchmark mean_frac_candidates_scap_eq_1 < 0.2: value=0.07
- [PASS] contrast_benchmark mean_explanation_vs_oracle_ratio >= 0.9: value=1.0
- [PASS] contrast_benchmark frac_explanation_matches_oracle >= 0.9: value=1.0
- [PASS] contrast_benchmark_sbert mean_explanation_vs_oracle_ratio >= 0.9: value=0.9999
- [PASS] contrast_benchmark_sbert frac_explanation_matches_oracle >= 0.9: value=0.95
- [PASS] failure_case mean_explanation_vs_oracle_ratio < 0.5: value=0.2914
- [PASS] failure_case frac_explanation_matches_oracle < 0.5: value=0.1
- [PASS] No NA cells in CSV (except allowed fields): total NA cells=12
- [PASS] contrast_benchmark mean_top_factor_drop > mean_random_factor_drop: top=0.3684, random=0.1548
- [PASS] contrast_benchmark top_rank_flip_rate > random_rank_flip_rate: top_flip=0.95, random_flip=0.65
- [PASS] failure_case mean_top_factor_drop < mean_random_factor_drop (informational): top=0.0923, random=0.171 (expected failure-mode pattern)

## Missing fields

| (none) | | |

## Notes

- Contrast V1 is intentionally excluded from the paper CSV.
- `contrast_benchmark_5seed` uses `summary_mean_std` when present; otherwise mean/std computed from `per_seed` summaries.
- Archetype distribution is extracted from `summary.archetype_distribution` or aggregated from `per_task` metadata for contrast conditions only.
- Tiered and failure_case runs have `archetype_distribution: null` in logs; CSV archetype columns are `NA` as expected.