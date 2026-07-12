# V3 Dreaming Ablation Summary

Generated at: `2026-07-12T16:51:02.750539`
Testset: `/root/autodl-tmp/CoWeaver/simulator/20_Tasks_Testset_v3_tiered.json`
Dreaming mode: `mock`

## Matching Metrics

| method | rho_last | delta vs MapScore | S_cap | S_need | MapScore |
|---|---:|---:|---:|---:|---:|
| scap_greedy | 0.9641 ± 0.0359 | +0.0000 | 0.9933 | 0.7486 | 0.9933 |
| mapscore_greedy | 0.9641 ± 0.0359 | +0.0000 | 0.9933 | 0.7486 | 0.9009 |
| mapscore_dreaming | 0.9641 ± 0.0359 | +0.0000 | 0.9933 | 0.7485 | 0.9009 |

## Outcome Metrics

| method | mutual | completion | req_sat | cand_sat | joint_reward | total_reward |
|---|---:|---:|---:|---:|---:|---:|
| scap_greedy | 0.6277 ± 0.0706 (+0.0000) | 0.7909 ± 0.0691 (+0.0000) | 0.7951 ± 0.0468 (+0.0000) | 0.6349 ± 0.0902 (+0.0000) | 1.0000 ± 0.0000 (+0.0000) | 0.8726 ± 0.0264 (+0.0000) |
| mapscore_greedy | 0.6277 ± 0.0706 (+0.0000) | 0.7909 ± 0.0691 (+0.0000) | 0.7951 ± 0.0468 (+0.0000) | 0.6349 ± 0.0902 (+0.0000) | 1.0000 ± 0.0000 (+0.0000) | 0.8726 ± 0.0264 (+0.0000) |
| mapscore_dreaming | 0.6277 ± 0.0706 (+0.0000) | 0.7908 ± 0.0691 (-0.0001) | 0.7951 ± 0.0468 (+0.0000) | 0.6349 ± 0.0902 (+0.0000) | 1.0000 ± 0.0000 (+0.0000) | 0.8726 ± 0.0264 (+0.0000) |

Values are mean ± std; parentheses show delta relative to `mapscore_greedy`.

## Process Metrics

| method | time feasibility | style compatibility | candidate usability | messages/task | API calls/task | tokens/task | wall-clock/task |
|---|---:|---:|---:|---:|---:|---:|---:|
| scap_greedy | 0.5426 ± 0.1071 (+0.0000) | 0.7699 ± 0.0468 (+0.0000) | 0.6454 ± 0.1048 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) |
| mapscore_greedy | 0.5426 ± 0.1071 (+0.0000) | 0.7699 ± 0.0468 (+0.0000) | 0.6454 ± 0.1048 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) |
| mapscore_dreaming | 0.5426 ± 0.1071 (+0.0000) | 0.7699 ± 0.0468 (+0.0000) | 0.6454 ± 0.1048 (+0.0000) | 12.0000 ± 0.0000 (+12.0000) | 12.0000 ± 0.0000 (+12.0000) | 6759.0000 ± 71.0000 (+6759.0000) | 0.0005 ± 0.0000 (+0.0005) |

## Dreaming Diagnostics

- rerank_rate: 0.0000
- delta_top1_s_cap: +0.0000
- delta_top1_s_need: +0.0000

Dream score correlations with outcomes:
- mutual_accept_probability: Pearson=0.0000, Spearman=0.0000
- completion_probability: Pearson=0.0000, Spearman=0.0000
- requester_satisfaction: Pearson=0.0000, Spearman=0.0000
- candidate_satisfaction: Pearson=0.0000, Spearman=0.0000
- joint_reward: Pearson=0.0000, Spearman=0.0000
- total_reward: Pearson=0.0000, Spearman=0.0000
