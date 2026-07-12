# V3 Dreaming Ablation Summary

Generated at: `2026-07-12T16:58:06.131689`
Testset: `/root/autodl-tmp/CoWeaver/simulator/20_Tasks_Testset_v3_tiered.json`
Dreaming mode: `mock`

## Matching Metrics

| method | rho_last | delta vs MapScore | S_cap | S_need | MapScore |
|---|---:|---:|---:|---:|---:|
| scap_greedy | 0.8683 ± 0.1087 | -0.1222 | 0.9930 | 0.5285 | 0.9930 |
| mapscore_greedy | 0.9905 ± 0.0187 | +0.0000 | 0.9453 | 0.8551 | 0.9113 |
| mapscore_dreaming | 0.9905 ± 0.0187 | +0.0000 | 0.9453 | 0.8551 | 0.9113 |

## Outcome Metrics

| method | mutual | completion | req_sat | cand_sat | joint_reward | total_reward |
|---|---:|---:|---:|---:|---:|---:|
| scap_greedy | 0.5351 ± 0.1055 (-0.0216) | 0.6790 ± 0.1466 (-0.0234) | 0.7091 ± 0.1215 (-0.0040) | 0.5356 ± 0.0840 (-0.0697) | 0.9575 ± 0.1276 (-0.0425) | 0.8077 ± 0.1054 (-0.0313) |
| mapscore_greedy | 0.5567 ± 0.0835 (+0.0000) | 0.7024 ± 0.1067 (+0.0000) | 0.7131 ± 0.0966 (+0.0000) | 0.6053 ± 0.0976 (+0.0000) | 1.0000 ± 0.0000 (+0.0000) | 0.8390 ± 0.0408 (+0.0000) |
| mapscore_dreaming | 0.5566 ± 0.0835 (-0.0001) | 0.7024 ± 0.1067 (+0.0000) | 0.7131 ± 0.0966 (+0.0000) | 0.6053 ± 0.0976 (+0.0000) | 1.0000 ± 0.0000 (+0.0000) | 0.8390 ± 0.0408 (+0.0000) |

Values are mean ± std; parentheses show delta relative to `mapscore_greedy`.

## Process Metrics

| method | time feasibility | style compatibility | candidate usability | messages/task | API calls/task | tokens/task | wall-clock/task |
|---|---:|---:|---:|---:|---:|---:|---:|
| scap_greedy | 0.5331 ± 0.1254 (-0.0023) | 0.6136 ± 0.1712 (-0.0284) | 0.5064 ± 0.1253 (-0.0775) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) |
| mapscore_greedy | 0.5354 ± 0.1174 (+0.0000) | 0.6420 ± 0.1932 (+0.0000) | 0.5839 ± 0.1286 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) |
| mapscore_dreaming | 0.5354 ± 0.1174 (+0.0000) | 0.6420 ± 0.1932 (+0.0000) | 0.5839 ± 0.1286 (+0.0000) | 12.0000 ± 0.0000 (+12.0000) | 12.0000 ± 0.0000 (+12.0000) | 6856.9000 ± 109.0577 (+6856.9000) | 0.0005 ± 0.0000 (+0.0005) |

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
