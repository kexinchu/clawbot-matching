# V3 Dreaming Ablation Summary

Generated at: `2026-07-12T17:52:48.456333`
Testset: `/root/autodl-tmp/CoWeaver/simulator/20_Tasks_Testset_v3_tiered.json`
Dreaming mode: `api`

## Matching Metrics

| method | rho_last | delta vs MapScore | S_cap | S_need | MapScore |
|---|---:|---:|---:|---:|---:|
| scap_greedy | 0.8683 ± 0.1087 | -0.1222 | 0.9930 | 0.5285 | 0.9930 |
| mapscore_greedy | 0.9905 ± 0.0187 | +0.0000 | 0.9453 | 0.8551 | 0.9113 |
| mapscore_dreaming | 0.9649 ± 0.0423 | -0.0256 | 0.9445 | 0.8002 | 0.8900 |

## Outcome Metrics

| method | mutual | completion | req_sat | cand_sat | joint_reward | total_reward |
|---|---:|---:|---:|---:|---:|---:|
| scap_greedy | 0.5368 ± 0.1032 (-0.0236) | 0.6790 ± 0.1466 (-0.0239) | 0.7098 ± 0.1193 (-0.0038) | 0.5353 ± 0.0837 (-0.0701) | 0.9575 ± 0.1276 (-0.0425) | 0.8074 ± 0.1057 (-0.0328) |
| mapscore_greedy | 0.5604 ± 0.0824 (+0.0000) | 0.7029 ± 0.1066 (+0.0000) | 0.7136 ± 0.0932 (+0.0000) | 0.6054 ± 0.0971 (+0.0000) | 1.0000 ± 0.0000 (+0.0000) | 0.8402 ± 0.0394 (+0.0000) |
| mapscore_dreaming | 0.5437 ± 0.0836 (-0.0167) | 0.6699 ± 0.1233 (-0.0330) | 0.6888 ± 0.1079 (-0.0248) | 0.5905 ± 0.0801 (-0.0149) | 0.9889 ± 0.0854 (-0.0111) | 0.8233 ± 0.0681 (-0.0169) |

Values are mean ± std; parentheses show delta relative to `mapscore_greedy`.

## Process Metrics

| method | time feasibility | style compatibility | candidate usability | messages/task | API calls/task | tokens/task | wall-clock/task |
|---|---:|---:|---:|---:|---:|---:|---:|
| scap_greedy | 0.5399 ± 0.1260 (-0.0017) | 0.6647 ± 0.1767 (-0.0248) | 0.4998 ± 0.1272 (-0.0744) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) |
| mapscore_greedy | 0.5416 ± 0.1158 (+0.0000) | 0.6895 ± 0.2010 (+0.0000) | 0.5742 ± 0.1290 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) | 0.0000 ± 0.0000 (+0.0000) |
| mapscore_dreaming | 0.5518 ± 0.1152 (+0.0102) | 0.7226 ± 0.1701 (+0.0331) | 0.5343 ± 0.1282 (-0.0399) | 12.0000 ± 0.0000 (+12.0000) | 12.0000 ± 0.0000 (+12.0000) | 10737.4833 ± 581.0657 (+10737.4833) | 43.8292 ± 7.0605 (+43.8292) |

## Dreaming Diagnostics

- rerank_rate: 0.4667
- delta_top1_s_cap: -0.0008
- delta_top1_s_need: -0.0549

Dream score correlations with outcomes:
- mutual_accept_probability: Pearson=-0.0230, Spearman=-0.0374
- completion_probability: Pearson=-0.1022, Spearman=-0.1457
- requester_satisfaction: Pearson=-0.1279, Spearman=-0.1939
- candidate_satisfaction: Pearson=0.1628, Spearman=0.2397
- joint_reward: Pearson=0.0364, Spearman=0.0000
- total_reward: Pearson=-0.0244, Spearman=-0.1058
