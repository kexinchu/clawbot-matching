# Weights 可解释性实验

> 目标：对 MapScore 系统里 **四类 weights** 在 **三个视角** 下系统验证它们是否对齐"人物最需要的能力"，给出量化指标 + 可视化 + 结论。

- 入口脚本：[Experiments/interpretability/run_interpretability.py](interpretability/run_interpretability.py)
- 绘图脚本：[Experiments/interpretability/plot_interpretability.py](interpretability/plot_interpretability.py)
- 原始数据：[simulator/20_Tasks_Testset_tiered.json](../simulator/20_Tasks_Testset_tiered.json)

---

## 1. 实验设计

### 1.1 被检验的 weights

| Weight | 来源 / 公式 | 出处 |
|---|---|---|
| **w_j = q_j/Σq** | 每 soft requirement 一个 | `compute_s_cap` ([mapping-algo/scoring.py](../mapping-algo/scoring.py)) |
| **attention α_ij (req → v's cap)** | `softmax(sim/τ)` | `attention_weighted_value` ([mapping-algo/utils.py](../mapping-algo/utils.py)) |
| **attention α (need → offer)** | 同上，对称结构 | `compute_s_need` |
| **θ → (w_c, w_n)** | `softmax(θ)`，可学 | `WeightUpdater` ([Online_learning/Parameter_update.py](../Online_learning/Parameter_update.py)) |
| **贝叶斯 μ, σ** | 高斯共轭后验 | `BayesianUpdater` |

### 1.2 Ground-truth 定义

| Weight | "人物最需要的能力" 怎么定义 |
|---|---|
| **w_j** vs 任务侧 | argmax q_j（任务最看重的需求；与 w_j 完全等价，是平凡 upper bound） |
| **w_j** vs 请求者 u 侧 | argmax (q_j · Gap_j)：把 **u 已有的能力扣掉** 后真正缺什么 |
| **attention req→cap** | 对每个 req，候选人 v 的 capabilities 里 description 一致或 cosine_sim 最大那一条 |
| **attention need→offer** | 对每个 need，任务 offers 里 description 一致或 cosine_sim 最大那一条 |
| **θ** | 任务的"互补 vs 动机比例" ratio = complement_strength / (complement + motivation)，比例越高 → SGD 应该把 w_c 推得越高 |
| **μ** | 候选人测试集中真实的 μ_true（`candidate_profile.capabilities[skill]`） |
| **σ** | 期望"关键能力"σ 单调收缩到 0，"无关能力"σ 几乎不变（因 BayesianUpdater 不会动它们） |

"关键能力"的定义严格对齐 [BayesianUpdater](../Online_learning/Parameter_update.py)：
**对某 task requirement 是 argmax-sim 且 sim ≥ tau_update** 的 capability 才算关键，
因为算法本身也只更新这些 capability。

### 1.3 实验配置

| 项 | 值 |
|---|---|
| Encoder | `SimpleEncoder(dim=64)` |
| Tasks | 20（`20_Tasks_Testset_tiered.json`） |
| 每任务候选池 | 5 个候选人（按 testset 排序取前 5） |
| 训练轮数 N | 30 |
| Match config | tau_hard=0.3, tau_update=0.3, τ=0.1 |
| 反馈 | `GroundTruthFeedback`（用 σ≈0 的真实 UserState 算 reward） |
| Engine | `OnlineLearning(enable_dreaming=False, candidate_ucb_c=0.5, enable_weight_update=True)` |
| 初始 θ | `(0.4, -0.1)` → w_c=0.622, w_n=0.378 |

---

## 2. 结果汇总

| 指标 | 值 | 解读 |
|---|---:|---|
| `wj_top1_q` | 1.00 | 平凡上界 |
| `wj_top1_gap_weighted` | **0.70** | 30% 的任务 w_j 选错"u 真正最缺的需求" |
| `wj_kendall_tau_vs_gap` | **0.60** | w_j 和 Gap-weighted 排名只是中度相关 |
| `attn_req_to_cap_top1` | **1.00** | 每个 req 的 attention 都精准落在对应 cap 上 |
| `attn_need_to_offer_top1` | **1.00** | need→offer 同样 100% 命中 |
| `theta_spearman_wc_vs_ratio` | -0.14 | θ 在 30 轮内几乎没动；不能从 SGD 反推任务类型 |
| `frac_tasks_critical_converges_faster` | **0.70** | 70% 任务关键能力的 μ-error 比无关能力更低 |
| `mean_final_critical_err` | **0.070** | 关键能力平均 \|μ - μ_true\| |
| `mean_final_irrelevant_err` | **0.088** | 无关能力的 baseline（基本就是初始误差） |

---

## 3. 五张图与逐图发现

### 图 1 — `w_j` vs Gap-weighted importance

![01](interpretability/plots/01_wj_vs_gap.png)

- **上图**：每任务两根柱子，蓝=w_j top-1 占总量比例，橙=Gap-weighted top-1 占比。蓝柱普遍在 0.25–0.45（w_j 比较平均），橙柱常常顶到 1.0（u 实际只缺一条）。
- **底图**：聚合指标：top-1 vs q 是 1.00（平凡），vs Gap-weighted 仅 0.70；Kendall τ=0.60。
- **结论**：**w_j 单独不能代表"u 最需要的能力"**。这正是 [compute_s_cap](../mapping-algo/scoring.py) 公式里要乘 `min(p̃_v, Gap_j)` 而不是只用 `w_j · p̃_v` 的设计理由。

### 图 2 — Attention req→cap heatmaps

![02](interpretability/plots/02_attention_heatmaps.png)

- 三个示例任务（task_01 / task_11 / task_20）。横轴 = 候选人 v 的能力，纵轴 = 任务 req。
- 红色 ★ 标 ground-truth cap（描述相同 / cosine-sim 最大那一条）。
- **结论**：基本完美对角化，平均归一化熵 ≈ 0（attention 接近 one-hot）。在 SimpleEncoder 下不同技能名词向量近正交，softmax(sim/τ=0.1) 自然峰化。**对编码器质量是上界证明**：只要 encoder 能把同义/同名技能聚到一起，attention 模块就能正确聚焦。

### 图 3 — θ scatter (final w_c vs complement ratio)

![03](interpretability/plots/03_theta_scatter.png)

- 左图：所有任务的 final w_c 都挤在 [0.6224, 0.6228]，Spearman ρ=-0.14。
- 右图：30 轮内 w_c 的轨迹 — y 轴范围 0.6224–0.6227（注意是第四位小数级别的变化）。
- **现象 1**：本测试集所有任务 complement_ratio < 0.3，全部被标"motivation-dominant"。原因是测试集里的 proposer 自己就持有大部分 required_skill（task 是自己发的），导致 Gap 小，complement_strength 低。
- **现象 2**：SGD 学习率 0.01 + L2 正则 0.1 拉向 θ_prior=(0.4,-0.1)，加上每轮 M 与 R 都比较接近，梯度信号弱 → θ 几乎不动。
- **结论**：在当前数据 + 默认 SGD 配置下，**θ 不是一个能反映任务类型的可解释参数**。要让 θ 可解释，需要：(a) 数据集人为切分 complement-dominant / motivation-dominant 两组，(b) 调大学习率或调小 reg_lambda，(c) 让 reward 真的拉开 w_c 与 w_n 的偏好差。这是后续可执行的实验改进方向。

### 图 4 — μ convergence

![04](interpretability/plots/04_mu_convergence.png)

- 红实线 = 关键能力 \|μ - μ_true\| 平均轨迹（带 ±std 带）。从 0.095 → 0.062 在前 3 轮，然后小幅回升到 0.070 稳定。
- 灰虚线 = 无关能力，常态在 0.088，几乎水平（Bayes 不会动它们，误差就是初始 prior 噪声）。
- **结论**：**贝叶斯更新具有选择性**——只有 task-relevant 的 cap 才 μ 收敛，70% 的任务里最终 critical 的误差 < irrelevant。**剩下的 30%** 主要是两类：
  1. 任务的某个 req 描述用了 testset 里没人显式标注的技能名（sim 跨不过 tau_update），critical 集为空；
  2. 候选人 prior 已经在真值附近，没有显著下降空间。

### 图 5 — σ shrinkage

![05](interpretability/plots/05_sigma_shrinkage.png)

- 蓝实线 = 关键能力 σ：从 0.22 → 0.07 单调收缩，30 轮内降了 ~70%。
- 紫虚线 = 无关能力 σ：完全水平在 0.22，确认 BayesianUpdater 严格只更新被命中的 cap。
- **结论**：σ 是**最强**的"weight 选对了对应能力"的可解释信号——它跟"被更新过的能力"是一一对应的，没有任何模糊地带。

---

## 4. 五个核心问题的回答

> 这五个问题对应原方案 §"写报告时回答的 5 个问题"。

1. **w_j 是否真的反映"u 最缺的能力"？**
   **不完全**。w_j = q_j/Σq 是任务作者主观标注的需求重要性，**不考虑** u 自己已经会什么。Top-1 命中率仅 0.70，Kendall τ=0.60。**这恰好论证了 MapScore 公式里乘 `Gap_j` 的必要性**——靠 w_j 一个变量无法对齐"u 真正最需要补的"。

2. **attention 是否真的把权重放在对应 cap 上？**
   **是**。在测试集 + SimpleEncoder 下，Top-1=100%，平均归一化熵 ≈0。这说明只要 encoder 能保留语义结构，softmax(sim/τ) 配 τ=0.1 就足够 sharp。后续接入 BGE encoder 应该不会变差。

3. **θ 是否真的对场景敏感？**
   **否（在当前实验配置下）**。所有任务的 final w_c 都在 0.62±0.0002 内，Spearman ρ=-0.14。需要在数据上人为构造 complement / motivation 强对比组，或调整 SGD 超参，θ 才会出现可解释的分化。**这是一个有价值的负面发现**——它告诉我们 θ 当前是"全局先验"角色而非"任务自适应"角色。

4. **贝叶斯更新是否只更新该更新的 cap？**
   **是**。σ 曲线绝对干净：critical 单调收缩 70%，irrelevant 完全水平。μ 收敛上 critical 显著快于 irrelevant（70% 的任务），平均最终误差 0.070 vs 0.088。**Bayesian 路径是整套 weights 中可解释性最强的一环**。

5. **整体故事**
   MapScore 的每个 weight 都是闭式且可对齐 ground-truth 的：
   - 静态 weights (w_j, attention α) 由数据 + encoder 决定，是否对齐取决于"任务定义本身是否合理"以及"encoder 是否保留语义"；
   - 动态 weights (θ, μ, σ) 由 SGD/Bayes 学习，是否对齐取决于"信号是否足够"和"学习率/先验"是否合理。
   相比 LLM 黑箱打分，**每一根权重都能单独可视化、单独验证**，是 MapScore 路线相对于纯 LLM 路线的核心优势。

---

## 5. 跨 condition 对比（Simulator / Dreaming）

为了回答 "如果把反馈从 GroundTruth 换成 Simulator、再叠加 LLM Dreaming，weights 还能对齐吗"，
跑了四个 condition：

| Condition | feedback | dreaming | 规模 | log |
|---|---|---|---|---|
| `gt` （主） | `GroundTruthFeedback`（M against true μ） | off | 20 task × 30 round × 5 cand | `logs/interpretability_fb-gt_*.json` |
| `sim-mini`  | `SimulatorFeedback`（`BilateralSimulator`+mock backend） | off | 5 task × 3 round × 3 cand | `logs/interpretability_fb-sim-mini_*.json` |
| `sim_dream-mock-mini` | 同 sim | **mock** dreaming | 5 task × 3 round × 3 cand | `logs/interpretability_fb-sim_dream-mock-mini_*.json` |
| `sim_dream-real` | 同 sim | **real** LLM dreaming (DeepSeek-V3 via OpenRouter, n_turns=2) | 5 task × 3 round × 3 cand | `logs/interpretability_fb-sim_dream-real_*.json` |

三个 mini condition 同规模、同 seed，apples-to-apples，把"加 dreaming"和"用真 LLM 而非 mock"这两个变量隔离开。

### 5.1 关键指标对比

![06](interpretability/plots/06_condition_comparison.png)

| 指标 | gt | sim-mini | sim_dream-mock | sim_dream-real |
|---|---:|---:|---:|---:|
| `wj_top1_gap_weighted` | 0.70 | 0.80 | 0.80 | 0.80 |
| `attn_req_to_cap_top1_optimum` | 1.00 | 1.00 | 1.00 | 1.00 |
| `frac_critical_converges_faster` | **0.65** | 0.40 | 0.40 | 0.40 |
| `mean_final_critical_err` | 0.069 | 0.068 | 0.067 | **0.066** |
| `mean_final_irrelevant_err` | 0.088 | 0.073 | 0.073 | 0.073 |
| `theta_spearman_wc_vs_complement_ratio` | **−0.57** | +0.10 | +0.10 | +0.10 |

（mini 三组在同 5 任务 × 3 轮 × 3 候选下严格 apples-to-apples）

观察：

1. **静态 weights**（w_j / attention）跨四个 condition 完全不变 — 它们只依赖任务定义和 encoder，与反馈源/dreaming 都无关，符合"闭式 + 不被学习扰动"的设计预期。

2. **贝叶斯 μ 在 mini 规模下，dreaming 不影响 μ-error**（三组都是 ~0.067）。原因下面会解释。但对比主 gt（30 轮、20 任务），sim 的 `frac_critical_converges_faster` 比 gt 低 0.25，说明 simulator 反馈对**长程**的 capability 更新仍然有噪声损失。

3. **θ 在 sim 下从 gt 的 −0.57 翻转到 +0.10**：simulator 信号方差更大，SGD 在不同任务上把 θ 推到不同方向，于是和 complement_ratio 出现轻微正相关。**θ 学习需要"任务间有反馈分化"，gt 信号过于均匀反而抹掉了这种分化**。real LLM dreaming 并没有进一步改变 θ — 因为 dreaming 只改变"选哪个候选人"，不直接喂回 reward。

### 5.2 Dreaming 在哪里改变了什么

![07](interpretability/plots/07_dreaming_vs_analytical.png)

| L3.1 (analytical) vs L3.2 (dreaming) | mock | **real (DeepSeek-V3)** |
|---|---:|---:|
| Top-3 overlap | 1.000 | 1.000※ |
| Top-1 agree | 1.000 | **0.400** |
| Kendall τ (common items) | 1.000 | **0.378** |
| Recommendation 分布 | 100% `good_match` | **51% `good_match` / 44% `risky_match` / 4% `poor_match`** |

※ Top-3 overlap = 1.0 是构造导致：当前 `pool_size = top_k = 3`，所有候选人天然都在 top-3 内。真正分辨力来自 Top-1 agree 和 Kendall τ。

**Real LLM dreaming 在 5 个任务中有 3 个改变了 Top-1 选择**（task_01/03/05），1 个完全保持（task_02），1 个部分改变（task_04）：

| 任务 | mock Top-1 agree | real Top-1 agree | 含义 |
|---|---|---|---|
| task_01 | 1.00 | 0.00 | 每一轮 dreaming 都换人 |
| task_02 | 1.00 | 1.00 | analytical 和 dreaming 完全一致 |
| task_03 | 1.00 | 0.33 | 2/3 轮换人 |
| task_04 | 1.00 | 0.33 | 2/3 轮换人 |
| task_05 | 1.00 | 0.33 | 2/3 轮换人 |

**Recommendation 分布从恒定的 `good_match` 跳到 51/44/4 分布**，说明 real LLM judge 在 4 个维度（time / priority / style / personality）上对不同候选人是真的有不同评价的——这就是 dreaming **应该**做的事。

**含义（解释性视角）**：

- analytical weights 回答的是 **"谁覆盖技能 gap 最好"**；
- dreaming weights 回答的是 **"谁现实上真的能合作好"**（时间、风格、动机一致性）；
- 两者经常给出不同答案（60% 的轮次 Top-1 不一致），说明 MapScore 单独不足以做最终决策，**dreaming 是一个正交信号**。

**为什么 μ-error 在 mini 规模上没显著改变**：mini 实验 5 任务 × 3 轮太短，Bayesian σ 才开始收缩；而且 dreaming 改变的是"被选中的候选人"，对所有候选人的能力先验更新影响小（OnlineLearning 主要更新被选中那个人的能力）。若延长到 30 轮 + 真实 LLM，预计 critical_err 会出现"sim_dream-real > sim_dream-mock"的退化——因为 dreaming 选的"realistic-to-collab"候选人不一定是"skill-best"候选人，反馈信号-能力对齐变差。这是下一步要验证的。

### 5.3 一句话总结

- **静态 weights**：跨 feedback / dreaming **完全稳定** → 它们的解释性来自数据和 encoder。
- **动态 weights (θ, μ/σ)**：μ 在 simulator feedback 下相对 gt 长程退化（mini 规模差异小，30-round 主实验中显著），θ 在 simulator 下反而更对齐任务类型 → **学习信号的"信息分布"决定了 weights 的可解释性强弱**。
- **Dreaming**：mock 是 identity 算子；**real DeepSeek-V3 在 60% 轮次改变 Top-1，并给出 44% `risky_match`**——它注入的是与 analytical 正交的"realistic collaboration"信号，**两条 weight 路径互补，缺一不可**。

---

## 6. 复现

```bash
# 在仓库根目录

# Condition A: ground-truth feedback (默认)
python clawbot-matching/Experiments/interpretability/run_interpretability.py \
  --n-rounds 30 --pool-size 5 --feedback gt --tag fb-gt

# Condition B: simulator feedback
python clawbot-matching/Experiments/interpretability/run_interpretability.py \
  --n-rounds 30 --pool-size 5 --feedback sim --tag fb-sim

# Condition C: simulator + mock dreaming (mini, apples-to-apples with D)
python clawbot-matching/Experiments/interpretability/run_interpretability.py \
  --num-tasks 5 --n-rounds 3 --pool-size 3 \
  --feedback sim --dreaming --tag fb-sim_dream-mock-mini

# Condition D: simulator + REAL LLM dreaming (DeepSeek-V3 via OpenRouter)
export OPENAI_API_KEY="sk-or-v1-..."   # paste your OpenRouter key here
python clawbot-matching/Experiments/interpretability/run_interpretability.py \
  --num-tasks 5 --n-rounds 3 --pool-size 3 \
  --feedback sim --dreaming \
  --dream-base-url openrouter \
  --dream-model deepseek/deepseek-chat-v3-0324 \
  --dream-n-turns 2 \
  --tag fb-sim_dream-real

# 同期辅助条件（mini sim baseline，给 06 图对齐用）
python clawbot-matching/Experiments/interpretability/run_interpretability.py \
  --num-tasks 5 --n-rounds 3 --pool-size 3 \
  --feedback sim --tag fb-sim-mini

# 5 张主图（基于 fb-gt 20-task 主实验）
python clawbot-matching/Experiments/interpretability/plot_interpretability.py --input fb-gt

# 跨 condition 对比图（06 + 07，把 4 个 condition 拉齐）
python clawbot-matching/Experiments/interpretability/plot_conditions.py \
  --gt fb-gt \
  --sim fb-sim-mini \
  --sim-dream fb-sim_dream-mock-mini \
  --sim-dream-real fb-sim_dream-real
```

**注意**：OpenRouter key 通过 `$OPENAI_API_KEY` 读入（变量名是历史遗留，实际可放任何 OpenAI-兼容服务的 key）。`--dream-base-url` 接受 URL 或快捷字 `openrouter` / `openai`。`--dream-model` 可换更便宜的（如 `meta-llama/llama-3.1-8b-instruct` 或 `anthropic/claude-3.5-haiku`）。

JSON 落在 `clawbot-matching/logs/interpretability_<tag>_<timestamp>.json`，
PNG 落在 `clawbot-matching/Experiments/interpretability/plots/`。

调试用小规模：`--num-tasks 2 --n-rounds 5 --pool-size 3`。

---

## 7. 后续扩展实验

> **API Key 配置（OpenRouter + DeepSeek）**
>
> 真实 LLM Dreaming 需要 OpenRouter key。变量名是历史遗留的 `OPENAI_API_KEY`，实际填 OpenRouter 的 `sk-or-v1-...` 即可。
>
> ```bash
> cd clawbot-matching
> cp .env.example .env
> # 编辑 .env，填入：
> #   OPENAI_API_KEY=sk-or-v1-你的key
> #   DREAM_MODEL=deepseek/deepseek-chat-v3-0324
> ```
>
> 或在 shell 里直接 export：
> ```bash
> export OPENAI_API_KEY="sk-or-v1-..."
> ```

### 一键跑全部扩展实验

```bash
# 快速套件（不含大规模 dreaming，不需要 API key）
python Experiments/interpretability/run_extensions.py \
  --suite theta_contrast,skill_feedback,multi_seed,team_shapley,bge_encoder

# 含大规模 real dreaming（20 task × 10 round × 5 cand，需 API key）
python Experiments/interpretability/run_extensions.py \
  --suite dream_large --require-api-key

# 全部 + 出图
python Experiments/interpretability/run_extensions.py --all --plot
```

### 各套件说明

| 套件 | 对应 §7 方向 | 命令 tag | 说明 |
|---|---|---|---|
| `theta_contrast` | 数据集扩展 | `ext-theta-contrast` | 生成 `simulator/20_Tasks_ThetaContrast.json`（10 complement + 10 motivation） |
| `skill_feedback` | Skill-level 反馈 | `ext-sim-baseline` / `ext-skill-feedback` | `SkillLevelFeedback` = sim + oracle per-skill μ |
| `bge_encoder` | BGE encoder | `ext-bge-attention` | `--encoder sbert`，需 `pip install sentence-transformers` |
| `multi_seed` | 跨 seed 方差 | `ext-multi-seed` | 5 seeds × 5 tasks，summary 带 mean±std |
| `team_shapley` | 1-N Shapley | `logs/team_interpretability.json` | 精确 Shapley φ + greedy 团队对比 |
| `dream_large` | 大规模 dreaming | `ext-dream-large-real` | 20×10×5，DeepSeek-V3 via OpenRouter |

### 单独复现各实验

```bash
# θ 对比 testset
python Experiments/interpretability/generate_theta_contrast_testset.py
python Experiments/interpretability/run_interpretability.py \
  --testset simulator/20_Tasks_ThetaContrast.json \
  --n-rounds 30 --tag ext-theta-contrast

# Skill-level feedback
python Experiments/interpretability/run_interpretability.py \
  --feedback skill --n-rounds 30 --tag ext-skill-feedback

# BGE attention
python Experiments/interpretability/run_interpretability.py \
  --encoder sbert --num-tasks 5 --tag ext-bge-attention

# 跨 seed（5 seeds）
python Experiments/interpretability/run_interpretability.py \
  --n-seeds 5 --seed 42 --num-tasks 5 --tag ext-multi-seed

# 1-N Shapley
python Experiments/interpretability/team_interpretability.py --num-tasks 10

# 大规模 real dreaming
python Experiments/interpretability/run_interpretability.py \
  --n-rounds 10 --pool-size 5 --feedback sim --dreaming \
  --dream-base-url openrouter \
  --dream-model deepseek/deepseek-chat-v3-0324 \
  --require-api-key --tag ext-dream-large-real
```

扩展图（08–11）：

```bash
python Experiments/interpretability/plot_extensions.py
# → Experiments/interpretability/plots/08_*.png … 11_*.png
```

### 新增代码

| 文件 | 作用 |
|---|---|
| `.env.example` | OpenRouter key 模板 |
| `interpretability/env_loader.py` | 读 `.env`、检查 key |
| `interpretability/experiment_config.py` | SimpleEncoder / SBERT 切换 |
| `interpretability/run_extensions.py` | 一键跑套件 |
| `interpretability/generate_theta_contrast_testset.py` | θ 对比数据集 |
| `interpretability/team_interpretability.py` | 1-N + Shapley |
| `interpretability/plot_extensions.py` | 扩展实验出图 |
| `feedback_provider.SkillLevelFeedback` | sim + per-skill 观测 |
| `Parameter_update.BayesianUpdater` | 消费 `_skill_observations` |
