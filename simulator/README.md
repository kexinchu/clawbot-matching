# Bilateral Matching Simulator

一个模块化的双边匹配模拟器，通过多 persona 内部 deliberation 模拟 requester 和 candidate 两侧的决策过程，输出可量化的 reward 信号，可用于 Layer 5 RLHF 学习管道。

**默认使用 Mock 后端（离线、无需 API key），也可切换到真实 LLM（OpenAI / Anthropic / vLLM）。**

---

## 目录结构

```
simulator/
├── __init__.py              # 公开 API 统一导出
├── types.py                  # 所有 dataclass 类型定义
├── config.py                 # SimulatorConfig 所有超参数
│
├── persona_generator.py      # 多 persona 生成（Requester / Candidate）
├── persona_judge.py          # persona 级别 judgment
├── deliberation.py           # 多轮 deliberation 引擎
├── importance_ranker.py      # DyLAN 风格 importance 排名 + top-k 剪枝
├── decision_model.py         # 效用聚合 + action 映射
│
├── bilateral_simulator.py    # 双边模拟编排（完整 pipeline）
├── outcome_simulator.py      # 事后交互结果模拟
├── reward.py                 # RewardResult 构建（三分量加权）
│
├── llm_backend.py            # BaseBackend 抽象类 + LLM 后端实现
├── mock_backend.py           # RuleBasedBackend（默认，离线）
├── prompts.py                # persona 生成 prompt 模板
├── utils.py                  # softmax、sampling、TraceLogger 等工具
│
├── demo.py                   # 端到端演示脚本
├── result/
│   └── simulate.py           # 批量运行脚本（支持多 seed）
│
├── tests/
│   ├── test_persona.py
│   ├── test_deliberation.py
│   ├── test_importance.py
│   ├── test_bilateral.py
│   └── test_reward.py
│
└── requirements.txt
```

---

## 核心概念

### Pipeline 概览

```
MatchingContext
    │
    ├── [Stage 1] Persona Generation
    │       RequesterGenerator → 4 个 Requester Persona
    │       CandidateGenerator → 4 个 Candidate Persona
    │
    ├── [Stage 2] Deliberation（每侧独立进行）
    │       Round 0: 每个 persona 给出初始 opinion
    │       Round 1+: Leader 总结冲突，各 persona 修订意见
    │       提前停止：majority accept/reject 概率 > threshold
    │
    ├── [Stage 3] Importance Ranking
    │       按 6 个信号打分（relevance、non_redundancy、confidence、
    │       consistency、decision_impact、decision_flip）
    │       softmax 归一化 + top-k 剪枝
    │
    ├── [Stage 4] Side Decision
    │       选中 persona 加权聚合 → utility + action_probs
    │       utility → accept / skip / reject
    │
    └── [Stage 5] Joint Action
            requester_action × candidate_action → JointAction
            joint_accept_prob = P_req(accept) × P_cand(accept)
```

### Action 空间

| Action | 语义 |
|--------|------|
| `accept` | 愿意合作 |
| `skip` | 保留观望 |
| `reject` | 明确拒绝 |

### Joint Action 空间

| JointAction | 语义 |
|-------------|------|
| `mutual_accept` | 双方均接受 ✅ |
| `requester_accept_candidate_skip` | requester 接受，candidate 观望 |
| `requester_skip_candidate_accept` | requester 观望，candidate 接受 |
| `mutual_skip` | 双方均观望 |
| `one_reject` | 一方拒绝 |
| `mutual_reject` | 双方均拒绝 ❌ |

### Reward 构成

```
total_reward = 0.40 × feedback_reward
             + 0.20 × efficiency_reward
             + 0.40 × quality_reward

feedback_reward   ∈ [-0.5, 1.0]   # 基于 JointAction 语义梯度
efficiency_reward ∈ [0, 1]         # 预期协商轮数越少越高
quality_reward   ∈ [0, 1]         # 完成率 + 双方满意度加权
```

### Layer-2 盲信号（Latent Signals）

`MatchingContext` 中包含 requester/candidate **不可见**的隐变量：

```python
latent_interpersonal_affinity   # 人际亲和度 [-1, 1]
latent_risk_tolerance           # 风险偏好 [0, 1]
latent_opportunity_bias         # 机会成本偏差 [-1, 1]
```

这些信号注入 soft-persona 的评分，但不暴露给 Layer 2，以减少自证循环风险。

---

## 快速开始

### 安装依赖

```bash
pip install -r simulator/requirements.txt
```

`requirements.txt` 只包含核心依赖，无需 LLM API key 即可运行 mock 模式。

### 运行 Demo（单次，mock 后端）

```bash
python -m simulator.demo
```

输出：persona 列表 → deliberation trace → side decisions → outcome → reward。

### 运行 Demo（真实 LLM）

```bash
# 在 ~/.bashrc 中设置
export OPENROUTER_API_KEY=sk-or-v1-xxxxx

# 运行
python -m simulator.demo
```

`demo.py` 默认使用 `backend_type="openai"` + `model_name="deepseek/deepseek-chat-v3-0324"`，可通过修改 `config` 切换模型。

### 批量模拟

```bash
# 单次运行，seed=42
python simulator/result/simulate.py

# 单次运行，指定 seed
python simulator/result/simulate.py --seed 123

# 多种子批量运行（6 seeds）
python simulator/result/simulate.py --multi --n-runs 6

# 使用 LLM 后端
python simulator/result/simulate.py --openai

# 完整参数示例
python simulator/result/simulate.py \
  --multi --n-runs 10 \
  --openai \
  --n-personas 4 \
  --max-rounds 3 \
  --top-k 3 \
  --out-dir result/my_run
```

结果保存在 `result/sim_YYYYMMDD_HHMMSS/` 目录下，每个 run 的 JSON 文件包含完整 trace。

---

## 配置参数（SimulatorConfig）

```python
# Persona 数量
num_requester_personas: int = 4
num_candidate_personas: int = 4

# Deliberation
max_deliberation_rounds: int = 3       # 最大轮数
early_stop_threshold: float = 0.70     # majority 概率 > 此值则提前停止

# Importance 排名
top_k_personas: int = 3
importance_temperature: float = 1.5    # softmax 温度

# Decision
decision_mode: "probabilistic" | "threshold"
accept_utility_threshold: float = 0.3
reject_utility_threshold: float = -0.3

# Reward 权重
reward_feedback_weight: float = 0.40
reward_efficiency_weight: float = 0.20
reward_quality_weight: float = 0.40

# Backend
backend_type: "mock" | "openai" | "anthropic" | "vllm"
model_name: str = "deepseek/deepseek-chat-v3-0324"

# Misc
random_seed: int = 42
trace_verbose: bool = True
```

---

## 后端类型

| backend_type | 说明 | 需要 API key |
|---|---|---|
| `mock`（默认）| 规则生成，完全离线 | 否 |
| `openai` | OpenRouter API（支持 deepseek、gpt-4o 等） | 是 |
| `anthropic` | Anthropic Claude API | 是 |
| `vllm` | 自托管 vLLM 服务器 | 是 |

切换后端只需修改 `SimulatorConfig.backend_type` 和 `model_name`，无需改动其他代码。

---

## 测试

```bash
pytest simulator/tests/ -v
```

所有测试文件共 5 个，覆盖 persona 生成、deliberation、importance ranking、bilateral simulator 和 reward 构造。Mock 后端具有确定性（固定 seed），测试结果可复现。

---

## 核心模块说明

### `bilateral_simulator.py` — BilateralSimulator

编排完整 pipeline：
1. 生成两侧 persona
2. 运行两侧 deliberation（独立）
3. 重要性排名 + top-k 剪枝
4. 计算两侧决策
5. 组合为 joint action

### `deliberation.py` — DeliberationEngine

- Round 0：所有 persona 并行给出初始 opinion
- Round 1+：Leader 总结当前分歧，各 persona 修订意见
- 提前停止条件：majority accept/reject 概率 ≥ `early_stop_threshold`

### `importance_ranker.py` — rank_persona_importance

6 个信号综合打分：
1. **relevance** — persona 关注维度与 task 的相关度
2. **non_redundancy** — 与其他 persona 关注点的 Jaccard 差异度
3. **confidence** — 该 persona 的置信度
4. **consistency** — 与群体均值的偏差
5. **decision_impact** — utility 绝对值（决策影响力度）
6. **decision_flip** — Leave-one-out 效用变化 × 阈值穿越信号

### `decision_model.py` — SideDecisionEngine

- 加权聚合选中 persona 的 utility → side utility
- 加权聚合 action_probs → action 概率分布
- 概率模式下从 action_probs 采样 action；阈值模式下用 utility 硬阈值

### `reward.py` — compute_reward

三分量 reward：
- `feedback_reward`：JointAction 语义梯度表（`mutual_accept`=1.0 → `mutual_reject`=-0.95）
- `efficiency_reward`：基于 expected_rounds（轮数越少越高）
- `quality_reward`：completion_prob × 40% + requester_satisfaction × 30% + candidate_satisfaction × 30%

### `llm_backend.py` — OpenAIBackend

`generate_personas()`：从模板池采样 persona，通过 LLM 注入上下文相关性
`generate_opinion()`：带 JSON schema 的结构化生成，返回 `PersonaOpinion`
`summarize_discussion()`：自由文本生成 leader 总结
`revise_opinion()`：结构化生成修订意见，带规则 fallback

---

## 扩展指南

### 添加新的 Backend

```python
from simulator.llm_backend import BaseBackend

class MyBackend(BaseBackend):
    def __init__(self, ...):
        ...

    # 实现 BaseBackend 的所有抽象方法
    def generate(self, prompt: str, **kwargs) -> str: ...
    def structured_generate(self, prompt: str, schema: dict, **kwargs) -> dict: ...
    def generate_personas(self, side: SideType, context: MatchingContext,
                          num: int, persona_seed: int) -> list[PersonaSpec]: ...
    def generate_opinion(self, persona: PersonaSpec, context: MatchingContext,
                         round_idx: int) -> PersonaOpinion: ...
    def summarize_discussion(self, opinions: list[PersonaOpinion],
                             round_idx: int) -> str: ...
    def revise_opinion(self, opinion: PersonaOpinion, other_opinions: list[PersonaOpinion],
                        summary: str, round_idx: int) -> PersonaOpinion: ...
```

### 添加新的 Persona 模板

在 `mock_backend.py` 的 `REQUESTER_PERSONA_TEMPLATES` / `CANDIDATE_PERSONA_TEMPLATES` 中添加 `PersonaSpec`，或通过 LLM backend 的模板池注入。

### 调整 Reward 权重

修改 `SimulatorConfig.reward_feedback_weight`、`reward_efficiency_weight`、`reward_quality_weight`，或在 `reward.py` 中调整子 reward 的内部逻辑。
