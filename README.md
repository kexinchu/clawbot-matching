# MapScore Matching Algorithm

双边匹配核心算法，实现 MBRL 架构的 Layer 2（World Model）和 Layer 3.1（Coarse Filtering）。

**核心公式：**

```
M(u, v, T) = σ(u,v,T) · (w_c · S_cap(v,u,T) + w_n · S_need(v,T))
```

- `σ` — 二值门控（安全等级 + 硬约束）
- `S_cap` — v 对 u 的能力缺口覆盖度 ∈ [0,1]
- `S_need` — 任务 T 对 v 的需求满足度 ∈ [0,1]
- `w_c + w_n = 1`，由 `softmax(θ)` 保证

---

## 目录结构

```
mapping-algo/
├── datatypes.py        # 核心数据结构（UserState、Task、MatchResult 等）
├── config.py           # 所有超参数（MatchConfig dataclass）
├── utils.py            # 向量工具（cosine_sim、attention_weighted_value）
├── scoring.py          # 评分函数（compute_gate、S_cap、S_need、MapScore）
├── pipeline.py         # 匹配流水线（1-1 排名、1-N 团队构建）
├── encoder.py          # SimpleEncoder（测试用，BoW hash 向量）
├── encoder_sbert.py    # SBERTEncoder（生产用，BAAI/bge-base-en-v1.5）
└── tests/
    ├── generate_fixtures.py          # 生成 JSON 测试数据
    ├── test_matching.py              # 代码正确性测试（39 个）
    ├── test_semantic_correctness.py  # 语义正确性测试（31 个）
    └── fixtures/
        ├── scenario_academic.json    # 学术合作场景（Maya + 4 名候选人）
        └── candidates_pool.json     # 10 名随机候选人池
```

---

## 快速开始

### 安装依赖

```bash
pip install numpy sentence-transformers pytest
```

### 运行测试

```bash
cd mapping-algo
python -m pytest tests/ -v
# 预期：70 passed
```

### 重新生成测试 fixtures（修改场景后执行）

```bash
python tests/generate_fixtures.py
```

---

## 数据结构

### 用户状态 `UserState`

每个用户由**能力集合**和**需求集合**描述（可变长度，不需要固定维度）：

```python
from datatypes import UserState, CapabilityEntry, NeedEntry
import numpy as np

# 每个能力是 (embedding, mu, sigma) 三元组
# mu: 熟练度均值 ∈ [0,1]
# sigma: 不确定度，用于 UCB 探索
user = UserState(
    user_id="alice_001",
    capabilities=[
        CapabilityEntry(
            embedding=encoder("clinical research"),  # 语义向量
            mu=0.85,
            sigma=0.2,
            source="explicit",       # "explicit" | "implicit" | "meta"
            description="clinical research",
        ),
        CapabilityEntry(
            embedding=encoder("biostatistics"),
            mu=0.75,
            sigma=0.15,
            source="explicit",
            description="biostatistics",
        ),
    ],
    needs=[
        NeedEntry(
            embedding=encoder("deep learning collaboration"),
            intensity=0.8,           # 需求强度 ∈ [0,1]
            description="deep learning collaboration",
        ),
    ],
    clearance_level=2,               # 安全等级，用于门控检查
)
```

### 任务 `Task`

```python
from datatypes import Task, TaskRequirement, TaskOffer

task = Task(
    task_id="task_001",
    goal="Need a clinical collaborator for multimodal patient triage paper",
    requirements=[
        # 硬约束：必须满足，否则 gate=0
        TaskRequirement(
            embedding=encoder("clinical data analysis"),
            level=0.8,               # 最低要求熟练度
            constraint_type="hard",
            description="clinical data analysis",
        ),
        # 软约束：越多越好，用于计算 S_cap
        TaskRequirement(
            embedding=encoder("paper writing"),
            level=0.6,
            constraint_type="soft",
            description="paper writing",
        ),
    ],
    offers=[
        TaskOffer(
            embedding=encoder("NeurIPS co-authorship"),
            strength=0.9,
            source="explicit",
            description="NeurIPS co-authorship",
        ),
    ],
    data_clearance=1,                # 最低安全等级要求
)
```

---

## 核心函数

### 评分组件（`scoring.py`）

```python
from scoring import compute_gate, compute_s_cap, compute_s_need, compute_match_score
from config import MatchConfig

cfg = MatchConfig()
theta = np.array([0.4, -0.4])  # w_c ≈ 0.69, w_n ≈ 0.31

# 1. 门控检查
sigma, reason = compute_gate(u, v, task, cfg)
# sigma=1 通过，sigma=0 失败（reason 说明原因）

# 2. 能力覆盖分 S_cap
s_cap, gap_details = compute_s_cap(v, u, task, cfg)
# gap_details: 每个软约束的覆盖细节

# 3. 需求满足分 S_need
s_need, need_details = compute_s_need(v, task, cfg)

# 4. 综合 MapScore（自动包含上面所有步骤）
result = compute_match_score(u, v, task, theta, cfg, use_ucb=False)
print(result.match_score)   # M(u,v,T)
print(result.to_dict())     # 结构化输出
```

### 1-1 匹配（`pipeline.py`）

对候选池排序，返回 Top-K：

```python
from pipeline import match_one_to_one

results = match_one_to_one(
    u=requester,
    task=task,
    pool=[candidate_1, candidate_2, ...],
    theta=theta,
    cfg=cfg,
    top_k=10,
    use_ucb=True,    # 开启 UCB 探索（适合冷启动阶段）
    round_t=5,       # 当前轮次，影响 UCB 的 β_t
)

for r in results:
    print(f"{r.candidate_id}: score={r.match_score:.4f}")
    print(r.to_dict())
```

**输出示例：**

```json
{
  "candidate_id": "bob_002",
  "match_score": 0.8731,
  "components": {
    "S_cap": 0.9124,
    "S_need": 0.7842,
    "w_c": 0.6900,
    "w_n": 0.3100,
    "sigma": 1
  },
  "gap_coverage_detail": {
    "paper writing": {
      "gap": 0.4200,
      "covered": 0.3800,
      "coverage": "90%"
    }
  },
  "need_satisfaction_detail": {
    "NeurIPS publication": {
      "need": 0.9000,
      "offer_matched": 0.8741,
      "satisfied": "97%"
    }
  }
}
```

### 1-N 团队构建（`pipeline.py`）

贪心子模最大化，选出互补团队：

```python
from pipeline import match_one_to_n

team = match_one_to_n(
    u=requester,
    task=task,
    pool=[c1, c2, c3, ...],
    theta=theta,
    cfg=cfg,          # cfg.n_max 控制最大团队规模
    use_ucb=True,
    round_t=5,
)

print(team.team_members)          # ["bob_002", "carol_003"]
print(team.collective_coverage)   # 0.8732（团队整体覆盖度）
print(team.termination_reason)    # "gap_covered" | "max_size" | "no_positive_gain"
print(team.to_dict())
```

**算法目标函数（MAX 模型）：**

```
R(S) = Σ_j min(max_{v∈S} p̃_v^j, Gap_j) × w_j
       ──────────────────────────────────────────
              Σ_j Gap_j × w_j
```

贪心每步选择边际收益最大的候选人，当所有 Gap 覆盖完毕或达到 `n_max` 时停止。

---

## 配置参数（`MatchConfig`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_dim` | 768 | 向量维度（与 encoder 一致） |
| `temperature` | 0.1 | 注意力 softmax 温度 τ（越小越 sharp） |
| `tau_hard` | 0.7 | 硬约束余弦相似度阈值（BGE 下推荐 0.63~0.70） |
| `alpha_infer` | 0.5 | inferred offer 折扣系数 |
| `ucb_beta_scale` | 2.0 | UCB 探索系数，β_t = √(scale × log(t)) |
| `n_max` | 5 | 1-N 团队最大规模 |
| `gap_epsilon` | 1e-4 | 认为 gap 已覆盖的阈值 |
| `theta_default` | [0.4, -0.4] | 默认权重 → w_c≈0.69, w_n≈0.31 |

```python
cfg = MatchConfig(
    tau_hard=0.63,      # BGE encoder 下使用此值
    temperature=0.05,   # 更 sharp 的注意力
    n_max=3,            # 最多选 3 人的团队
)
```

---

## Encoder 选择

| Encoder | 维度 | 模型大小 | 适用场景 |
|---------|------|---------|---------|
| `SimpleEncoder` | 64 | 极小 | 单元测试（不依赖 GPU/网络） |
| `SBERTEncoder("BAAI/bge-base-en-v1.5")` | 768 | ~440MB | 英文，本地 CPU 可用 |
| `SBERTEncoder("BAAI/bge-m3")` | 1024 | ~2.2GB | 中英双语，生产推荐 |

```python
# 开发/测试
from encoder import SimpleEncoder
enc = SimpleEncoder(dim=64, seed=42)

# 生产（英文）
from encoder_sbert import SBERTEncoder
enc = SBERTEncoder("BAAI/bge-base-en-v1.5")

# 生产（双语）
enc = SBERTEncoder("BAAI/bge-m3")
```

**注意：** `tau_hard` 需要随 encoder 调整：

| Encoder | 推荐 tau_hard |
|---------|--------------|
| SimpleEncoder | 0.3（BoW 相似度低） |
| bge-base-en-v1.5 | 0.63 |
| bge-m3 | 0.65~0.70 |

---

## 注意力机制的关键性质

本算法使用 softmax 注意力进行软匹配：

```
p̃_v^j = Σ_i  softmax(sim(req_j, cap_i) / τ)_i  ×  μ_i
```

**重要限制**：当候选人只有**单一能力**时，softmax([x]) = [1.0]，
此时不论 embedding 相似度如何，注意力权重均为 1。
即"JavaScript 工程师"在"需要临床分析"的任务上，p̃_v 仍等于其 μ。

这在 SimpleEncoder（随机正交向量）下会导致语义失效，
但在 BGE/SBERT 下（相关词 sim≈0.8，无关词 sim≈0.1）
配合温度 τ=0.05~0.1，softmax 会显著峰化，行为正常。

---

## 测试体系

```
tests/
├── test_matching.py              # 39 个代码正确性测试
│   ├── TestVectorUtils           # cosine_sim、softmax、attention（8个）
│   ├── TestGate                  # clearance、hard constraint（4个）
│   ├── TestScap / TestSneed      # 值域、细节结构（各4个）
│   ├── TestMapScore              # 公式、权重、to_dict（5个）
│   ├── TestUCB                   # UCB 提升、sigma 敏感性（2个）
│   ├── TestOneToOne              # Top-K、空池、排名（5个）
│   └── TestOneToN                # 团队构建、子模性、覆盖度（7个）
│
└── test_semantic_correctness.py  # 31 个语义正确性测试
    ├── Scenario A  Gap coverage dominance
    ├── Scenario B  S_need dominance（相同 S_cap，需求匹配决胜）
    ├── Scenario C  Gate 绝对性（被过滤者永不出现）
    ├── Scenario D  UCB 探索（不确定性候选人被拉升）
    ├── Scenario E  团队子模性（互补优于重复）
    ├── Scenario F  权重 θ 敏感性
    ├── Scenario G  硬约束边界（mu=0.70 pass，mu=0.69 fail）
    ├── Scenario H  1-N 终止条件
    └── Scenario I  u 自身能力减少 gap
```
