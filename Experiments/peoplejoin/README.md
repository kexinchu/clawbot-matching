# PeopleJoin-Reactive Baseline (CoWeaver adaptation)

Adapted from **PeopleJoin** (Jhamtani et al., ACL Findings 2025) into CoWeaver’s
collaborator-matching setting as method name `peoplejoin_reactive`.

This baseline implements an LLM-mediated **search → ask → decide** discovery loop.
It does **not** use MapScore, S_cap, S_need, UCB, Dreaming scores, or oracle
simulator latents for selection.

---

## Mapping from original PeopleJoin → CoWeaver

| PeopleJoin (original) | CoWeaver adaptation |
|---|---|
| Enterprise people directory + retrieval | BM25 over public **candidate cards** (role, summary, strengths, risks) |
| Message a person to gather missing info | `ask_candidate` answered by a **profile-conditioned responder** |
| ReAct controller decides next action | `PeopleJoinReactiveController` with strict JSON actions |
| Task completion via multi-person info seeking | Single **top-1 collaborator** selection for a requester task |
| Organizational messaging substrate | Simulated asks with shared LLM backend (or deterministic mock) |

### What we keep

- ReAct-style controller with explicit tool/actions.
- Retrieval over a **public directory**, not full private dossiers up front.
- Active information gathering via targeted questions before committing.
- Explicit interaction / API cost accounting.

### What we change for CoWeaver

- Target is **collaborator matching** (one candidate), not multi-hop organizational QA.
- Directory docs are CoWeaver `candidate_card` rows, not email/org graphs.
- Candidate answers are grounded in CoWeaver **structured public profiles**
  (capabilities, needs, availability, workload, timezone, style, constraints, history).
- Final evaluation uses CoWeaver’s bilateral + outcome simulators **post-hoc only**.
- Fair protocol uses the v3 20-task × 20-candidate benchmark and budgets below.

---

## Information boundaries

```
Public to controller at start
  ├── requester public profile
  ├── task (skills, offers, metadata)
  └── candidate directory cards (via BM25 search)

Public after ask_candidate
  └── natural-language answers derived from that candidate’s structured profile

Never visible to PeopleJoin (selection time)
  ├── context_latents / latent_* preferences / affinity / risk / opportunity bias
  ├── bilateral decision internals
  ├── simulator outcome / reward / oracle labels
  └── MapScore / S_cap / S_need (post-hoc diagnostics only)
```

Guardrail: `assert_peoplejoin_public_inputs(...)` rejects serialized prompts /
observations containing forbidden keys.

---

## Actions and budgets (default v3 protocol)

Actions (strict JSON):

```json
{"action":"search_relevant_people","query":"..."}
{"action":"ask_candidate","candidate_id":"...","question":"..."}
{"action":"finish","selected_candidate_id":"...","rationale":"..."}
```

Default budgets per task:

| Budget | Default |
|---|---|
| max BM25 searches | 3 |
| max `ask_candidate` | 6 |
| max controller actions | 10 |
| BM25 top-k | 5 |
| seeds (formal) | 0,1,2 |

Notes:

- BM25 search does **not** count as an LLM API call.
- Controller + responder LLM calls **do** count toward API / token / wall-clock cost.
- Malformed JSON gets **one** safe retry; still failing → deterministic fallback + error log.
- Invalid / out-of-pool `finish` is rejected; a deterministic in-pool fallback is used if needed.

---

## Why this is “LLM-mediated collaborator discovery”

Unlike static scorers, PeopleJoin must **discover** who to consider, **query** them
for profile-grounded details, and only then commit. The extra information is
earned through messages and is therefore reported as interaction cost — not as
free access to hidden oracle state.

---

## Contrast with other paradigms

| Method | Focus |
|---|---|
| **PeopleJoin-Reactive** | Active search + information gathering before matching |
| **AgenticPay** (external) | Requester–candidate **bilateral negotiation / payment** dynamics |
| **CoWeaver (MapScore + online learning)** | Analytical matching, uncertainty (UCB), optional targeted Dreaming rerank |

PeopleJoin is therefore an external **discovery** baseline, not a substitute for
CoWeaver’s scoring formula or AgenticPay’s negotiation protocol.

---

## Modules

```
Experiments/peoplejoin/
  bm25_retriever.py          # public-card BM25 index
  candidate_responder.py     # profile-grounded ask answers
  reactive_controller.py     # ReAct controller + budgets + trace
  llm_client.py              # mock / OpenAI-compatible client
  isolation.py               # public-input assertions
  README.md                  # this file
```

Runner:

```
Online_learning/tests/run_v3_peoplejoin_baseline.py
```

Compares `lappas_coverage`, `mapscore_greedy`, `peoplejoin_reactive` on
`simulator/20_Tasks_Testset_v3_tiered.json`.

---

## How to run

### Unit tests + mock smoke

```bash
cd /path/to/CoWeaver
pytest Online_learning/tests/test_peoplejoin_reactive.py -q

# 2-task mock smoke via runner
python Online_learning/tests/run_v3_peoplejoin_baseline.py --mode mock --max-tasks 2 --seeds 0
```

Mock mode is for **code validation only**. Do not cite mock numbers as formal
baseline results.

### Formal API experiment (only with a working key)

```bash
export OPENAI_API_KEY=...
python Online_learning/tests/run_v3_peoplejoin_baseline.py \
  --mode api \
  --seeds 0,1,2 \
  --model openai/gpt-4o-mini \
  --base-url https://openrouter.ai/api/v1
```

Outputs:

- JSON log under `logs/v3_peoplejoin_*`
- Markdown summary under `Experiments/v3_peoplejoin_*`
- Per-task full interaction traces inside the JSON

The runner refuses `--mode api` without `OPENAI_API_KEY` and never fabricates
formal results.
