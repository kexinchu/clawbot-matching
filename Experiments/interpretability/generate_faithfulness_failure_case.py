"""Generate `simulator/20_Tasks_FaithfulnessFailureCase.json`.

This is a **diagnostic** testset, not a "good results" testset. It
deliberately reproduces the canonical failure mode we discovered while
designing the V1 contrast testset:

  * Each candidate has exactly **one off-topic, high-intensity need**
    (skill names like ``mentorship`` / ``networking`` / ``funding``
    that do not appear anywhere in ``offer_skills``).
  * The need is the only one, so its share of S_need is 100 %.
  * The decomposition's ``contribution_to_M`` for this need equals
    ``w_n × 1.0 ≈ 0.38`` — almost always larger than any single
    requirement's contribution.
  * The explanation therefore picks this need as the *top* factor.
  * But because the need is off-topic, the attention layer
    ``softmax(sim / τ)`` is near-uniform across all offers, so the
    "linked_offer" we report is whatever happens to have the
    marginally highest cosine — *not* a semantically meaningful match.
  * Removing that linked offer barely moves ``o_tilde`` (and can even
    increase it when the removed offer had below-average strength),
    so the top-factor *score drop* is small or negative.

The expected result is therefore a **failure of the deletion test**:

  * mean_top_factor_drop ≤ mean_random_factor_drop;
  * frac_top_beats_random < 0.5;
  * mean_true_top_drop_percentile far below 1.0.

That failure is the whole point — it documents the limit of
share-based attribution under attention soft-matching with weak
semantic links. Once we know the failure pattern, downstream
explanation pipelines can be designed to either avoid it (require
``linked_offer`` similarity > threshold) or surface it (mark the
attribution as "non-actionable").
"""

from __future__ import annotations

import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
SOURCE_PATH = _REPO / "simulator" / "20_Tasks_Testset_tiered.json"
OUT_PATH = _REPO / "simulator" / "20_Tasks_FaithfulnessFailureCase.json"


# Skills guaranteed NOT to appear in the testset offer set. Picking
# words that exist in *some* candidate `_OFF_TOPIC_CAPS` set is fine —
# the attention spuriousness is at the offer / need level, not the
# capability level.
_OFF_TOPIC_NEEDS = [
    "mentorship",
    "networking",
    "funding",
    "career_growth",
    "executive_coaching",
    "press_relations",
    "patenting",
]


# Five candidate templates: keep the capability profiles roughly similar
# to contrast V2 so this testset focuses *purely* on the off-topic-need
# pathology rather than mixing it with other faithfulness questions.
_CANDIDATE_TEMPLATES: List[Tuple[str, Dict[str, Any]]] = [
    ("strong_general", {
        "cap_mu_c1": 0.85,
        "cap_mu_c2": 0.75,
        "cap_mu_other": 0.60,
    }),
    ("strong_specialist", {
        "cap_mu_c1": 0.95,
        "cap_mu_c2": 0.20,
        "cap_mu_other": 0.45,
    }),
    ("medium", {
        "cap_mu_c1": 0.55,
        "cap_mu_c2": 0.55,
        "cap_mu_other": 0.55,
    }),
    ("weak", {
        "cap_mu_c1": 0.25,
        "cap_mu_c2": 0.20,
        "cap_mu_other": 0.40,
    }),
    ("distractor", {
        "cap_mu_c1": 0.30,
        "cap_mu_c2": 0.10,
        "cap_mu_other": 0.10,
    }),
]


def _build_caps(
    template: Dict[str, Any],
    critical: List[str],
    other: List[str],
) -> Dict[str, float]:
    c1 = critical[0]
    c2 = critical[1] if len(critical) > 1 else c1
    caps: Dict[str, float] = {
        c1: float(template["cap_mu_c1"]),
        c2: float(template["cap_mu_c2"]),
    }
    for s in other[:3]:
        caps[s] = float(template["cap_mu_other"])
    return caps


def _priors_from_caps(caps: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {s: {"mu_init": float(mu), "sigma_init": 0.10}
            for s, mu in caps.items()}


def _make_candidate(
    task_id: str,
    slot_idx: int,
    template_name: str,
    caps: Dict[str, float],
    off_topic_need: str,
    intensity: float,
) -> Dict[str, Any]:
    cid = f"{task_id}_candidate_{slot_idx:02d}"
    needs = {off_topic_need: float(intensity)}
    return {
        "candidate_profile": {
            "user_id": cid,
            "role": template_name,
            "capabilities": dict(caps),
            "needs": needs,
            "preferences": {"availability": "high", "timezone": "UTC",
                            "current_load": 0.4},
            "constraints": {"workload": 0.4},
            "history_summary": (
                f"Synthetic {template_name} for {task_id}. "
                f"Has a single off-topic need: {off_topic_need}."
            ),
        },
        "candidate_card": {
            "candidate_id": cid,
            "summary": f"{template_name} candidate for {task_id}.",
            "highlighted_strengths":
                sorted(caps, key=caps.get, reverse=True)[:3],
            "highlighted_risks": [],
            "explanation": (
                "Synthetic failure-case candidate. "
                "Single off-topic high-intensity need is intentional."
            ),
        },
        "requester_personas": [],
        "candidate_personas": [],
        "context_latents": {},
        "capability_priors": _priors_from_caps(caps),
        "tier": "faithfulness_failure_case",
        "tier_meta": {
            "template": template_name,
            "off_topic_need": off_topic_need,
            "off_topic_intensity": float(intensity),
        },
    }


def _build_failure_task(
    base_entry: Dict[str, Any],
    task_idx: int,
    rng: random.Random,
) -> Dict[str, Any]:
    out = deepcopy(base_entry)
    task = out["task"]
    task_id = task["task_id"]

    base_required = dict(task.get("required_skills", {}))
    if not base_required:
        return out

    # Bump levels a bit so gaps are non-trivial — we want S_cap to vary,
    # just not be the source of the deletion-test failure.
    new_required = {
        s: float(min(1.0, lvl + 0.15)) for s, lvl in base_required.items()
    }
    task["required_skills"] = new_required
    task["offer_skills"] = {s: float(lvl) for s, lvl in new_required.items()}
    task["title"] = task.get("title", "") + " [failure-case]"
    task["description"] = (
        task.get("description", "")
        + " (single off-topic need per candidate — diagnostic dataset)."
    )

    # Proposer is weak on the top-2 reqs (so candidates with strong caps
    # there get high S_cap and become the winner).
    ordered = sorted(new_required.items(), key=lambda kv: -kv[1])
    critical = [ordered[0][0], ordered[1][0]]
    other = [s for s, _ in ordered[2:]]
    proposer_caps = {critical[0]: 0.10, critical[1]: 0.10}
    for s in other:
        proposer_caps[s] = 0.45
    proposer_caps.setdefault("communication", 0.55)
    proposer_caps.setdefault("project_management", 0.65)
    out["proposer_profile"] = {
        **deepcopy(out["proposer_profile"]),
        "capabilities": proposer_caps,
    }

    perm = list(_CANDIDATE_TEMPLATES)
    rng.shuffle(perm)
    cands: List[Dict[str, Any]] = []
    template_map: Dict[str, str] = {}
    off_topic_need_map: Dict[str, str] = {}
    for slot_idx, (template_name, template_spec) in enumerate(perm, start=1):
        caps = _build_caps(template_spec, critical, other)
        # Each candidate gets a *different* off-topic need so the
        # explanation's top-factor-per-candidate links to different
        # offers — keeps the failure pattern from being identical for
        # all candidates.
        off_topic = _OFF_TOPIC_NEEDS[
            (task_idx * 5 + slot_idx) % len(_OFF_TOPIC_NEEDS)
        ]
        # Use LOW intensity (0.4) — when the need's intensity is far
        # below the average offer strength, satisfied=intensity for
        # every offer choice, so the perturbation can't reduce
        # `o_tilde` enough to move satisfied. This is the original V1
        # failure mode where contribution_to_S_need=1.0 (single need)
        # but perturbation_drop≈0 (saturated satisfaction).
        cand = _make_candidate(
            task_id, slot_idx, template_name, caps, off_topic, intensity=0.40,
        )
        cands.append(cand)
        cid = cand["candidate_profile"]["user_id"]
        template_map[cid] = template_name
        off_topic_need_map[cid] = off_topic
    out["candidates"] = cands

    out["metadata"] = {
        **(out.get("metadata") or {}),
        "candidate_templates": template_map,
        "candidate_off_topic_needs": off_topic_need_map,
        "failure_mode": "single_off_topic_need_with_high_intensity",
    }
    return out


def generate(num_tasks: int = 10, seed: int = 42) -> Dict[str, Any]:
    source = json.loads(SOURCE_PATH.read_text())
    base_tasks = source.get("tasks", [])[:num_tasks]
    rng = random.Random(seed)
    failure_tasks = [
        _build_failure_task(t, idx, rng) for idx, t in enumerate(base_tasks)
    ]
    return {
        "metadata": {
            "generated_by": str(
                Path("Experiments/interpretability")
                / "generate_faithfulness_failure_case.py"
            ),
            "source": str(Path("simulator") / SOURCE_PATH.name),
            "goal": (
                "diagnostic dataset: every candidate has a single off-topic "
                "need with high intensity, designed to trigger the "
                "attribution-share ≠ perturbation-sensitivity failure"
            ),
            "num_tasks": len(failure_tasks),
            "candidate_templates": [t[0] for t in _CANDIDATE_TEMPLATES],
            "off_topic_needs": _OFF_TOPIC_NEEDS,
            "expected_failure_pattern": [
                "mean_top_factor_drop <= mean_random_factor_drop",
                "frac_top_beats_random significantly below 1.0",
                "mean_true_top_drop_percentile far below 1.0",
                "oracle_top differs from explanation_top",
                "high explanation_rank_among_all_perturbations",
            ],
        },
        "tasks": failure_tasks,
    }


def main() -> int:
    payload = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote simulator/{OUT_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
