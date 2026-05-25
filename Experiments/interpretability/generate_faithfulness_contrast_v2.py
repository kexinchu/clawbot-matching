"""Generate `simulator/20_Tasks_FaithfulnessContrastV2.json` — a richer
contrast testset than V1.

Three things V1 did poorly that V2 fixes:

  1. **Winner monoculture.** V1 always gave `high_cap_low_need` to slot
     ``candidate_01`` and that archetype always won. V2 rotates the
     intended winner archetype across tasks and randomly permutes which
     candidate slot gets which archetype, so winners are not
     concentrated on one ``user_id`` or one archetype.
  2. **No archetype metadata.** V1 stored archetype only inside
     ``tier_meta``; V2 also writes it into ``task_entry["metadata"]`` so
     ``run_faithfulness.py`` can tally winner-archetype distributions.
  3. **Single task profile.** V1 used the same task params for every
     task. V2 uses four "task profiles" deliberately designed to favour
     a different archetype each — large gaps on critical reqs favour
     ``high_cap_low_need``, uniform gaps favour ``balanced``, one
     dominant requirement favours ``specialist``, tiny gaps + matched
     needs favour ``low_cap_high_need``.

The actual winner is *not* enforced — it's whatever the unmodified
``compute_match_score`` produces under fixed θ. The "intended" archetype
is a design hint we record so we can later check whether our task
profiles really do bias the score the way we expected.

JSON schema is identical to V1 / the tiered testset so the
``run_faithfulness.py`` / ``run_interpretability.py`` builders consume
it unchanged.
"""

from __future__ import annotations

import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
SOURCE_PATH = _REPO / "simulator" / "20_Tasks_Testset_tiered.json"
OUT_PATH = _REPO / "simulator" / "20_Tasks_FaithfulnessContrastV2.json"


# Off-topic capability names — used by the distractor archetype to fill
# slots without accidentally covering any required skill.
_OFF_TOPIC_CAPS = [
    "graphic_design",
    "video_editing",
    "social_media",
    "copywriting",
    "translation",
    "event_planning",
]

# Five archetypes. The semantics live in `_archetype_caps_and_needs`.
_ARCHETYPES = [
    "high_cap_low_need",
    "low_cap_high_need",
    "balanced",
    "specialist",
    "distractor",
]

# Per-task assignment of intended winners. We weight the rotation toward
# the three S_cap-discriminative profiles (high_cap / balanced /
# specialist) and only include two low_cap-winning tasks. low_cap can
# only win when S_cap saturates for everyone (the formula gives them
# their best S_need but no S_cap edge), so each low_cap-winning task
# necessarily has frac_candidates_scap_eq_1 ≈ 1.0 — including too many
# would push the testset-level saturation back up. 2/20 keeps it low
# while still exercising the S_need pathway end-to-end.
_INTENDED_ROTATION: List[str] = (
    ["high_cap_low_need"] * 6
    + ["balanced"] * 6
    + ["specialist"] * 6
    + ["low_cap_high_need"] * 2
)
assert len(_INTENDED_ROTATION) == 20


# ---------------------------------------------------------------------------
# Task-profile rewrites
# ---------------------------------------------------------------------------

def _apply_task_profile(
    task_required: Dict[str, float],
    profile: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (new_required, new_proposer_caps) for a given task profile.

    Each profile shapes the proposer-induced gaps and the requirement
    weights so a *different* archetype tends to win under the fixed
    EVAL θ.

    The four profiles:
      * ``high_cap_low_need`` — large gaps on the two most-demanded
        skills; the high-cap winner benefits from strong coverage.
      * ``balanced``         — uniform medium gaps on every required
        skill; an even-cap candidate beats a peaky one because the
        weighted average favours breadth.
      * ``specialist``       — one requirement is pushed to level=1.0
        and the proposer is empty there, while all other reqs are mostly
        covered; the specialist archetype's single strong cap dominates.
      * ``low_cap_high_need``— small gaps everywhere (proposer is
        already strong); S_cap saturates for almost all candidates so
        S_need (and therefore matched needs) becomes the tiebreaker.
    """
    ordered = sorted(task_required.items(), key=lambda kv: -kv[1])
    c1, c2 = ordered[0][0], ordered[1][0] if len(ordered) > 1 else ordered[0][0]
    other = [s for s, _ in ordered[2:]]

    if profile == "high_cap_low_need":
        # Large gaps on c1, c2 (proposer is near-empty there); the
        # high-cap winner benefits from strong coverage on those.
        required = {
            s: float(min(1.0, lvl + 0.15)) for s, lvl in task_required.items()
        }
        proposer = {c1: 0.10, c2: 0.10}
        for s in other:
            proposer[s] = 0.45
    elif profile == "balanced":
        # Uniform medium levels and gaps deliberately tuned so NO
        # archetype's caps fully cover them: required=0.80, proposer=
        # 0.15 → gap=0.65 (above high_cap c2=0.75? no, 0.65<0.75). The
        # gap is below high_cap's strong caps but above balanced's
        # medium caps — so high_cap covers fully on c1,c2 but partial
        # on others; balanced is partial everywhere. The
        # winner-decider becomes "average coverage + matched-need".
        required = {s: 0.80 for s in task_required}
        proposer = {s: 0.15 for s in task_required}
    elif profile == "specialist":
        # c1 pushed to 1.0 and proposer is empty there (gap=1.0), so
        # specialist's c1=0.95 dominates without saturating S_cap;
        # all other reqs are mostly covered by the proposer so they
        # become tiebreakers, not differentiators.
        required = {s: float(lvl) for s, lvl in task_required.items()}
        required[c1] = 1.0
        proposer = {c1: 0.0}
        for s in [c2] + other:
            proposer[s] = 0.80
    elif profile == "low_cap_high_need":
        # Tiny gaps everywhere → S_cap saturates for the strong-cap
        # archetypes → S_need decides → low_cap (matched needs) wins.
        # This profile is deliberately saturated; we keep it to ≤ 2
        # tasks in the rotation so the testset-level
        # frac_candidates_scap_eq_1 stays low.
        required = {
            s: float(min(1.0, lvl + 0.05)) for s, lvl in task_required.items()
        }
        proposer = {s: float(min(0.85, lvl + 0.05)) for s, lvl in required.items()}
    else:
        raise ValueError(f"Unknown task profile: {profile!r}")

    # Soft skills that stay on the proposer regardless of profile.
    proposer.setdefault("communication", 0.55)
    proposer.setdefault("project_management", 0.65)
    return required, proposer


# ---------------------------------------------------------------------------
# Archetype caps / needs
# ---------------------------------------------------------------------------

def _archetype_caps_and_needs(
    archetype: str,
    required: Dict[str, float],
    offers: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Capabilities + needs for one archetype, given the (rewritten)
    required-skills dict and the (already-overridden) offer-strength
    dict.

    Needs are deliberately **on-topic** (skill names that appear in
    ``required`` / ``offer_skills``) so the per-need ``linked_offer`` is
    a real, perturbable target. To keep ``S_need`` below 1.0 we make at
    least one need's intensity exceed the matching offer's strength —
    typically by targeting the *weakest offered skill* (computed from
    ``offers``, not ``required``, because ``_build_v2_task`` may have
    dampened individual offers — picking by required-level would aim
    the need at the wrong skill).
    """
    ordered = sorted(required.items(), key=lambda kv: -kv[1])
    c1 = ordered[0][0]
    c2 = ordered[1][0] if len(ordered) > 1 else c1
    other = [s for s, _ in ordered[2:]] or [c1]
    # Weakest-offer skill (not weakest-required). For uniform offers
    # this still falls back to a deterministic choice.
    weakest = min(offers.items(), key=lambda kv: kv[1])[0] if offers else c1

    if archetype == "high_cap_low_need":
        caps = {c1: 0.85, c2: 0.75}
        for i, s in enumerate(other[:3]):
            caps[s] = 0.55 + 0.05 * i
        needs = {weakest: 0.95}
        return caps, needs

    if archetype == "low_cap_high_need":
        caps = {c1: 0.25, c2: 0.20}
        for i, s in enumerate(other[:3]):
            caps[s] = 0.35 + 0.05 * i
        # Match needs to the strong critical offers — these candidates
        # benefit most from S_need when S_cap saturates for everyone.
        needs = {c1: 0.85, c2: 0.80}
        return caps, needs

    if archetype == "balanced":
        caps = {c1: 0.55, c2: 0.55}
        for i, s in enumerate(other[:3]):
            caps[s] = 0.55 + 0.03 * i
        needs = {c1: 0.50, weakest: 0.90}
        return caps, needs

    if archetype == "specialist":
        caps = {c1: 0.95, c2: 0.20}
        for i, s in enumerate(other[:2]):
            caps[s] = 0.40 + 0.05 * i
        needs = {c2: 0.70, weakest: 0.85}
        return caps, needs

    if archetype == "distractor":
        caps: Dict[str, float] = {}
        for i, s in enumerate(_OFF_TOPIC_CAPS[:3]):
            caps[s] = 0.85 - 0.05 * i
        # Tiny presence on c1 so they're not gate-failed.
        caps[c1] = 0.30
        needs = {weakest: 0.90}
        return caps, needs

    raise ValueError(f"Unknown archetype: {archetype!r}")


# ---------------------------------------------------------------------------
# Candidate dict builder
# ---------------------------------------------------------------------------

def _priors_from_caps(caps: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {
        skill: {"mu_init": float(mu), "sigma_init": 0.10}
        for skill, mu in caps.items()
    }


def _make_candidate(
    task_id: str,
    slot_idx: int,
    archetype: str,
    caps: Dict[str, float],
    needs: Dict[str, float],
) -> Dict[str, Any]:
    cid = f"{task_id}_candidate_{slot_idx:02d}"
    return {
        "candidate_profile": {
            "user_id": cid,
            "role": archetype,
            "capabilities": dict(caps),
            "needs": dict(needs),
            "preferences": {
                "availability": "high",
                "timezone": "UTC",
                "current_load": 0.4,
            },
            "constraints": {"workload": 0.4},
            "history_summary": (
                f"Synthetic {archetype} candidate for {task_id}."
            ),
        },
        "candidate_card": {
            "candidate_id": cid,
            "summary": f"{archetype} candidate for task {task_id}.",
            "highlighted_strengths": sorted(caps, key=caps.get, reverse=True)[:3],
            "highlighted_risks": [],
            "explanation": (
                f"Synthetic faithfulness-contrast-V2 candidate ({archetype})."
            ),
        },
        "requester_personas": [],
        "candidate_personas": [],
        "context_latents": {},
        "capability_priors": _priors_from_caps(caps),
        "tier": "faithfulness_contrast_v2",
        "tier_meta": {"archetype": archetype},
    }


# ---------------------------------------------------------------------------
# Per-task builder
# ---------------------------------------------------------------------------

def _build_v2_task(
    base_entry: Dict[str, Any],
    task_idx: int,
    rng: random.Random,
) -> Dict[str, Any]:
    out = deepcopy(base_entry)
    task = out["task"]
    task_id = task["task_id"]

    intended = _INTENDED_ROTATION[task_idx % len(_INTENDED_ROTATION)]
    # The rotation has exactly 20 entries so indexing matches the
    # tiered source 1:1; the modulo is defensive in case the source
    # ever ships a different N.
    new_required, new_proposer_caps = _apply_task_profile(
        dict(task.get("required_skills", {})), intended,
    )
    task["required_skills"] = new_required
    # Default: offers mirror the (rewritten) required levels. For the
    # `balanced` profile we override this to dampen the offers on every
    # skill *except* c1 — that weakens high_cap_low_need's
    # single-high-intensity matched need (linked to `weakest`) while
    # leaving balanced's spread of matched-light needs roughly
    # untouched. Without this dampening, high_cap consistently wins the
    # balanced profile because its higher S_cap is not offset by enough
    # S_need disadvantage.
    if intended == "balanced":
        ordered_keys = [
            s for s, _ in sorted(new_required.items(), key=lambda kv: -kv[1])
        ]
        c1 = ordered_keys[0]
        task["offer_skills"] = {
            s: (0.95 if s == c1 else 0.40) for s in new_required
        }
    else:
        task["offer_skills"] = {s: float(l) for s, l in new_required.items()}
    task["title"] = task.get("title", "") + " [v2-contrast]"
    task["description"] = (
        task.get("description", "")
        + f" (task profile favours archetype: {intended}.)"
    )

    out["proposer_profile"] = {
        **deepcopy(out["proposer_profile"]),
        "capabilities": new_proposer_caps,
    }

    # Random archetype-to-slot permutation per task — guarantees that
    # winners are not stuck on candidate_01.
    perm = list(_ARCHETYPES)
    rng.shuffle(perm)
    cands: List[Dict[str, Any]] = []
    candidate_archetypes: Dict[str, str] = {}
    offer_dict = task["offer_skills"]
    for slot_idx, archetype in enumerate(perm, start=1):
        caps, needs = _archetype_caps_and_needs(
            archetype, new_required, offer_dict,
        )
        cand = _make_candidate(task_id, slot_idx, archetype, caps, needs)
        cands.append(cand)
        candidate_archetypes[cand["candidate_profile"]["user_id"]] = archetype
    out["candidates"] = cands

    out["metadata"] = {
        **(out.get("metadata") or {}),
        "intended_winner_archetype": intended,
        "candidate_archetypes": candidate_archetypes,
        "task_profile": intended,
        "perm_seed_index": task_idx,
    }
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate(seed: int = 42) -> Dict[str, Any]:
    source = json.loads(SOURCE_PATH.read_text())
    base_tasks = source.get("tasks", [])
    rng = random.Random(seed)
    contrast_tasks = [
        _build_v2_task(t, idx, rng) for idx, t in enumerate(base_tasks)
    ]
    return {
        "metadata": {
            "generated_by": str(
                Path("Experiments/interpretability")
                / "generate_faithfulness_contrast_v2.py"
            ),
            "source": str(Path("simulator") / SOURCE_PATH.name),
            "goal": (
                "non-saturated S_cap + heterogeneous winner archetype + "
                "randomised candidate-slot assignment"
            ),
            "num_tasks": len(contrast_tasks),
            "candidate_archetypes": _ARCHETYPES,
            "intended_winner_rotation": _INTENDED_ROTATION,
            "off_topic_caps": _OFF_TOPIC_CAPS,
            "random_seed": seed,
            "design_note": (
                "Each task chooses one of four 'task profiles' designed "
                "to favour a particular archetype. Candidates are assigned "
                "to slots via a random permutation per task so winners are "
                "spread across user_ids."
            ),
        },
        "tasks": contrast_tasks,
    }


def main() -> int:
    payload = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote simulator/{OUT_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
