"""Generate `simulator/20_Tasks_FaithfulnessContrast.json`.

This testset is purpose-built for the faithfulness experiment. The
original tiered testset is great for the weight-learning experiment but
poor for explanation tests because:

  * Most candidates already cover the requester's residual gaps, so
    `S_cap` saturates at 1.0 for everyone and the entire winner/loser
    distinction collapses into `S_need`.
  * Score gaps are tiny (≈0.01), so even a 1-step counterfactual is
    forced to fight saturated logits.

The contrast testset fixes both:

  1. **Proposer has real gaps.** We delete or weaken the proposer's
     capability in the top-2 most-demanded required skills, and bump
     those requirement levels up so the resulting `Gap_j` is large.
  2. **Candidates are deliberately heterogeneous.** Five archetypes:
       - `high_cap_low_need` — strong capabilities, no needs
       - `low_cap_high_need` — weak capabilities, strong matched needs
       - `balanced` — medium across the board, mixed needs
       - `specialist` — strong in one critical skill, weak in the rest
       - `distractor` — mostly off-topic skills, one minor matched skill
  3. **Offers don't trivially satisfy every need.** Each archetype has
     at least one need pointing at a skill *not* in `offer_skills`, so
     `S_need` actually varies between candidates.
  4. **Capability priors are written to match each candidate's true μ,**
     so `build_learning_candidate` reproduces the intended profile
     without a learning-phase reshuffle.

We preserve the JSON shape of `20_Tasks_Testset_tiered.json` so
`run_interpretability`'s builders consume it unchanged.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
SOURCE_PATH = _REPO / "simulator" / "20_Tasks_Testset_tiered.json"
OUT_PATH = _REPO / "simulator" / "20_Tasks_FaithfulnessContrast.json"


# Skills that exist in the testset universe but are not normally task-
# required. Used by the distractor archetype to fill capability slots
# without accidentally covering the gap.
_OFF_TOPIC_CAPS = [
    "graphic_design",
    "video_editing",
    "social_media",
    "copywriting",
    "translation",
    "event_planning",
]

# Note: we deliberately do NOT use off-topic *needs* (e.g. "mentorship",
# "career_coaching") here. The SimpleEncoder collapses them into a
# near-uniform attention pattern over offers, which makes the
# `linked_offer` field meaningless and breaks the deletion test
# (perturbing any single offer barely moves o_tilde). Instead we use
# *on-topic* needs whose intensity exceeds the matching offer's strength
# — that drives S_need < 1.0 with a clean, perturbable link.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bumped_required(required: Dict[str, float]) -> Dict[str, float]:
    """Bump each requirement level up so the resulting gaps are visible."""
    return {
        skill: float(min(1.0, float(level) + 0.10))
        for skill, level in required.items()
    }


def _split_required(
    required: Dict[str, float],
) -> Tuple[List[str], List[str]]:
    """Return (critical, non_critical) skill lists. Critical = top 2 by level."""
    ordered = sorted(required.items(), key=lambda kv: -kv[1])
    critical = [s for s, _ in ordered[:2]]
    non_critical = [s for s, _ in ordered[2:]]
    return critical, non_critical


def _rewrite_proposer(
    base_proposer: Dict[str, Any],
    critical: List[str],
    non_critical: List[str],
) -> Dict[str, Any]:
    """Reset proposer capabilities so the critical skills are barely
    covered. We keep a few unrelated 'soft' skills from the original
    proposer so the proposer feels like a real person instead of an
    empty user.
    """
    out = deepcopy(base_proposer)
    new_caps: Dict[str, float] = {}
    for s in critical:
        new_caps[s] = 0.10
    for s in non_critical:
        new_caps[s] = 0.40
    # keep a couple of unrelated soft skills from the original profile
    preserved_keys = ("communication", "project_management")
    orig_caps = base_proposer.get("capabilities", {})
    for k in preserved_keys:
        if k in orig_caps and k not in new_caps:
            new_caps[k] = float(orig_caps[k])
    out["capabilities"] = new_caps
    # leave proposer.needs alone — Layer 2 doesn't read them for u-side scoring,
    # but they keep the profile schema intact.
    return out


def _priors_from_caps(caps: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """capability_priors mirrors each cap so build_learning_candidate
    reproduces the intended μ. σ_init is deliberately small because the
    faithfulness experiment is about scoring, not learning.
    """
    return {
        skill: {"mu_init": float(mu), "sigma_init": 0.10}
        for skill, mu in caps.items()
    }


def _make_candidate(
    task_id: str,
    idx: int,
    caps: Dict[str, float],
    needs: Dict[str, float],
    archetype: str,
) -> Dict[str, Any]:
    """Build a candidate dict that matches `run_interpretability`'s
    expected schema (candidate_profile + capability_priors + tier + tier_meta).
    The persona-level fields are stubbed out — the faithfulness
    experiment never reads them.
    """
    cid = f"{task_id}_candidate_{idx:02d}"
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
            "summary": (
                f"{archetype} candidate for task {task_id}."
            ),
            "highlighted_strengths": sorted(caps, key=caps.get, reverse=True)[:3],
            "highlighted_risks": [],
            "explanation": (
                f"Synthetic faithfulness-contrast candidate ({archetype})."
            ),
        },
        "requester_personas": [],
        "candidate_personas": [],
        "context_latents": {},
        "capability_priors": _priors_from_caps(caps),
        "tier": "faithfulness_contrast",
        "tier_meta": {"archetype": archetype},
    }


# ---------------------------------------------------------------------------
# Candidate archetypes
# ---------------------------------------------------------------------------

def _archetype_caps_and_needs(
    archetype: str,
    critical: List[str],
    non_critical: List[str],
    required: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (capabilities, needs) for one candidate archetype.

    Designs are deliberately polarised:

      * high_cap_low_need: covers critical reqs strongly (but not
        saturated). One on-topic high-intensity need targets the
        *weakest* offer so S_need stays below 1.0 with a clean,
        perturbable link.
      * low_cap_high_need: weak on critical reqs so S_cap stays well
        below 1.0; needs match the strong offers so S_need is high.
      * balanced: passable across the board; one matched + one
        high-intensity-low-offer need so S_need lands in the middle.
      * specialist: dominates the most-demanded critical skill but is
        weak on the second one, exercising S_cap's per-requirement
        averaging. Needs target python + the weakest offer.
      * distractor: invests in off-topic capabilities; minimum overlap
        with the task; one need targets the weakest offer.

    The deletion test specifically requires (need, linked_offer) pairs
    where attention is *concentrated* on the linked offer — that's why
    every need here is an on-topic skill name, not an off-topic one.
    """
    c1 = critical[0] if critical else (non_critical[0] if non_critical else "skill_a")
    c2 = critical[1] if len(critical) > 1 else c1
    other = non_critical or critical
    # Weakest offer strength → a need on this skill stays under-satisfied
    # (satisfied = offer.strength < need.intensity), which is exactly
    # the regime in which the deletion test produces a clear top factor.
    weakest = min(required.items(), key=lambda kv: kv[1])[0] if required else c1

    if archetype == "high_cap_low_need":
        # Critical caps deliberately below the proposer-induced gap
        # (≈0.9 for the top requirement) so the winner does NOT saturate
        # S_cap at 1.0 — that's the whole point of the contrast testset.
        caps = {c1: 0.85, c2: 0.75}
        for i, s in enumerate(other[:3]):
            caps[s] = 0.55 + 0.05 * i
        # High-intensity need on the weakest offer:
        #   satisfied = min(weakest_offer_strength, 0.95) < 0.95 → S_need < 1.0.
        needs = {weakest: 0.95}
        return caps, needs

    if archetype == "low_cap_high_need":
        caps = {c1: 0.25, c2: 0.20}
        for i, s in enumerate(other[:3]):
            caps[s] = 0.35 + 0.05 * i
        # Needs matched to the strong offers — these candidates *want*
        # the work, and the offers can fully cover them.
        needs = {c1: 0.85, c2: 0.80}
        return caps, needs

    if archetype == "balanced":
        caps = {c1: 0.55, c2: 0.55}
        for i, s in enumerate(other[:3]):
            caps[s] = 0.50 + 0.03 * i
        # One fully-satisfied need (matched to strong c1 offer)
        # + one partially-satisfied need (matched to weakest offer).
        needs = {c1: 0.50, weakest: 0.90}
        return caps, needs

    if archetype == "specialist":
        caps = {c1: 0.95, c2: 0.20}
        for i, s in enumerate(other[:2]):
            caps[s] = 0.40 + 0.05 * i
        # Wants help with their weakness (c2) AND the weak offer.
        needs = {c2: 0.70, weakest: 0.85}
        return caps, needs

    if archetype == "distractor":
        caps = {}
        for i, s in enumerate(_OFF_TOPIC_CAPS[:3]):
            caps[s] = 0.85 - 0.05 * i
        # Tiny presence on one critical skill so they're not gate-failed.
        caps[c1] = 0.30
        needs = {weakest: 0.90}
        return caps, needs

    raise ValueError(f"Unknown archetype: {archetype!r}")


_ARCHETYPE_ORDER = [
    "high_cap_low_need",
    "low_cap_high_need",
    "balanced",
    "specialist",
    "distractor",
]


# ---------------------------------------------------------------------------
# Per-task builder
# ---------------------------------------------------------------------------

def _build_contrast_task(base_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite one task entry into a contrast version."""
    out = deepcopy(base_entry)
    task = out["task"]
    task_id = task["task_id"]

    base_required = dict(task.get("required_skills", {}))
    if not base_required:
        return out
    bumped = _bumped_required(base_required)
    task["required_skills"] = bumped
    task["title"] = task.get("title", "") + " [faithfulness-contrast]"
    task["description"] = (
        task.get("description", "")
        + " (proposer is intentionally weak on the top-2 required skills.)"
    )

    critical, non_critical = _split_required(bumped)

    out["proposer_profile"] = _rewrite_proposer(
        out["proposer_profile"], critical, non_critical,
    )

    # Offers mirror the (bumped) required skills so that need→offer
    # attention can attend to them. The variation in S_need comes from
    # candidate needs hitting both matched and unmatched skills.
    task["offer_skills"] = {s: float(level) for s, level in bumped.items()}

    new_cands: List[Dict[str, Any]] = []
    for i, arche in enumerate(_ARCHETYPE_ORDER, start=1):
        caps, needs = _archetype_caps_and_needs(
            arche, critical, non_critical, bumped,
        )
        new_cands.append(_make_candidate(task_id, i, caps, needs, arche))
    out["candidates"] = new_cands

    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def generate() -> Dict[str, Any]:
    source = json.loads(SOURCE_PATH.read_text())
    base_tasks = source.get("tasks", [])
    contrast_tasks = [_build_contrast_task(t) for t in base_tasks]
    return {
        "metadata": {
            "generated_by": str(
                Path("Experiments/interpretability")
                / "generate_faithfulness_contrast_testset.py"
            ),
            "source": str(Path("simulator") / SOURCE_PATH.name),
            "goal": (
                "non-saturated S_cap and contrastive winner-loser explanations"
            ),
            "num_tasks": len(contrast_tasks),
            "candidate_archetypes": _ARCHETYPE_ORDER,
            "off_topic_caps": _OFF_TOPIC_CAPS,
            "design_note": (
                "Needs are on-topic skill names matched to specific offers, "
                "with intensity > offer.strength when partial satisfaction "
                "is desired. This keeps the deletion-test linked_offer "
                "field actionable under SimpleEncoder."
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
