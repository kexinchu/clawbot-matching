"""
simulator/importance_ranker.py
==============================
DyLAN-inspired persona importance ranking + top-k pruning.

rank_persona_importance():
  - Computes a raw importance score per persona
  - Softmax-normalizes to weights summing to 1.0
  - Returns list[PersonaImportance] sorted descending

select_top_k():
  - Prunes to top_k personas using the ranked weights
  - Returns (selected, pruned) lists of PersonaSpec
"""
from __future__ import annotations

from simulator.config import SimulatorConfig
from simulator.types import (
    DeliberationTrace,
    MatchingContext,
    PersonaImportance,
    PersonaOpinion,
    PersonaSpec,
)
from simulator.utils import softmax


def rank_persona_importance(
    opinions: dict[str, PersonaOpinion],
    context: MatchingContext,
    trace: DeliberationTrace,
    config: SimulatorConfig,
) -> list[PersonaImportance]:
    """
    Compute per-persona importance scores inspired by DyLAN.

    Raw score components (all summed with learned or heuristic weights):
      1. relevance: how many task-relevant skills/criteria does this persona cover?
      2. non_redundancy: how unique is this persona's concerns relative to others?
      3. confidence: higher confidence → higher weight
      4. consistency: does this persona's opinion align with the final group summary?
      5. decision_impact: how decisive is this persona's utility (far from 0)?
      6. decision_flip: leave-one-out utility change × threshold-crossing signal
         (lightweight approximation — not strict causal attribution)

    Returns a sorted list of PersonaImportance (descending by weight).
    """
    if not opinions:
        return []

    personas = list(opinions.keys())
    n = len(personas)
    raw_scores: dict[str, float] = {}

    # Pre-compute full-group weights for decision-flip computation
    # (weights are normalized after this loop, so raw values are proportional)
    _tmp_scores: dict[str, float] = {}
    for pid, opinion in opinions.items():
        relevance = _compute_relevance(pid, context)
        non_redundancy = _compute_non_redundancy(pid, opinions)
        conf = opinion.confidence
        consistency = _compute_consistency(opinion, trace.final_opinions)
        impact = abs(opinion.utility_score)
        _tmp_scores[pid] = 0.20 * relevance + 0.25 * non_redundancy + 0.20 * conf + 0.15 * consistency + 0.20 * impact

    # Normalize to get proportional weights (needed for LOO utility computation)
    total = sum(_tmp_scores.values())
    weights_proportional = {pid: s / total for pid, s in _tmp_scores.items()} if total > 0 else {}

    for pid, opinion in opinions.items():
        relevance = _compute_relevance(pid, context)
        non_redundancy = _compute_non_redundancy(pid, opinions)
        conf = opinion.confidence
        consistency = _compute_consistency(opinion, trace.final_opinions)
        impact = abs(opinion.utility_score)
        flip = _compute_decision_flip_impact(pid, opinion, opinions, weights_proportional, config)

        # Heuristic weights (could be learned later)
        raw = (
            0.20 * relevance
            + 0.25 * non_redundancy
            + 0.20 * conf
            + 0.15 * consistency
            + 0.20 * impact
            + 0.10 * flip   # lighter weight — approximation, not ground truth
        )
        raw_scores[pid] = raw

    # Normalize with softmax (temperature-controlled)
    scores = [raw_scores[p] for p in personas]
    weights = softmax(scores, temperature=config.importance_temperature)

    ranked = sorted(
        [
            PersonaImportance(
                persona_id=p,
                raw_score=raw_scores[p],
                normalized_weight=w,
            )
            for p, w in zip(personas, weights)
        ],
        key=lambda x: x.normalized_weight,
        reverse=True,
    )
    return ranked


def _compute_relevance(persona_id: str, context: MatchingContext) -> float:
    """Relevance: how much does this persona's focus align with the task?"""
    skill_personas = {"rq_skill", "rq_goal", "cd_fit", "cd_interest"}
    risk_personas = {"rq_time", "rq_trust", "rq_incentive", "cd_opp", "cd_workload"}
    collab_personas = {"rq_collab", "rq_trust", "cd_trust"}

    task_keywords = (
        set(context.task.title.lower().split())
        | set(context.task.description.lower().split())
    )

    if persona_id in skill_personas:
        # High relevance if task has skill requirements
        return min(1.0, len(context.task.required_skills) / 3.0)
    elif persona_id in risk_personas:
        # High relevance if task has risk indicators
        risk_indicators = {"urgent", "complex", "experimental", "tight deadline"}
        return 0.5 + 0.5 * min(1.0, len(task_keywords & risk_indicators) / 2.0)
    elif persona_id in collab_personas:
        # High relevance if no prior history
        return 0.6 if context.history is None else 0.4
    else:
        return 0.4


def _compute_non_redundancy(
    persona_id: str,
    opinions: dict[str, PersonaOpinion],
) -> float:
    """
    Non-redundancy: how different is this persona's concern set from others?
    Returns 1 - avg_jaccard_similarity to others.
    """
    self_concerns = set(c.lower() for c in opinions[persona_id].extracted_concerns)
    if not self_concerns:
        return 0.5  # Neutral if no concerns raised

    similarities = []
    for other_pid, other_op in opinions.items():
        if other_pid == persona_id:
            continue
        other_set = set(c.lower() for c in other_op.extracted_concerns)
        if not other_set:
            continue
        intersection = len(self_concerns & other_set)
        union = len(self_concerns | other_set)
        sim = intersection / union if union > 0 else 0.0
        similarities.append(sim)

    if not similarities:
        return 0.7  # Unique by default
    avg_sim = sum(similarities) / len(similarities)
    return 1.0 - avg_sim


def _compute_consistency(
    opinion: PersonaOpinion,
    final_opinions: dict[str, PersonaOpinion],
) -> float:
    """
    Consistency: does this persona's view align with the group average?
    Returns 1 - normalized_distance to group mean.
    """
    if not final_opinions:
        return 0.5
    group_mean = sum(o.utility_score for o in final_opinions.values()) / len(final_opinions)
    distance = abs(opinion.utility_score - group_mean)
    return max(0.0, 1.0 - distance)


def _compute_decision_flip_impact(
    pid: str,
    opinion: PersonaOpinion,
    opinions: dict[str, PersonaOpinion],
    weights: dict[str, float],
    config: SimulatorConfig,
) -> float:
    """
    Decision-flip impact (lightweight approximation — not strict causal attribution).

    Measures how much the aggregated utility would change if this persona were removed,
    and whether that change would cross an accept/skip or skip/reject threshold.

    Signal = |Δutility| × threshold_crossing_indicator
      - |Δutility|: magnitude of the leave-one-out utility change
      - crossing_indicator: 1.0 if removing this persona flips the decision bucket
                            (e.g., from ACCEPT to SKIP), 0.5 if it moves utility
                            substantially without crossing a threshold, 0.0 otherwise

    This is intentionally approximate: it uses proportional (not yet softmax-normalized)
    weights and static thresholds, so it captures rough influence rather than true
    causal counterfactual attribution.
    """
    if len(opinions) <= 1:
        return 0.0

    # Full-group utility
    def weighted_utility(op_dict: dict[str, PersonaOpinion]) -> float:
        total_w = sum(weights.get(p, 0.0) for p in op_dict)
        if total_w == 0:
            return 0.0
        return sum(o.utility_score * weights.get(p, 0.0) for p, o in op_dict.items()) / total_w

    full_util = weighted_utility(opinions)
    loo_util = weighted_utility({p: o for p, o in opinions.items() if p != pid})

    delta = abs(full_util - loo_util)
    if delta < 1e-6:
        return 0.0

    # Check threshold crossing
    accept_th = config.accept_utility_threshold   # default 0.3
    reject_th = config.reject_utility_threshold  # default -0.3
    skip_band = accept_th - reject_th             # width of skip region

    def in_accept(u: float) -> bool:
        return u >= accept_th

    def in_reject(u: float) -> bool:
        return u <= reject_th

    def in_skip(u: float) -> bool:
        return not in_accept(u) and not in_reject(u)

    # Count crossings: each crossing of a bucket boundary counts as a decision flip
    crossings = 0
    if in_accept(full_util) != in_accept(loo_util):
        crossings += 1
    if in_reject(full_util) != in_reject(loo_util):
        crossings += 1

    crossing_indicator = 1.0 if crossings > 0 else (0.5 if delta > skip_band * 0.3 else 0.0)

    # Cap at 1.0; in practice this is a [0, ~1.0] signal
    return min(1.0, delta * 2.0 * crossing_indicator)


def select_top_k(
    personas: list[PersonaSpec],
    importance_scores: list[PersonaImportance],
    config: SimulatorConfig,
) -> tuple[list[PersonaSpec], list[PersonaSpec]]:
    """
    Prune to top_k personas by importance weight.

    Returns (selected, pruned) as two separate lists of PersonaSpec.
    """
    k = config.top_k_personas
    ranked_map = {imp.persona_id: imp for imp in importance_scores}

    sorted_personas = sorted(
        personas,
        key=lambda p: ranked_map.get(p.persona_id, PersonaImportance(p.persona_id, 0.0, 0.0)).normalized_weight,
        reverse=True,
    )

    selected = sorted_personas[:k]
    pruned = sorted_personas[k:]
    return selected, pruned
