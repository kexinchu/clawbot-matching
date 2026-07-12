"""Many-to-one deferred acceptance (requester-proposing Gale–Shapley).

All stable-matching methods in this experiment must call the same allocator.
Preference construction is the only allowed method difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DAResult:
    """Final matching plus proposal/accept/reject trace."""

    matching: dict[str, str | None]  # task_id -> candidate_id or None
    candidate_loads: dict[str, list[str]]  # candidate_id -> assigned task_ids
    trace: list[dict[str, Any]] = field(default_factory=list)
    proposals: int = 0
    accepts: int = 0
    rejects: int = 0
    replacements: int = 0


def _rank_index(preference_list: list[str], other_id: str) -> int:
    """0-based rank; lower is better. Missing => +inf."""
    try:
        return preference_list.index(other_id)
    except ValueError:
        return 10**9


def deferred_acceptance(
    task_ids: list[str],
    candidate_ids: list[str],
    task_prefs: dict[str, list[str]],
    candidate_prefs: dict[str, list[str]],
    *,
    candidate_capacity: int = 1,
    task_capacity: int = 1,
    tie_break_seed: int = 0,
) -> DAResult:
    """Requester/task-proposing many-to-one deferred acceptance.

    Parameters
    ----------
    task_prefs:
        task_id -> ordered list of acceptable candidate_ids (best first).
    candidate_prefs:
        candidate_id -> ordered list of acceptable task_ids (best first).
    candidate_capacity:
        Maximum number of tasks a candidate may hold (default 1).
    task_capacity:
        Maximum number of candidates a task may hold (default 1).
        Currently only task_capacity=1 is exercised by the benchmark.
    tie_break_seed:
        Unused for ordering (lists are already tie-broken), but recorded in
        the trace for reproducibility metadata.
    """
    if task_capacity != 1:
        raise NotImplementedError("Only task_capacity=1 is supported in this allocator.")
    if candidate_capacity < 1:
        raise ValueError("candidate_capacity must be >= 1")

    # Only keep mutually acceptable edges present in both preference lists.
    filtered_task_prefs: dict[str, list[str]] = {}
    for tid in task_ids:
        prefs = []
        for cid in task_prefs.get(tid, []):
            if tid in candidate_prefs.get(cid, []):
                prefs.append(cid)
        filtered_task_prefs[tid] = prefs

    filtered_cand_prefs: dict[str, list[str]] = {}
    for cid in candidate_ids:
        prefs = []
        for tid in candidate_prefs.get(cid, []):
            if cid in filtered_task_prefs.get(tid, []):
                prefs.append(tid)
        filtered_cand_prefs[cid] = prefs

    next_proposal_idx = {tid: 0 for tid in task_ids}
    matching: dict[str, str | None] = {tid: None for tid in task_ids}
    holdings: dict[str, list[str]] = {cid: [] for cid in candidate_ids}

    free_tasks = [tid for tid in task_ids if filtered_task_prefs.get(tid)]
    # Deterministic queue order: original task_ids order, then FIFO of freed tasks.
    free_set = set(free_tasks)
    queue = list(free_tasks)

    trace: list[dict[str, Any]] = []
    proposals = accepts = rejects = replacements = 0

    def cand_prefers(cid: str, new_tid: str, old_tid: str) -> bool:
        return _rank_index(filtered_cand_prefs[cid], new_tid) < _rank_index(
            filtered_cand_prefs[cid], old_tid
        )

    def worst_held(cid: str) -> str:
        held = holdings[cid]
        return max(held, key=lambda tid: _rank_index(filtered_cand_prefs[cid], tid))

    while queue:
        tid = queue.pop(0)
        free_set.discard(tid)
        if matching[tid] is not None:
            continue
        prefs = filtered_task_prefs.get(tid, [])
        idx = next_proposal_idx[tid]
        if idx >= len(prefs):
            matching[tid] = None
            continue

        cid = prefs[idx]
        next_proposal_idx[tid] = idx + 1
        proposals += 1
        event: dict[str, Any] = {
            "event": "propose",
            "task_id": tid,
            "candidate_id": cid,
            "tie_break_seed": tie_break_seed,
        }

        held = holdings[cid]
        if len(held) < candidate_capacity:
            # Accept provisionally.
            holdings[cid].append(tid)
            matching[tid] = cid
            accepts += 1
            event["result"] = "accept"
            trace.append(event)
            continue

        # At capacity: keep the candidate's favourite `capacity` proposers.
        worst = worst_held(cid)
        if cand_prefers(cid, tid, worst):
            # Replace worst.
            holdings[cid].remove(worst)
            matching[worst] = None
            holdings[cid].append(tid)
            matching[tid] = cid
            accepts += 1
            replacements += 1
            event["result"] = "replace"
            event["replaced_task_id"] = worst
            trace.append(event)
            if worst not in free_set and next_proposal_idx[worst] < len(
                filtered_task_prefs.get(worst, [])
            ):
                queue.append(worst)
                free_set.add(worst)
        else:
            rejects += 1
            event["result"] = "reject"
            trace.append(event)
            if tid not in free_set and next_proposal_idx[tid] < len(prefs):
                queue.append(tid)
                free_set.add(tid)

    return DAResult(
        matching=matching,
        candidate_loads=holdings,
        trace=trace,
        proposals=proposals,
        accepts=accepts,
        rejects=rejects,
        replacements=replacements,
    )


def is_stable_matching(
    matching: dict[str, str | None],
    task_prefs: dict[str, list[str]],
    candidate_prefs: dict[str, list[str]],
    *,
    candidate_capacity: int = 1,
) -> bool:
    """Return True iff ``matching`` has no blocking pair under stated prefs."""
    holdings: dict[str, list[str]] = {cid: [] for cid in candidate_prefs}
    for tid, cid in matching.items():
        if cid is not None:
            holdings.setdefault(cid, []).append(tid)

    for tid, prefs in task_prefs.items():
        current = matching.get(tid)
        current_rank = (
            _rank_index(prefs, current) if current is not None else 10**9
        )
        for cid in prefs:
            if _rank_index(prefs, cid) >= current_rank:
                break
            # Task prefers cid to current assignment.
            held = holdings.get(cid, [])
            if tid not in candidate_prefs.get(cid, []):
                continue
            if len(held) < candidate_capacity:
                return False
            # Capacity full: blocking if cid prefers tid to some held task.
            for held_tid in held:
                if _rank_index(candidate_prefs[cid], tid) < _rank_index(
                    candidate_prefs[cid], held_tid
                ):
                    return False
    return True


def count_blocking_pairs(
    matching: dict[str, str | None],
    task_prefs: dict[str, list[str]],
    candidate_prefs: dict[str, list[str]],
    *,
    candidate_capacity: int = 1,
) -> int:
    """Count blocking pairs under the given preference lists."""
    holdings: dict[str, list[str]] = {cid: [] for cid in candidate_prefs}
    for tid, cid in matching.items():
        if cid is not None:
            holdings.setdefault(cid, []).append(tid)

    blocking = 0
    seen: set[tuple[str, str]] = set()
    for tid, prefs in task_prefs.items():
        current = matching.get(tid)
        current_rank = (
            _rank_index(prefs, current) if current is not None else 10**9
        )
        for cid in prefs:
            if _rank_index(prefs, cid) >= current_rank:
                break
            if tid not in candidate_prefs.get(cid, []):
                continue
            held = holdings.get(cid, [])
            blocks = False
            if len(held) < candidate_capacity:
                blocks = True
            else:
                for held_tid in held:
                    if _rank_index(candidate_prefs[cid], tid) < _rank_index(
                        candidate_prefs[cid], held_tid
                    ):
                        blocks = True
                        break
            if blocks and (tid, cid) not in seen:
                seen.add((tid, cid))
                blocking += 1
    return blocking
