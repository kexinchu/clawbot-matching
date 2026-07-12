"""ReAct-style PeopleJoin controller for collaborator discovery."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Experiments.peoplejoin.bm25_retriever import BM25CandidateRetriever
from Experiments.peoplejoin.candidate_responder import CandidateResponder, public_task_view
from Experiments.peoplejoin.isolation import assert_peoplejoin_public_inputs
from Experiments.peoplejoin.llm_client import PeopleJoinLLMClient

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class PeopleJoinBudget:
    max_search: int = 3
    max_ask: int = 6
    max_actions: int = 10
    search_top_k: int = 5


@dataclass
class PeopleJoinTraceStep:
    step: int
    action: Dict[str, Any]
    observation: Any
    raw_controller_output: str = ""
    parse_error: Optional[str] = None
    retried: bool = False


@dataclass
class PeopleJoinResult:
    method: str
    selected_candidate_id: str
    rationale: str
    trace: List[Dict[str, Any]]
    interaction_cost: Dict[str, Any]
    retrieval_diagnostics: Dict[str, Any]
    finish_fallback: bool
    errors: List[str]
    seed: int
    mock_mode: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CONTROLLER_SYSTEM_TEMPLATE = """You are a PeopleJoin-Reactive collaborator discovery controller.
Your job is to find the best collaborator for a requester task by searching a public candidate directory and asking candidates questions.
You do NOT have MapScore, S_cap, S_need, UCB, Dreaming scores, or oracle outcomes.
You must never invent candidate ids outside the pool.

Requester public profile (JSON):
{requester_json}

Task (JSON):
{task_json}

Action schema — reply with ONE strict JSON object only:
1) {{"action":"search_relevant_people","query":"..."}}
2) {{"action":"ask_candidate","candidate_id":"...","question":"..."}}
3) {{"action":"finish","selected_candidate_id":"...","rationale":"..."}}

Budgets:
- max_search={max_search}
- max_ask={max_ask}
- max_total_actions={max_actions}
- each search returns at most top-{search_top_k} public directory cards

Candidate directory contains only public cards (role, summary, strengths, risks).
Use search to discover candidates, ask to gather profile-grounded details, then finish with exactly one candidate_id from the pool.
Remaining budget this turn: search_left={search_left}, ask_left={ask_left}, actions_left={actions_left}.
Known candidate ids from prior search/asks: {known_ids}
"""


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class PeopleJoinReactiveController:
    """LLM controller that loops over search / ask / finish under budgets."""

    def __init__(
        self,
        *,
        task_entry: dict,
        llm: PeopleJoinLLMClient,
        budget: Optional[PeopleJoinBudget] = None,
        retriever: Optional[BM25CandidateRetriever] = None,
        responder: Optional[CandidateResponder] = None,
    ):
        self.task_entry = task_entry
        self.task = public_task_view(task_entry["task"])
        self.requester = {
            k: task_entry["proposer_profile"].get(k)
            for k in (
                "user_id",
                "role",
                "capabilities",
                "needs",
                "preferences",
                "constraints",
                "history_summary",
            )
        }
        self.budget = budget or PeopleJoinBudget()
        self.llm = llm
        self.retriever = retriever or BM25CandidateRetriever.from_task_entry(task_entry)
        self.responder = responder or CandidateResponder.from_task_entry(task_entry, llm)
        self.pool_ids = {
            c["candidate_profile"]["user_id"] for c in task_entry.get("candidates", [])
        }
        self.errors: List[str] = []
        self.trace: List[PeopleJoinTraceStep] = []
        self.search_calls = 0
        self.ask_calls = 0
        self.action_calls = 0
        self.known_ids: List[str] = []
        self.contacted_ids: List[str] = []
        self.first_search_top5: List[str] = []
        self.malformed_json_count = 0
        self.invalid_action_count = 0
        self.finish_fallback = False
        assert_peoplejoin_public_inputs(self.requester, self.task)

    def _remaining(self) -> Tuple[int, int, int]:
        return (
            max(0, self.budget.max_search - self.search_calls),
            max(0, self.budget.max_ask - self.ask_calls),
            max(0, self.budget.max_actions - self.action_calls),
        )

    def _system_prompt(self) -> str:
        search_left, ask_left, actions_left = self._remaining()
        prompt = CONTROLLER_SYSTEM_TEMPLATE.format(
            requester_json=json.dumps(self.requester, indent=2, sort_keys=True),
            task_json=json.dumps(self.task, indent=2, sort_keys=True),
            max_search=self.budget.max_search,
            max_ask=self.budget.max_ask,
            max_actions=self.budget.max_actions,
            search_top_k=self.budget.search_top_k,
            search_left=search_left,
            ask_left=ask_left,
            actions_left=actions_left,
            known_ids=json.dumps(self.known_ids),
        )
        assert_peoplejoin_public_inputs(prompt)
        return prompt

    def _history_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    "Begin collaborator discovery. Choose the next action as strict JSON. "
                    f"Default first action should search using the task goal: {self.task.get('title', '')}."
                ),
            }
        ]
        for step in self.trace:
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(step.action, sort_keys=True),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Observation:\n"
                    + json.dumps(step.observation, sort_keys=True, default=str)
                    + "\nChoose the next action as strict JSON.",
                }
            )
        assert_peoplejoin_public_inputs(messages)
        return messages

    def _mock_controller(self, _system: str, messages: List[Dict[str, str]]) -> str:
        """Deterministic ReAct policy for mock mode / fallbacks."""
        search_left, ask_left, actions_left = self._remaining()
        if actions_left <= 0:
            cid = self._fallback_candidate_id()
            return json.dumps(
                {
                    "action": "finish",
                    "selected_candidate_id": cid,
                    "rationale": "Budget exhausted; selecting best-known candidate from search.",
                }
            )

        # First action or no known ids: search.
        if self.search_calls == 0 or (not self.known_ids and search_left > 0):
            query = (
                f"{self.task.get('title', '')} "
                f"{' '.join((self.task.get('required_skills') or {}).keys())}"
            ).strip()
            return json.dumps({"action": "search_relevant_people", "query": query})

        # Ask unanswered known candidates until budget / need a finish.
        for cid in self.known_ids:
            if cid not in self.contacted_ids and ask_left > 0:
                q = (
                    "What are your top capabilities for this task, your availability and workload, "
                    "and whether the task offers match your needs?"
                )
                return json.dumps(
                    {"action": "ask_candidate", "candidate_id": cid, "question": q}
                )

        # Optional second search with needs/offers vocabulary if budget remains.
        if search_left > 0 and self.search_calls < 2:
            query = (
                f"role strengths risks "
                f"{' '.join((self.task.get('offers') or {}).keys())}"
            )
            return json.dumps({"action": "search_relevant_people", "query": query})

        # Finish with first contacted, else first known, else pool fallback.
        cid = self._fallback_candidate_id()
        rationale = (
            f"Selected {cid} after {self.search_calls} search(es) and "
            f"{self.ask_calls} ask(s) based on directory fit and profile answers."
        )
        return json.dumps(
            {
                "action": "finish",
                "selected_candidate_id": cid,
                "rationale": rationale,
            }
        )

    def _fallback_candidate_id(self) -> str:
        if self.contacted_ids:
            return self.contacted_ids[0]
        if self.known_ids:
            return self.known_ids[0]
        if self.first_search_top5:
            return self.first_search_top5[0]
        # Deterministic: lexicographically first pool id.
        return sorted(self.pool_ids)[0]

    def _parse_action(self, raw: str) -> Tuple[Optional[dict], Optional[str]]:
        obj = _extract_json_object(raw)
        if obj is None:
            return None, "malformed_json"
        action = str(obj.get("action", "")).strip()
        if action not in {"search_relevant_people", "ask_candidate", "finish"}:
            return None, f"invalid_action:{action or 'missing'}"
        if action == "search_relevant_people":
            if "query" not in obj or not str(obj.get("query", "")).strip():
                return None, "missing_query"
        elif action == "ask_candidate":
            if not str(obj.get("candidate_id", "")).strip() or not str(obj.get("question", "")).strip():
                return None, "missing_ask_fields"
        elif action == "finish":
            if not str(obj.get("selected_candidate_id", "")).strip():
                return None, "missing_selected_candidate_id"
        return obj, None

    def _execute(self, action: dict) -> Tuple[Any, Optional[str]]:
        name = action["action"]
        search_left, ask_left, actions_left = self._remaining()
        if actions_left <= 0:
            return {"error": "action_budget_exhausted"}, "action_budget_exhausted"

        if name == "search_relevant_people":
            if search_left <= 0:
                self.invalid_action_count += 1
                return {"error": "search_budget_exhausted"}, "search_budget_exhausted"
            query = str(action.get("query", ""))
            hits = self.retriever.search(query, top_k=self.budget.search_top_k)
            self.search_calls += 1
            self.action_calls += 1
            ids = [h.candidate_id for h in hits]
            if not self.first_search_top5:
                self.first_search_top5 = list(ids)
            for cid in ids:
                if cid not in self.known_ids:
                    self.known_ids.append(cid)
            obs = {"results": [h.to_observation() for h in hits]}
            assert_peoplejoin_public_inputs(obs)
            return obs, None

        if name == "ask_candidate":
            if ask_left <= 0:
                self.invalid_action_count += 1
                return {"error": "ask_budget_exhausted"}, "ask_budget_exhausted"
            cid = str(action.get("candidate_id", "")).strip()
            question = str(action.get("question", "")).strip()
            if cid not in self.pool_ids:
                self.invalid_action_count += 1
                return {
                    "error": "candidate_not_in_pool",
                    "candidate_id": cid,
                }, "candidate_not_in_pool"
            answer = self.responder.answer(cid, question)
            self.ask_calls += 1
            self.action_calls += 1
            if cid not in self.known_ids:
                self.known_ids.append(cid)
            if cid not in self.contacted_ids:
                self.contacted_ids.append(cid)
            obs = {"candidate_id": cid, "answer": answer}
            assert_peoplejoin_public_inputs(obs)
            return obs, None

        if name == "finish":
            cid = str(action.get("selected_candidate_id", "")).strip()
            rationale = str(action.get("rationale", "")).strip() or "No rationale provided."
            if cid not in self.pool_ids:
                self.invalid_action_count += 1
                return {
                    "error": "finish_candidate_not_in_pool",
                    "candidate_id": cid,
                }, "finish_candidate_not_in_pool"
            self.action_calls += 1
            return {
                "selected_candidate_id": cid,
                "rationale": rationale,
                "done": True,
            }, None

        self.invalid_action_count += 1
        return {"error": "unknown_action"}, "unknown_action"

    def run(self) -> PeopleJoinResult:
        selected_id: Optional[str] = None
        rationale = ""
        max_steps = self.budget.max_actions + 3  # allow parse retries headroom
        api_degraded = False

        for _ in range(max_steps):
            if self.action_calls >= self.budget.max_actions and selected_id is None:
                break
            system = self._system_prompt()
            messages = self._history_messages()
            try:
                raw = self.llm.complete(system, messages, mock_fn=self._mock_controller)
            except Exception as exc:
                # After client-side retries, keep experiment alive with a
                # deterministic step and explicit degradation flag.
                api_degraded = True
                self.errors.append(f"controller_api_failure:{type(exc).__name__}:{str(exc)[:200]}")
                raw = self._mock_controller(system, messages)
                self.errors.append("used_deterministic_action_fallback_after_api_error")

            action, parse_err = self._parse_action(raw)
            retried = False
            if parse_err is not None:
                self.malformed_json_count += 1
                self.errors.append(parse_err)
                # One safe retry.
                retry_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid strict JSON for the action schema. "
                            "Reply again with ONE valid JSON action object only."
                        ),
                    },
                ]
                try:
                    raw2 = self.llm.complete(system, retry_messages, mock_fn=self._mock_controller)
                except Exception as exc:
                    api_degraded = True
                    self.errors.append(
                        f"controller_retry_api_failure:{type(exc).__name__}:{str(exc)[:200]}"
                    )
                    raw2 = self._mock_controller(system, messages)
                action, parse_err2 = self._parse_action(raw2)
                retried = True
                raw = raw2
                if parse_err2 is not None:
                    self.malformed_json_count += 1
                    self.errors.append(parse_err2)
                    # Deterministic fallback action.
                    action = json.loads(self._mock_controller(system, messages))
                    self.errors.append("used_deterministic_action_fallback")

            assert action is not None
            try:
                obs, exec_err = self._execute(action)
            except Exception as exc:
                # Responder API failure: record and continue toward finish.
                api_degraded = True
                self.errors.append(f"execute_api_failure:{type(exc).__name__}:{str(exc)[:200]}")
                obs, exec_err = {"error": "execute_api_failure", "detail": str(exc)[:200]}, "execute_api_failure"
            step = PeopleJoinTraceStep(
                step=len(self.trace) + 1,
                action=action,
                observation=obs,
                raw_controller_output=raw,
                parse_error=exec_err,
                retried=retried,
            )
            self.trace.append(step)
            if exec_err in {
                "candidate_not_in_pool",
                "finish_candidate_not_in_pool",
                "execute_api_failure",
            }:
                # Continue loop; do not accept invalid selection.
                continue
            if isinstance(obs, dict) and obs.get("done"):
                selected_id = str(obs["selected_candidate_id"])
                rationale = str(obs.get("rationale", ""))
                break

        if selected_id is None or selected_id not in self.pool_ids:
            self.finish_fallback = True
            selected_id = self._fallback_candidate_id()
            rationale = (
                rationale
                or "Finish fallback: budgets exhausted or invalid finish; selected deterministic candidate."
            )
            self.errors.append("finish_fallback")

        interaction_cost = {
            "search_calls": self.search_calls,
            "candidate_messages": self.ask_calls,
            "unique_candidates_contacted": len(self.contacted_ids),
            "controller_api_calls": 0,
            "responder_api_calls": len(self.responder.ask_log),
            "total_api_calls": self.llm.usage.api_calls,
            "input_tokens": self.llm.usage.input_tokens,
            "output_tokens": self.llm.usage.output_tokens,
            "total_tokens": self.llm.usage.total_tokens,
            "wall_clock_seconds": round(self.llm.usage.wall_clock_seconds, 6),
            "api_errors": list(self.llm.usage.errors),
            "api_degraded": api_degraded or bool(self.llm.usage.errors),
        }
        # Controller calls = total - responder asks (same shared client).
        interaction_cost["controller_api_calls"] = max(
            0, interaction_cost["total_api_calls"] - interaction_cost["responder_api_calls"]
        )
        if self.llm.usage.errors:
            self.errors.extend(self.llm.usage.errors)

        retrieval_diagnostics = {
            "first_search_top5": list(self.first_search_top5),
            "selected_in_first_search_top5": selected_id in self.first_search_top5,
            "unique_candidates_contacted": len(self.contacted_ids),
            "contacted_candidate_ids": list(self.contacted_ids),
            "known_candidate_ids": list(self.known_ids),
            "finish_fallback": self.finish_fallback,
            "malformed_json_count": self.malformed_json_count,
            "invalid_action_count": self.invalid_action_count,
            "api_degraded": interaction_cost["api_degraded"],
            "search_log": list(self.retriever.search_log),
            "ask_log": list(self.responder.ask_log),
        }

        return PeopleJoinResult(
            method="peoplejoin_reactive",
            selected_candidate_id=selected_id,
            rationale=rationale,
            trace=[asdict(s) for s in self.trace],
            interaction_cost=interaction_cost,
            retrieval_diagnostics=retrieval_diagnostics,
            finish_fallback=self.finish_fallback,
            errors=list(self.errors),
            seed=self.llm.seed,
            mock_mode=self.llm.mock_mode,
        )


def run_peoplejoin_reactive(
    task_entry: dict,
    *,
    llm: Optional[PeopleJoinLLMClient] = None,
    budget: Optional[PeopleJoinBudget] = None,
    seed: int = 0,
    mode: str = "mock",
    model: str = "openai/gpt-4o-mini",
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.2,
) -> PeopleJoinResult:
    """Convenience entry point used by the evaluation runner."""
    client = llm or PeopleJoinLLMClient(
        mode=mode,
        model=model,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
        allow_mock_fallback=(mode == "mock"),
    )
    client.reset_usage()
    controller = PeopleJoinReactiveController(
        task_entry=task_entry,
        llm=client,
        budget=budget,
    )
    return controller.run()
