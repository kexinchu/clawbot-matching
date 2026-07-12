"""Profile-conditioned candidate responder for PeopleJoin ask_candidate."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from Experiments.peoplejoin.isolation import assert_peoplejoin_public_inputs
from Experiments.peoplejoin.llm_client import PeopleJoinLLMClient

RESPONDER_SYSTEM_TEMPLATE = """You are a collaboration candidate answering questions about yourself.
Be honest, concise, and answer ONLY using information supported by your public structured profile below.
Do NOT invent capabilities, commitments, outcomes, or knowledge of other candidates.
Do NOT claim access to hidden simulator variables, rewards, MapScore, or other people's private data.
If the profile does not support an answer, say you do not know from your profile.

Your public structured profile (JSON):
{profile_json}

Task context the requester shared (public only):
{task_json}
"""


def public_candidate_profile(profile: dict) -> Dict[str, Any]:
    """Extract only fields the responder is allowed to use."""
    prefs = profile.get("preferences", {}) or {}
    constraints = profile.get("constraints", {}) or {}
    return {
        "user_id": profile.get("user_id"),
        "role": profile.get("role"),
        "capabilities": profile.get("capabilities", {}) or {},
        "needs": profile.get("needs", {}) or {},
        "availability": prefs.get("availability"),
        "timezone": prefs.get("timezone"),
        "workload": constraints.get("workload", prefs.get("current_load")),
        "current_load": prefs.get("current_load"),
        "communication_style": prefs.get("communication_style", "not specified"),
        "schedule_flexibility": constraints.get("schedule_flexibility"),
        "interests": prefs.get("interests"),
        "constraints": constraints,
        "history_summary": profile.get("history_summary"),
    }


def public_task_view(task: dict) -> Dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "title": task.get("title"),
        "description": task.get("description"),
        "required_skills": task.get("required_skills", {}) or {},
        "offers": task.get("offers", {}) or {},
        "metadata": task.get("metadata", {}) or {},
    }


class CandidateResponder:
    """Answer ask_candidate using only public structured profile + task."""

    def __init__(
        self,
        *,
        llm: PeopleJoinLLMClient,
        profiles_by_id: Dict[str, dict],
        task: dict,
    ):
        self.llm = llm
        self.profiles_by_id = {
            cid: public_candidate_profile(prof) for cid, prof in profiles_by_id.items()
        }
        self.task = public_task_view(task)
        assert_peoplejoin_public_inputs(self.profiles_by_id, self.task)
        self.ask_log: List[Dict[str, Any]] = []

    @classmethod
    def from_task_entry(cls, task_entry: dict, llm: PeopleJoinLLMClient) -> "CandidateResponder":
        profiles = {
            c["candidate_profile"]["user_id"]: c["candidate_profile"]
            for c in task_entry.get("candidates", [])
        }
        return cls(llm=llm, profiles_by_id=profiles, task=task_entry["task"])

    def build_system_prompt(self, candidate_id: str) -> str:
        profile = self.profiles_by_id[candidate_id]
        prompt = RESPONDER_SYSTEM_TEMPLATE.format(
            profile_json=json.dumps(profile, indent=2, sort_keys=True),
            task_json=json.dumps(self.task, indent=2, sort_keys=True),
        )
        assert_peoplejoin_public_inputs(prompt)
        return prompt

    def answer(self, candidate_id: str, question: str) -> str:
        if candidate_id not in self.profiles_by_id:
            raise KeyError(f"Unknown candidate_id for responder: {candidate_id}")
        system = self.build_system_prompt(candidate_id)
        messages = [{"role": "user", "content": question}]
        assert_peoplejoin_public_inputs(system, messages)

        def mock_fn(_system: str, msgs: List[Dict[str, str]]) -> str:
            return self._mock_answer(candidate_id, msgs[-1]["content"] if msgs else question)

        reply = self.llm.complete(system, messages, mock_fn=mock_fn)
        record = {
            "candidate_id": candidate_id,
            "question": question,
            "answer": reply,
        }
        assert_peoplejoin_public_inputs(record)
        self.ask_log.append(record)
        return reply

    def _mock_answer(self, candidate_id: str, question: str) -> str:
        """Deterministic template responder for unit/smoke tests."""
        profile = self.profiles_by_id[candidate_id]
        q = (question or "").lower()
        caps = profile.get("capabilities", {}) or {}
        needs = profile.get("needs", {}) or {}
        offers = self.task.get("offers", {}) or {}
        reqs = self.task.get("required_skills", {}) or {}

        parts: List[str] = []
        if any(k in q for k in ("capabilit", "skill", "experience", "can you", "strength")):
            top_caps = sorted(caps.items(), key=lambda x: (-float(x[1]), x[0]))[:5]
            parts.append(
                "My top capabilities are: "
                + ", ".join(f"{k}={float(v):.2f}" for k, v in top_caps)
                + "."
            )
            covered = [s for s, lvl in reqs.items() if float(caps.get(s, 0.0)) >= float(lvl)]
            if covered:
                parts.append("I meet these required skills at the stated levels: " + ", ".join(covered) + ".")
            else:
                parts.append("I may not fully meet every required skill level from my profile alone.")

        if any(k in q for k in ("need", "interest", "offer", "benefit", "want", "looking")):
            top_needs = sorted(needs.items(), key=lambda x: (-float(x[1]), x[0]))[:4]
            parts.append(
                "My needs/interests include: "
                + ", ".join(f"{k}={float(v):.2f}" for k, v in top_needs)
                + "."
            )
            matches = []
            for skill, intensity in needs.items():
                offer_val = float(offers.get(skill, 0.0))
                if offer_val > 0 and float(intensity) > 0.3:
                    matches.append(f"{skill}(offer={offer_val:.2f}, need={float(intensity):.2f})")
            if matches:
                parts.append("Task offers that align with my needs: " + ", ".join(matches) + ".")
            else:
                parts.append("From my profile, task offers do not strongly match my top needs.")

        if any(k in q for k in ("avail", "time", "hour", "schedule", "when", "timezone", "tz")):
            parts.append(
                f"Availability={profile.get('availability')}; timezone={profile.get('timezone')}; "
                f"schedule_flexibility={profile.get('schedule_flexibility')}."
            )

        if any(k in q for k in ("workload", "load", "busy", "bandwidth", "capacity")):
            parts.append(
                f"Current workload/load={profile.get('workload')}; "
                f"current_load={profile.get('current_load')}."
            )

        if any(k in q for k in ("style", "communicat", "collab", "work with", "prefer")):
            parts.append(
                f"Communication/collaboration style={profile.get('communication_style')}. "
                f"History: {profile.get('history_summary')}"
            )

        if any(k in q for k in ("other candidate", "who else", "compare", "rank")):
            parts.append("I only know my own profile and cannot speak for other candidates.")

        if not parts:
            # Generic profile-grounded answer.
            top_caps = sorted(caps.items(), key=lambda x: (-float(x[1]), x[0]))[:3]
            parts.append(
                f"I am a {profile.get('role')} ({candidate_id}). "
                f"Availability={profile.get('availability')}, workload={profile.get('workload')}, "
                f"timezone={profile.get('timezone')}. "
                "Key capabilities: "
                + ", ".join(f"{k}={float(v):.2f}" for k, v in top_caps)
                + ". "
                f"History: {profile.get('history_summary')}"
            )

        answer = " ".join(parts)
        # Soft refusal for clearly unsupported fabrication asks.
        if re.search(r"\b(guarantee|promise|oracle|mapscore|latent)\b", q):
            answer += " I will not invent guarantees, MapScore values, or hidden outcomes."
        return answer.strip()
