"""
MBRL Matching System — Layer 3: LLM Dream Simulation
=====================================================

After Layer 2 analytical ranking (greedy + UCB), this module
simulates agent-agent conversations for top-K candidates to assess
soft compatibility that the formula cannot capture:

  1. Time & energy availability
  2. Real priority alignment
  3. Collaboration style
  4. Personality fit

Outputs top-3 refined candidates with compatibility reports.
"""

import json
import os
import httpx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Online_learning import (
    UserProfile, Capability, Task, MatchResult, WorldModel,
    dummy_create_task,
)

# Set your API key via environment variable:
#   export ANTHROPIC_API_KEY="sk-ant-..."
# If not set, runs in mock mode with synthetic conversations.
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ============================================================
# Extended Profile: soft attributes for dream simulation
# ============================================================

@dataclass
class SoftProfile:
    """Soft attributes not captured by (Cap, Need) vectors."""
    availability: str          # e.g. "20h/week", "fulltime", "evenings only"
    timezone: str              # e.g. "UTC+8", "EST"
    deadline_pressure: str     # "relaxed", "moderate", "tight"
    collab_style: str          # "async-first", "sync-heavy", "mixed"
    communication: str         # "concise", "detailed", "visual"
    personality_notes: str     # freeform from LLM extraction
    priorities: List[str]      # ranked list of what matters most


@dataclass
class ExtendedProfile:
    """UserProfile + SoftProfile for dream simulation."""
    profile: UserProfile
    soft: SoftProfile

    @property
    def user_id(self) -> str:
        return self.profile.user_id


# ============================================================
# Candidate pool: synthetic extended profiles for testing
# ============================================================

def create_test_candidates() -> Tuple[ExtendedProfile, Task, List[ExtendedProfile]]:
    """
    Create Alice (requester) + task + 5 candidates with diverse soft attributes.
    Layer 2 would have ranked these; we simulate the top-5 entering Layer 3.
    """

    task = Task(
        task_id="task_001",
        goal="Build Bayesian churn model, target NeurIPS",
        Q_T={"bayesian": 0.8, "python": 0.6, "paper_writing": 0.7}
    )

    alice = ExtendedProfile(
        profile=UserProfile(
            user_id="alice",
            capability={
                "bayesian": Capability(0.3, 0.2),
                "python": Capability(0.8, 0.1),
                "paper_writing": Capability(0.4, 0.3),
            },
            need={"bayesian": 0.8, "python": 0.1, "paper_writing": 0.7},
        ),
        soft=SoftProfile(
            availability="30h/week",
            timezone="UTC-5 (EST)",
            deadline_pressure="tight — submission in 3 months",
            collab_style="sync-heavy, likes daily standups",
            communication="concise, action-oriented",
            personality_notes="Startup CTO, fast-paced, direct communicator, "
                             "values efficiency over perfection",
            priorities=["hit NeurIPS deadline", "solid Bayesian modeling",
                        "clean reproducible code"],
        ),
    )

    candidates = [
        # Bob: strong match analytically, good soft fit
        ExtendedProfile(
            profile=UserProfile(
                user_id="bob",
                capability={
                    "bayesian": Capability(0.9, 0.1),
                    "python": Capability(0.5, 0.2),
                    "paper_writing": Capability(0.8, 0.1),
                },
                need={"bayesian": 0.2, "python": 0.7, "paper_writing": 0.9},
            ),
            soft=SoftProfile(
                availability="20h/week",
                timezone="UTC-5 (EST)",
                deadline_pressure="moderate — also has coursework",
                collab_style="mixed, prefers structured weekly meetings",
                communication="detailed, likes to explain reasoning",
                personality_notes="Stats PhD student, methodical, thorough, "
                                 "sometimes slow but high quality output",
                priorities=["first-author NeurIPS paper", "learn industry data pipelines",
                            "build portfolio for job search"],
            ),
        ),

        # Carol: high uncertainty, creative but chaotic
        ExtendedProfile(
            profile=UserProfile(
                user_id="carol",
                capability={
                    "bayesian": Capability(0.6, 0.4),
                    "python": Capability(0.7, 0.35),
                    "paper_writing": Capability(0.5, 0.4),
                },
                need={"bayesian": 0.5, "python": 0.3, "paper_writing": 0.8},
            ),
            soft=SoftProfile(
                availability="15h/week — also freelancing",
                timezone="UTC+1 (CET)",
                deadline_pressure="relaxed — no hard deadlines personally",
                collab_style="async-first, replies in batches",
                communication="visual, loves diagrams and notebooks",
                personality_notes="Creative ML researcher, lots of ideas, "
                                 "jumps between projects, inconsistent follow-through",
                priorities=["explore novel Bayesian methods", "add to publications list",
                            "flexible schedule"],
            ),
        ),

        # Dave: solid skills, timezone clash
        ExtendedProfile(
            profile=UserProfile(
                user_id="dave",
                capability={
                    "bayesian": Capability(0.85, 0.15),
                    "python": Capability(0.75, 0.1),
                    "paper_writing": Capability(0.6, 0.2),
                },
                need={"bayesian": 0.3, "python": 0.2, "paper_writing": 0.7},
            ),
            soft=SoftProfile(
                availability="25h/week",
                timezone="UTC+8 (SGT)",
                deadline_pressure="moderate",
                collab_style="async-first, very responsive on Slack",
                communication="concise, code-speaks-louder",
                personality_notes="Senior ML engineer at a Singapore startup, "
                                 "pragmatic, ships fast, prefers working code over theory",
                priorities=["get a top-venue publication", "transition to research role",
                            "learn academic writing conventions"],
            ),
        ),

        # Eve: perfect skills but overcommitted
        ExtendedProfile(
            profile=UserProfile(
                user_id="eve",
                capability={
                    "bayesian": Capability(0.95, 0.05),
                    "python": Capability(0.9, 0.05),
                    "paper_writing": Capability(0.85, 0.1),
                },
                need={"bayesian": 0.1, "python": 0.1, "paper_writing": 0.3},
            ),
            soft=SoftProfile(
                availability="5h/week — leading 2 other projects",
                timezone="UTC-5 (EST)",
                deadline_pressure="very tight — own deadlines competing",
                collab_style="async only, slow to respond",
                communication="terse, bullet points",
                personality_notes="Tenured professor, brilliant but stretched thin, "
                                 "delegates heavily, hard to get time with",
                priorities=["add another publication to lab output",
                            "mentor junior researchers", "minimal time commitment"],
            ),
        ),

        # Frank: junior but enthusiastic and available
        ExtendedProfile(
            profile=UserProfile(
                user_id="frank",
                capability={
                    "bayesian": Capability(0.4, 0.3),
                    "python": Capability(0.6, 0.25),
                    "paper_writing": Capability(0.3, 0.35),
                },
                need={"bayesian": 0.9, "python": 0.5, "paper_writing": 0.9},
            ),
            soft=SoftProfile(
                availability="40h/week — dedicated to this",
                timezone="UTC-5 (EST)",
                deadline_pressure="none — gap year, fully flexible",
                collab_style="sync-heavy, loves pair programming",
                communication="detailed, asks lots of questions",
                personality_notes="Recent CS masters grad, eager to learn, "
                                 "high energy, very responsive, needs mentoring",
                priorities=["learn Bayesian methods hands-on",
                            "get first research publication",
                            "build relationship with experienced researchers"],
            ),
        ),
    ]

    return alice, task, candidates


# ============================================================
# Agent Persona Builder
# ============================================================

def build_agent_persona(ext: ExtendedProfile, role: str, task: Task) -> str:
    """
    Build a system prompt for an agent representing this user.
    role: 'requester' or 'candidate'
    """

    cap_summary = ", ".join(
        f"{d}: {c.mu:.1f}" for d, c in ext.profile.capability.items()
    )
    need_summary = ", ".join(
        f"{d}: {n:.1f}" for d, n in ext.profile.need.items()
    )
    priorities = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(ext.soft.priorities))

    if role == "requester":
        role_desc = (
            f"You are the AI agent representing {ext.user_id}, who posted the task: "
            f"\"{task.goal}\".\n"
            f"Your goal is to assess whether this candidate is a good fit — "
            f"not just skills, but whether the collaboration will actually work.\n"
            f"Ask about their availability, working style, and what they actually "
            f"want from this collaboration. Probe for potential conflicts."
        )
    else:
        role_desc = (
            f"You are the AI agent representing {ext.user_id}, who is being "
            f"considered for the task: \"{task.goal}\".\n"
            f"Your goal is to honestly represent your user's situation — "
            f"their real availability, true priorities, and working preferences.\n"
            f"Don't oversell. If there are constraints, mention them."
        )

    return f"""{role_desc}

Your user's profile:
- Capabilities: {cap_summary}
- Needs: {need_summary}
- Availability: {ext.soft.availability}
- Timezone: {ext.soft.timezone}
- Deadline pressure: {ext.soft.deadline_pressure}
- Collaboration style: {ext.soft.collab_style}
- Communication preference: {ext.soft.communication}
- Personality: {ext.soft.personality_notes}
- Priorities (ranked):
{priorities}

Rules:
- Respond in 2-4 sentences per turn. Be natural, not robotic.
- Represent your user honestly — including constraints and concerns.
- Focus on practical collaboration logistics, not just skills.
- If something might not work, say so clearly.
"""


# ============================================================
# Dream Conversation Simulator
# ============================================================

JUDGE_SYSTEM = """You are a matching quality evaluator. You just observed a simulated
negotiation between two AI agents — one representing a task requester, one representing
a candidate collaborator.

Evaluate their compatibility on 4 dimensions. For each, give a score from 0.0 to 1.0
and a one-sentence justification.

Respond ONLY with valid JSON, no markdown, no backticks:
{
  "time_energy": {
    "score": 0.0,
    "reason": "..."
  },
  "priority_alignment": {
    "score": 0.0,
    "reason": "..."
  },
  "collab_style": {
    "score": 0.0,
    "reason": "..."
  },
  "personality_fit": {
    "score": 0.0,
    "reason": "..."
  },
  "overall_compatibility": 0.0,
  "top_risk": "one sentence",
  "top_synergy": "one sentence",
  "recommendation": "strong_match | good_match | risky_match | poor_match"
}
"""


class DreamSimulator:
    """
    Simulate agent-agent conversations using the Anthropic API.
    Each conversation is a 'dream' — the match is tested in imagination
    before committing real user attention.

    If ANTHROPIC_API_KEY is not set, runs in mock mode with rule-based responses.
    """

    def __init__(self, n_turns: int = 3, model: str = "claude-sonnet-4-20250514"):
        self.n_turns = n_turns
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.mock_mode = not bool(API_KEY)
        if not self.mock_mode:
            self.client = httpx.Client(timeout=60.0)
        if self.mock_mode:
            print("  [Mock mode — set ANTHROPIC_API_KEY for real LLM calls]\n")

    def _call_llm(self, system: str, messages: list) -> str:
        """Call Anthropic API, or return mock response."""
        if self.mock_mode:
            return self._mock_response(system, messages)

        resp = self.client.post(
            self.api_url,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": 512,
                "system": system,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def _mock_response(self, system: str, messages: list) -> str:
        """
        Rule-based mock: extract key info from system prompt
        and generate realistic-sounding responses.
        """
        # Detect if this is the judge call
        if "matching quality evaluator" in system:
            return self._mock_judge(messages)

        # Extract user info from system prompt
        def extract(key):
            for l in system.split("\n"):
                if key in l:
                    return l.split(key)[1].strip()
            return ""

        avail = extract("Availability:")
        tz = extract("Timezone:")
        style = extract("Collaboration style:")
        personality = extract("Personality:")
        deadline = extract("Deadline pressure:")
        comm = extract("Communication preference:")

        is_requester = "posted the task" in system
        turn = len(messages)

        if is_requester:
            if turn <= 1:
                return (
                    f"Thanks for connecting! A few practical questions before we dive in: "
                    f"How many hours per week can you realistically commit to this? "
                    f"We're on a tight 3-month timeline for NeurIPS. "
                    f"Also, what's your preferred way of collaborating — daily syncs or async?"
                )
            else:
                last_msg = messages[-1]["content"] if messages else ""
                if "5h" in last_msg or "stretched" in last_msg:
                    return (
                        f"That's a concern — 5 hours per week might not be enough given our deadline. "
                        f"The Bayesian modeling alone will need significant iteration time. "
                        f"Could you see any way to increase that, even temporarily for the push phase?"
                    )
                elif "15h" in last_msg or "freelancing" in last_msg:
                    return (
                        f"15 hours could work if we're very structured about it. "
                        f"One thing I want to understand better — you mentioned freelancing on the side. "
                        f"How do you handle competing priorities when a client deadline hits?"
                    )
                else:
                    return (
                        f"That sounds workable. Let me ask about your approach — "
                        f"how would you structure the Bayesian modeling pipeline? "
                        f"And any concerns about the 3-month NeurIPS timeline?"
                    )
        else:
            if turn <= 1:
                return (
                    f"Hi! Great to connect. I can commit about {avail}. "
                    f"I'm based in {tz}. "
                    f"I generally prefer {style}. "
                    f"My main motivation here is getting a strong publication — "
                    f"that aligns well with the NeurIPS goal."
                )
            elif turn <= 3:
                concern = ""
                if "5h" in avail:
                    concern = (f"I should be transparent — with only {avail}, "
                               f"I'm stretched thin across two other projects. "
                               f"I can offer high-level guidance but can't do the heavy lifting. ")
                elif "15h" in avail:
                    concern = (f"With {avail} and freelancing commitments, "
                               f"there might be weeks where I'm less available. "
                               f"I'd need async flexibility for those stretches. ")
                elif "40h" in avail:
                    concern = (f"I have {avail} fully dedicated to this — no competing commitments. "
                               f"I'm eager to learn and happy to take on the bulk of implementation work. ")

                approach = ""
                if "pragmatic" in personality or "ships fast" in personality:
                    approach = ("I'd start with a simple baseline model, validate quickly, "
                                "then iterate. I prefer working code over lengthy theoretical discussions.")
                elif "methodical" in personality:
                    approach = ("I'd start with a thorough literature review, then design the model "
                                "carefully before coding. Quality over speed for me.")
                elif "creative" in personality or "ideas" in personality:
                    approach = ("I'd want to explore some novel approaches — maybe a neural process "
                                "prior or a meta-learning angle. Could differentiate the paper.")
                elif "eager" in personality:
                    approach = ("I'd love to learn the approach from you and implement it step by step. "
                                "I'm a fast learner and very responsive to feedback.")
                else:
                    approach = ("I'd propose a standard hierarchical Bayesian model first, "
                                "then explore extensions based on the data characteristics.")

                return f"{concern}{approach}"
            else:
                style_note = ""
                if "async" in style and "sync-heavy" in system:
                    style_note = "One thing to flag — I work best async, but I see you prefer daily standups. We'd need to find a middle ground."
                elif "sync" in style:
                    style_note = "I enjoy pair programming and real-time collaboration, so the sync-heavy style works great for me."

                return (
                    f"{style_note} "
                    f"Overall I think this could be a productive collaboration. "
                    f"My communication style is {comm.lower()}, which I hope works for you."
                ).strip()

    def _mock_judge(self, messages: list) -> str:
        """Generate mock compatibility scores based on conversation content."""
        text = messages[0]["content"].lower() if messages else ""

        # Time & energy
        time_score = 0.7
        if "5h" in text or "5 hours" in text or "stretched thin" in text:
            time_score = 0.25
        elif "40h" in text or "fully dedicated" in text or "no competing" in text:
            time_score = 0.95
        elif "15h" in text or "freelancing" in text:
            time_score = 0.45
        elif "20h" in text:
            time_score = 0.7
        elif "25h" in text:
            time_score = 0.8

        # Priority alignment
        priority_score = 0.65
        if "neurips" in text and "publication" in text and "motivation" in text:
            priority_score = 0.85
        if "minimal time" in text or "other projects" in text or "guidance" in text:
            priority_score = 0.35
        if "eager to learn" in text or "dedicated" in text:
            priority_score = 0.8

        # Collab style
        style_score = 0.6
        if "async" in text and "sync-heavy" in text:
            style_score = 0.35
        elif "middle ground" in text:
            style_score = 0.5
        elif "pair programming" in text or "aligned" in text or "works great" in text:
            style_score = 0.85
        elif "structured weekly" in text:
            style_score = 0.7

        # Personality
        personality_score = 0.6
        if "eager" in text or "enthusiastic" in text or "fast learner" in text:
            personality_score = 0.85
        elif "transparent" in text and "stretched" in text:
            personality_score = 0.4
        elif "methodical" in text or "thorough" in text:
            personality_score = 0.7
        elif "creative" in text and "inconsistent" in text:
            personality_score = 0.5
        elif "pragmatic" in text or "ships fast" in text:
            personality_score = 0.75

        overall = round((time_score + priority_score + style_score + personality_score) / 4, 2)

        if overall >= 0.75:
            rec = "strong_match"
        elif overall >= 0.6:
            rec = "good_match"
        elif overall >= 0.45:
            rec = "risky_match"
        else:
            rec = "poor_match"

        # Dynamic risk/synergy based on scores
        weakest = min(
            [("time commitment", time_score), ("priority alignment", priority_score),
             ("collaboration style", style_score), ("personality fit", personality_score)],
            key=lambda x: x[1]
        )
        strongest = max(
            [("time commitment", time_score), ("priority alignment", priority_score),
             ("collaboration style", style_score), ("personality fit", personality_score)],
            key=lambda x: x[1]
        )

        risk_msgs = {
            "time commitment": "Insufficient time availability may cause delays and bottlenecks",
            "priority alignment": "Misaligned goals could lead to divergent effort allocation",
            "collaboration style": "Communication style mismatch may cause friction in daily work",
            "personality fit": "Personality differences could create tension under deadline pressure",
        }
        synergy_msgs = {
            "time commitment": "Strong time commitment ensures consistent progress",
            "priority alignment": "Shared publication goals create natural accountability",
            "collaboration style": "Compatible working styles enable smooth daily collaboration",
            "personality fit": "Complementary personalities balance speed and thoroughness",
        }

        return json.dumps({
            "time_energy": {"score": time_score, "reason": f"Based on stated {text.split('commit about')[1].split('.')[0].strip() if 'commit about' in text else 'availability'} availability"},
            "priority_alignment": {"score": priority_score, "reason": "Alignment between candidate goals and task objectives"},
            "collab_style": {"score": style_score, "reason": "Working style compatibility with requester preferences"},
            "personality_fit": {"score": personality_score, "reason": "Communication and personality compatibility assessment"},
            "overall_compatibility": overall,
            "top_risk": risk_msgs[weakest[0]],
            "top_synergy": synergy_msgs[strongest[0]],
            "recommendation": rec,
        })

    def simulate_conversation(
        self,
        requester: ExtendedProfile,
        candidate: ExtendedProfile,
        task: Task,
    ) -> dict:
        """
        Run a multi-turn agent-agent conversation, then judge compatibility.

        Returns dict with conversation transcript, scores, and recommendation.
        """

        sys_requester = build_agent_persona(requester, "requester", task)
        sys_candidate = build_agent_persona(candidate, "candidate", task)

        # Conversation history for each agent
        messages_req = []   # what the requester agent sees
        messages_cand = []  # what the candidate agent sees

        transcript = []

        # Requester opens
        opener = (
            f"Hi, I'm the agent for {requester.user_id}. We're looking for a collaborator "
            f"on: {task.goal}. I'd like to discuss how we might work together — "
            f"availability, expectations, and working style."
        )
        transcript.append({"role": f"agent_{requester.user_id}", "text": opener})
        messages_cand.append({"role": "user", "content": opener})

        # Multi-turn conversation
        for turn in range(self.n_turns):
            # Candidate responds
            cand_reply = self._call_llm(sys_candidate, messages_cand)
            transcript.append({"role": f"agent_{candidate.user_id}", "text": cand_reply})

            messages_cand.append({"role": "assistant", "content": cand_reply})
            messages_req.append({"role": "user", "content": cand_reply})

            # Requester responds (except on last turn)
            if turn < self.n_turns - 1:
                req_reply = self._call_llm(sys_requester, messages_req)
                transcript.append({"role": f"agent_{requester.user_id}", "text": req_reply})

                messages_req.append({"role": "assistant", "content": req_reply})
                messages_cand.append({"role": "user", "content": req_reply})

        # Judge the conversation
        conv_text = "\n\n".join(
            f"[{t['role']}]: {t['text']}" for t in transcript
        )

        judge_prompt = (
            f"Here is the negotiation transcript between agent_{requester.user_id} "
            f"(requester) and agent_{candidate.user_id} (candidate) for the task: "
            f"\"{task.goal}\".\n\n{conv_text}"
        )

        judge_result = self._call_llm(
            JUDGE_SYSTEM,
            [{"role": "user", "content": judge_prompt}],
        )

        # Parse judge output
        try:
            scores = json.loads(judge_result)
        except json.JSONDecodeError:
            cleaned = judge_result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            scores = json.loads(cleaned)

        return {
            "candidate_id": candidate.user_id,
            "transcript": transcript,
            "compatibility": scores,
        }


# ============================================================
# Layer 3: Full Planning Pipeline
# ============================================================

@dataclass
class RefinedCandidate:
    """Final output of Layer 3: candidate with both analytical and dream scores."""
    candidate_id: str
    analytical_score: float     # M from Layer 2
    S_cap: float
    S_need: float
    dream_score: float          # overall_compatibility from dream sim
    combined_score: float       # weighted combination
    compatibility: dict         # full compatibility breakdown
    top_risk: str
    top_synergy: str
    recommendation: str


class PlanningLayer:
    """
    Layer 3: Analytical coarse filter → LLM dream refinement → top-3 output.

    Stage 1 (MBPO-style short rollout):
      - Run Match Score on full candidate pool
      - Take top-K by UCB-enhanced M̃

    Stage 2 (World Models-style dreaming):
      - For each top-K candidate, simulate agent-agent conversation
      - Judge soft compatibility
      - Combine analytical + dream scores
      - Output top-3
    """

    def __init__(
        self,
        world_model: WorldModel,
        dream_simulator: DreamSimulator,
        top_k: int = 5,
        top_n: int = 3,
        analytical_weight: float = 0.5,
        dream_weight: float = 0.5,
    ):
        self.world_model = world_model
        self.dreamer = dream_simulator
        self.top_k = top_k
        self.top_n = top_n
        self.w_analytical = analytical_weight
        self.w_dream = dream_weight

    def run(
        self,
        requester: ExtendedProfile,
        candidates: List[ExtendedProfile],
        task: Task,
    ) -> List[RefinedCandidate]:
        """
        Full Layer 3 pipeline.
        """

        print("=" * 60)
        print("LAYER 3: PLANNING — COARSE FILTER + DREAM SIMULATION")
        print("=" * 60)

        # ---- Stage 1: Analytical coarse filter ----
        print(f"\n[Stage 1] Analytical ranking of {len(candidates)} candidates...")

        scored = []
        for cand in candidates:
            result = self.world_model.compute_match(
                requester.profile, cand.profile, task
            )
            scored.append((cand, result))

        scored.sort(key=lambda x: x[1].M, reverse=True)

        print(f"  {'Candidate':<12} {'S_cap':>8} {'S_need':>8} {'M':>8}")
        print(f"  {'-'*40}")
        for cand, result in scored:
            marker = " ←" if scored.index((cand, result)) < self.top_k else ""
            print(f"  {cand.user_id:<12} {result.S_cap:>8.4f} {result.S_need:>8.4f} "
                  f"{result.M:>8.4f}{marker}")

        top_k_candidates = scored[:self.top_k]
        print(f"\n  Top-{self.top_k} enter dream simulation.\n")

        # ---- Stage 2: Dream simulation ----
        print(f"[Stage 2] Dream simulation — agent-agent conversations...")
        print(f"  (This calls the LLM {self.top_k} times × ~{self.dreamer.n_turns * 2 + 1} turns)\n")

        refined = []
        for i, (cand, analytical_result) in enumerate(top_k_candidates):
            print(f"  --- Dream {i+1}/{len(top_k_candidates)}: "
                  f"{requester.user_id} ↔ {cand.user_id} ---")

            dream_result = self.dreamer.simulate_conversation(
                requester, cand, task
            )

            # Print conversation
            for turn in dream_result["transcript"]:
                label = turn["role"]
                text = turn["text"][:120] + ("..." if len(turn["text"]) > 120 else "")
                print(f"    [{label}]: {text}")

            compat = dream_result["compatibility"]
            dream_score = compat.get("overall_compatibility", 0.5)

            # Combined score
            combined = (self.w_analytical * analytical_result.M
                        + self.w_dream * dream_score)

            print(f"\n    Compatibility scores:")
            for dim in ["time_energy", "priority_alignment", "collab_style", "personality_fit"]:
                if dim in compat:
                    s = compat[dim]
                    print(f"      {dim:<22s}: {s['score']:.2f} — {s['reason']}")
            print(f"    Overall: {dream_score:.2f}  |  "
                  f"Risk: {compat.get('top_risk', 'N/A')}")
            print(f"    Synergy: {compat.get('top_synergy', 'N/A')}")
            print(f"    Recommendation: {compat.get('recommendation', 'N/A')}")
            print(f"    Combined score: {self.w_analytical:.1f}×{analytical_result.M:.3f} + "
                  f"{self.w_dream:.1f}×{dream_score:.3f} = {combined:.4f}")
            print()

            refined.append(RefinedCandidate(
                candidate_id=cand.user_id,
                analytical_score=analytical_result.M,
                S_cap=analytical_result.S_cap,
                S_need=analytical_result.S_need,
                dream_score=dream_score,
                combined_score=combined,
                compatibility=compat,
                top_risk=compat.get("top_risk", ""),
                top_synergy=compat.get("top_synergy", ""),
                recommendation=compat.get("recommendation", ""),
            ))

        # ---- Final ranking ----
        refined.sort(key=lambda x: x.combined_score, reverse=True)
        top_n = refined[:self.top_n]

        print("=" * 60)
        print(f"FINAL OUTPUT: Top-{self.top_n} Candidates")
        print("=" * 60)

        for rank, r in enumerate(top_n, 1):
            print(f"\n  #{rank} {r.candidate_id}")
            print(f"     Analytical M = {r.analytical_score:.4f} "
                  f"(S_cap={r.S_cap:.2f}, S_need={r.S_need:.2f})")
            print(f"     Dream score  = {r.dream_score:.2f}")
            print(f"     Combined     = {r.combined_score:.4f}")
            print(f"     Recommendation: {r.recommendation}")
            print(f"     Risk:    {r.top_risk}")
            print(f"     Synergy: {r.top_synergy}")

        return top_n
