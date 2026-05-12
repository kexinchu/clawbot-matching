"""Tiny HTTP server that powers Showcase_candidate.html.

Endpoints:
  GET  /                       → serves Showcase_candidate.html
  GET  /api/scenarios          → 3 demo scenarios with their top-3 dreaming candidates
  POST /api/feedback           → body: {scenario_id, candidate_id, action, stars}
                                 returns the computed reward + (optional)
                                 Bayesian update applied to the candidate.

The server keeps one OnlineLearning engine per scenario in memory so that
Bayesian updates persist across UI clicks within a session. Restart the
server to reset.

Run:
    python Online_learning/showcase_server.py
Then open http://localhost:8765/ in your browser.

Uses only the Python standard library — no extra dependencies required.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (_HERE, _REPO / "mapping-algo", _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from WorldModel import WorldModel  # noqa: E402
from Online_learning import OnlineLearning  # noqa: E402
from Reward_function import RewardFunction  # noqa: E402
from pipeline import match_one_to_one  # noqa: E402
from datatypes import MatchResult  # noqa: E402
from LLM_Dreaming.LLM_dreaming import DreamSimulator, PlanningLayer  # noqa: E402

from ui_adapter import (  # noqa: E402
    CFG,
    CandidateSpec,
    ScenarioSpec,
    action_to_feedback,
    build_demo_scenarios,
    scenario_to_backend,
)


logger = logging.getLogger("showcase_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ── Session state ──────────────────────────────────────────────────────

class ScenarioSession:
    """Holds the live state for one scenario across UI clicks."""

    def __init__(self, spec: ScenarioSpec, top_n: int = 3) -> None:
        self.spec = spec
        backend = scenario_to_backend(spec)
        self.requester = backend["requester"]
        self.task = backend["task"]
        self.candidate_states = backend["candidate_states"]
        self.candidate_by_id = {c.user_id: c for c in self.candidate_states}
        self.spec_by_id = {c.user_id: c for c in spec.candidates}
        self.world_model = WorldModel(config=CFG, theta_c=0.4, theta_n=-0.1)
        self.dream_simulator = DreamSimulator()
        self.planning_layer = PlanningLayer(
            world_model=self.world_model,
            dream_simulator=self.dream_simulator,
            top_k=max(6, len(self.candidate_states)),
            top_n=top_n,
        )
        self.engine = OnlineLearning(
            self.world_model,
            dream_simulator=self.dream_simulator,
            enable_dreaming=True,
            top_k=max(6, len(self.candidate_states)),
            top_n=top_n,
        )
        self.reward_fn = RewardFunction()
        self.lock = threading.Lock()
        self.history: List[Dict[str, Any]] = []
        self._top_cache: Optional[List[Dict[str, Any]]] = None

    # ---- top-3 candidates with dreaming ---------------------------

    def compute_top_candidates(self, force: bool = False) -> List[Dict[str, Any]]:
        with self.lock:
            if self._top_cache is not None and not force:
                return self._top_cache

            ranked: List[MatchResult] = match_one_to_one(
                self.requester,
                self.task,
                self.candidate_states,
                self.world_model.theta,
                self.world_model.config,
                top_k=len(self.candidate_states),
                use_ucb=True,
                round_t=1,
            )
            refined = self.planning_layer.run_from_layer3_output(
                requester=self.requester,
                task=self.task,
                candidate_profiles=self.candidate_states,
                layer3_output=ranked,
                verbose=False,
            )

            self._top_cache = [self._refined_to_json(r) for r in refined]
            return self._top_cache

    def _refined_to_json(self, refined: Any) -> Dict[str, Any]:
        spec = self.spec_by_id.get(refined.candidate_id)
        cap = refined.compatibility or {}

        def _safe_score(key: str, default: float = 0.5) -> float:
            v = cap.get(key)
            if isinstance(v, dict):
                return float(v.get("score", default))
            if isinstance(v, (int, float)):
                return float(v)
            return default

        soft_scores = {
            "time": _safe_score("time_energy"),
            "priority": _safe_score("priority_alignment"),
            "style": _safe_score("collab_style"),
            "personality": _safe_score("personality_fit"),
        }

        return {
            "candidate_id": refined.candidate_id,
            "name": spec.name if spec else refined.candidate_id,
            "initials": spec.initials if spec else refined.candidate_id[:2].upper(),
            "role": spec.role if spec else "",
            "avatar": spec.avatar if spec else "#6B6B6B",
            "combined": round(float(refined.combined_score), 4),
            "analytical_score": round(float(refined.analytical_score), 4),
            "dream_score": round(float(refined.dream_score), 4),
            "recommendation": refined.recommendation or "good_match",
            "capCoverage": int(round(float(refined.s_cap) * 100)),
            "needSatisfaction": int(round(float(refined.s_need) * 100)),
            "softScores": soft_scores,
            "strengths": spec.strengths if spec else [],
            "risks": spec.risks if spec else [],
            "stats": spec.stats if spec else {},
            "topSynergy": refined.top_synergy or "",
            "topRisk": refined.top_risk or "",
        }

    # ---- feedback → reward + Bayesian update ----------------------

    def apply_feedback(self, candidate_id: str, action: str,
                       stars: Optional[int]) -> Dict[str, Any]:
        with self.lock:
            if candidate_id not in self.candidate_by_id:
                raise KeyError(f"Unknown candidate '{candidate_id}'")

            feedback = action_to_feedback(action, stars)
            reward = self.reward_fn.compute(feedback)

            cand_state = self.candidate_by_id[candidate_id]
            bayes = self.engine.bayesian_updater.update(cand_state, self.task, reward.R)
            self.engine.selection_counts[candidate_id] += 1

            updated_caps = {
                cap.description: {"mu": round(cap.mu, 4), "sigma": round(cap.sigma, 4)}
                for cap in cand_state.capabilities
            }

            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "candidate_id": candidate_id,
                "action": action,
                "stars": stars,
                "feedback": feedback,
                "reward": {
                    "r_feedback": reward.r_feedback,
                    "r_efficiency": reward.r_efficiency,
                    "r_quality": reward.r_quality,
                    "R": reward.R,
                },
                "bayesian_updates": bayes,
                "capabilities_after_update": updated_caps,
            }
            self.history.append(entry)
            # Invalidate cached top-3 so the next /api/scenarios reflects
            # the updated belief.
            self._top_cache = None
            return entry


# ── HTTP server ────────────────────────────────────────────────────────

class ShowcaseHandler(BaseHTTPRequestHandler):
    server_version = "ShowcaseServer/0.1"
    sessions: Dict[str, ScenarioSession] = {}
    html_path = _REPO / "Showcase_candidate.html"

    # quieter access logs
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    # ----- helpers -----

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Allow cross-origin so the page also works when opened from file://
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path) -> None:
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self._send_json(404, {"error": "html not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc

    # ----- handlers -----

    def do_OPTIONS(self) -> None:  # noqa: N802 (HTTP verb spelling)
        self._send_json(204, {})

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path in ("/", "/index.html", "/Showcase_candidate.html"):
            self._send_html(self.html_path)
            return
        if url.path == "/api/scenarios":
            payload = []
            for sid, session in self.sessions.items():
                spec = session.spec
                payload.append({
                    "id": sid,
                    "label": spec.label,
                    "requester": {
                        "name": spec.requester_name,
                        "initials": spec.requester_initials,
                        "role": spec.requester_role,
                        "task": spec.requester_task,
                        "accent": spec.accent,
                    },
                    "candidates": session.compute_top_candidates(),
                    "history": session.history,
                })
            self._send_json(200, {"scenarios": payload})
            return
        self._send_json(404, {"error": "not found", "path": url.path})

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/api/feedback":
            self._send_json(404, {"error": "not found", "path": url.path})
            return
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        scenario_id = body.get("scenario_id")
        candidate_id = body.get("candidate_id")
        action = body.get("action")
        stars = body.get("stars")

        if scenario_id not in self.sessions:
            self._send_json(404, {"error": f"unknown scenario '{scenario_id}'"})
            return
        if action not in ("accepted", "skipped", "rejected"):
            self._send_json(400, {"error": f"invalid action '{action}'"})
            return
        try:
            entry = self.sessions[scenario_id].apply_feedback(candidate_id, action, stars)
        except (KeyError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, entry)


def build_handler_class(port: int) -> Tuple[type, Dict[str, ScenarioSession]]:
    sessions: Dict[str, ScenarioSession] = {}
    for spec in build_demo_scenarios():
        sessions[spec.scenario_id] = ScenarioSession(spec)
    # Pre-warm top candidates so the first /api/scenarios call is fast.
    for sid, sess in sessions.items():
        logger.info("Pre-computing top candidates for scenario '%s'...", sid)
        sess.compute_top_candidates()

    handler_cls = type("BoundShowcaseHandler", (ShowcaseHandler,), {
        "sessions": sessions,
    })
    return handler_cls, sessions


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    handler_cls, _sessions = build_handler_class(args.port)
    httpd = ThreadingHTTPServer((args.host, args.port), handler_cls)
    logger.info("Showcase server listening on http://%s:%d/", args.host, args.port)
    logger.info("Open http://%s:%d/ in your browser.", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
