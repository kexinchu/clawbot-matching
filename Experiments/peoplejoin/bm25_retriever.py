"""BM25 retrieval over public candidate cards only."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from Experiments.peoplejoin.isolation import assert_peoplejoin_public_inputs

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class DirectoryEntry:
    """Public directory row built from a candidate_card (+ role)."""

    candidate_id: str
    role: str = ""
    summary: str = ""
    highlighted_strengths: List[str] = field(default_factory=list)
    highlighted_risks: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "role": self.role,
            "summary": self.summary,
            "highlighted_strengths": list(self.highlighted_strengths),
            "highlighted_risks": list(self.highlighted_risks),
        }

    def index_text(self) -> str:
        parts = [
            self.candidate_id,
            self.role,
            self.summary,
            " ".join(self.highlighted_strengths),
            " ".join(self.highlighted_risks),
            self.explanation,
        ]
        return " ".join(p for p in parts if p)


@dataclass
class SearchHit:
    candidate_id: str
    score: float
    rank: int
    summary: str
    strengths: List[str]
    risks: List[str]
    role: str = ""

    def to_observation(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "score": round(float(self.score), 6),
            "role": self.role,
            "summary": self.summary,
            "highlighted_strengths": list(self.strengths),
            "highlighted_risks": list(self.risks),
        }


class BM25CandidateRetriever:
    """Okapi BM25 over public candidate directory cards."""

    def __init__(
        self,
        entries: Sequence[DirectoryEntry],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.entries = list(entries)
        self.k1 = float(k1)
        self.b = float(b)
        self._docs_tokens: List[List[str]] = [tokenize(e.index_text()) for e in self.entries]
        self._doc_len = [len(toks) for toks in self._docs_tokens]
        self._avgdl = (
            sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        )
        self._df: Counter = Counter()
        for toks in self._docs_tokens:
            self._df.update(set(toks))
        self._N = len(self._docs_tokens)
        self.search_log: List[Dict[str, Any]] = []

        # Isolation: directory must never contain hidden fields.
        assert_peoplejoin_public_inputs([e.to_public_dict() for e in self.entries])

    @classmethod
    def from_task_entry(cls, task_entry: dict) -> "BM25CandidateRetriever":
        entries: List[DirectoryEntry] = []
        for cand in task_entry.get("candidates", []):
            card = cand.get("candidate_card", {}) or {}
            profile = cand.get("candidate_profile", {}) or {}
            entries.append(
                DirectoryEntry(
                    candidate_id=str(
                        card.get("candidate_id")
                        or profile.get("user_id")
                        or ""
                    ),
                    role=str(profile.get("role", "")),
                    summary=str(card.get("summary", "")),
                    highlighted_strengths=list(card.get("highlighted_strengths") or []),
                    highlighted_risks=list(card.get("highlighted_risks") or []),
                    explanation=str(card.get("explanation", "")),
                )
            )
        return cls(entries)

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # Standard BM25+ style idf with +1 smoothing.
        return math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> List[tuple[str, float]]:
        q_tokens = tokenize(query)
        if not q_tokens or self._N == 0:
            return [(e.candidate_id, 0.0) for e in self.entries]

        scores: List[tuple[str, float]] = []
        for idx, (entry, doc_tokens, dl) in enumerate(
            zip(self.entries, self._docs_tokens, self._doc_len)
        ):
            tf = Counter(doc_tokens)
            s = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self._idf(term)
                freq = tf[term]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                s += idf * (freq * (self.k1 + 1.0)) / denom
            scores.append((entry.candidate_id, float(s)))
        scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scores

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        assert_peoplejoin_public_inputs({"query": query})
        ranked = self.score(query)
        by_id = {e.candidate_id: e for e in self.entries}
        hits: List[SearchHit] = []
        for rank, (cid, sc) in enumerate(ranked[: max(0, int(top_k))], start=1):
            entry = by_id[cid]
            hits.append(
                SearchHit(
                    candidate_id=cid,
                    score=sc,
                    rank=rank,
                    summary=entry.summary,
                    strengths=list(entry.highlighted_strengths),
                    risks=list(entry.highlighted_risks),
                    role=entry.role,
                )
            )
        record = {
            "query": query,
            "top_k": int(top_k),
            "results": [h.to_observation() for h in hits],
        }
        assert_peoplejoin_public_inputs(record)
        self.search_log.append(record)
        return hits

    def directory_overview(self, max_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = [e.to_public_dict() for e in self.entries]
        if max_entries is not None:
            rows = rows[: max_entries]
        assert_peoplejoin_public_inputs(rows)
        return rows
