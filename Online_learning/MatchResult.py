from dataclasses import dataclass
from typing import Dict

@dataclass
class MatchResult:
    """Output of Layer 2 world model."""
    S_cap: float
    S_need: float
    M: float                # match score
    gap: Dict[str, float]   # per-dimension gap
