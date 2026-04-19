from dataclasses import dataclass
from typing import Dict
from Capability import Capability

@dataclass
class UserProfile:
    """Structured user state: s_v = (Cap_v, Need_v)."""
    user_id: str
    capability: Dict[str, Capability]   # domain → (μ, σ)
    need: Dict[str, float]              # domain → need intensity ∈ [0, 1]
