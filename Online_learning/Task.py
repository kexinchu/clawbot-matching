from dataclasses import dataclass
from typing import Dict

@dataclass
class Task:
    """Task representation: T = (goal, Q_T)."""
    task_id: str
    goal: str
    Q_T: Dict[str, float]   # domain → required skill level ∈ [0, 1]
