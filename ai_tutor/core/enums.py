"""
ExecutorStatus enum indicating outcome of executing an objective.
"""
from enum import Enum

class ExecutorStatus(Enum):
    """Status codes returned by ExecutorAgent.run"""
    COMPLETED = "completed"
    CONTINUE  = "continue"  # new non‑terminal step
    STUCK     = "stuck" 