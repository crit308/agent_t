"""
Schema definitions for PlannerAgent outputs.
"""
from pydantic import BaseModel
from typing import List

class Objective(BaseModel):
    topic: str
    learning_goal: str
    target_mastery: float
    priority: int = 5

class PlannerOutput(BaseModel):
    objectives: List[Objective] 