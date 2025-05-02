from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from uuid import UUID

class UserConceptMastery(BaseModel):
    """Tracks user's mastery of a specific concept using a Bayesian alpha/beta model."""
    alpha: int = 1
    beta: int = 1
    last_interaction_outcome: Optional[str] = None # e.g., 'correct', 'incorrect', 'asked_question'
    attempts: int = 0
    confusion_points: List[str] = Field(default_factory=list, description="Specific points user struggled with on this topic")
    last_accessed: Optional[str] = Field(None, description="ISO 8601 timestamp of when the concept was last accessed")

    @property
    def mastery(self) -> float:
        """Posterior mean mastery probability."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> int:
        """Total number of observations (alpha+beta)."""
        return self.alpha + self.beta

class UserModelState(BaseModel):
    """Represents the AI's understanding of the user's knowledge state and preferences."""
    concepts: Dict[str, UserConceptMastery] = Field(default_factory=dict)
    overall_progress: float = 0.0 # e.g., percentage of lesson plan covered
    current_topic: Optional[str] = None
    current_topic_segment_index: int = 0 # Tracks progress within the *current topic's* explanation
    learning_pace_factor: float = 1.0 # Controls pacing adjustment (e.g., >1 faster, <1 slower)
    preferred_interaction_style: Optional[Literal['explanatory', 'quiz_heavy', 'socratic']] = None # Can be set or inferred
    session_summary_notes: List[str] = Field(default_factory=list) # High-level notes about session progress/user behavior
    # Use string forward references for types from agents.models to avoid direct import here if possible
    current_section_objectives: List['LearningObjective'] = Field(default_factory=list, description="Learning objectives for the currently active section.") 
    mastered_objectives_current_section: List[str] = Field(default_factory=list, description="Titles of objectives mastered in the current section.")
    pending_interaction_type: Optional[Literal['checking_question', 'summary_prompt']] = None
    pending_interaction_details: Optional[Dict[str, Any]] = None # e.g., {'question_text': '...', 'interaction_id': 'xyz'} 