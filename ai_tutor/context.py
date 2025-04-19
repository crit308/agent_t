from __future__ import annotations

import json
from uuid import UUID
from typing import Optional, List, Any, Dict, Literal, TYPE_CHECKING, Union
from dataclasses import is_dataclass, asdict

# Simplified JSONEncoder default: always return serializable types
def _custom_json_encoder_default(self, obj):
    """Custom default method for JSONEncoder to handle various non-serializable types."""
    # Handle UUIDs
    if isinstance(obj, UUID):
        return str(obj)
    # Handle Pydantic models
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Handle dataclasses
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    # Fallback: use string representation for anything else
    return str(obj)

# Monkey-patch the default method of json.JSONEncoder
# Now, any part of the application using standard json.dumps
# (including libraries like httpx used by the tracer) will use this logic.
json.JSONEncoder.default = _custom_json_encoder_default

from pydantic import BaseModel, ConfigDict
from datetime import datetime # Keep datetime import for conversion if needed
from pydantic import Field

# Use TYPE_CHECKING to prevent runtime circular imports for type hints
if TYPE_CHECKING:
    from ai_tutor.agents.models import LessonPlan, QuizQuestion, LearningObjective, FocusObjective
    from ai_tutor.agents.analyzer_agent import AnalysisResult

class UserConceptMastery(BaseModel):
    """Tracks user's mastery of a specific concept using a Bayesian alpha/beta model."""
    alpha: int = 1
    beta: int = 1
    # Add more detail on outcomes
    last_interaction_outcome: Optional[str] = None # e.g., 'correct', 'incorrect', 'asked_question'
    attempts: int = 0
    # Add tracking for specific struggles
    confusion_points: List[str] = Field(default_factory=list, description="Specific points user struggled with on this topic")
    # Change datetime to string to avoid schema validation issues with optional format
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
    # Add fields for personalization
    learning_pace_factor: float = 1.0 # Controls pacing adjustment (e.g., >1 faster, <1 slower)
    preferred_interaction_style: Optional[Literal['explanatory', 'quiz_heavy', 'socratic']] = None # Can be set or inferred
    session_summary_notes: List[str] = Field(default_factory=list) # High-level notes about session progress/user behavior
    # Add fields for tracking interaction state
    current_section_objectives: List['LearningObjective'] = Field(default_factory=list, description="Learning objectives for the currently active section.") # Use forward reference
    mastered_objectives_current_section: List[str] = Field(default_factory=list, description="Titles of objectives mastered in the current section.")
    pending_interaction_type: Optional[Literal['checking_question', 'summary_prompt']] = None
    pending_interaction_details: Optional[Dict[str, Any]] = None # e.g., {'question_text': '...', 'interaction_id': 'xyz'}

class TutorContext(BaseModel):
    """Context object for an AI Tutor session."""
    # Add model_config to specify custom JSON encoders
    model_config = ConfigDict(
        json_encoders={
            UUID: str  # Tell Pydantic to serialize UUID objects as strings
        }
    )
    # user_id: User ID from Supabase Auth, can be string or UUID
    user_id: Union[str, UUID]
    session_id: UUID # Use UUID
    folder_id: Optional[UUID] = None # Link to the folder ID
    vector_store_id: Optional[str] = None
    session_goal: Optional[str] = None  # Add session_goal to store high-level session objective
    uploaded_file_paths: List[str] = Field(default_factory=list)
    analysis_result: Optional['AnalysisResult'] = None # Use forward reference
    knowledge_base_path: Optional[str] = None # Add path to KB file
    lesson_plan: Optional['LessonPlan'] = None # Use forward reference
    current_quiz_question: Optional['QuizQuestion'] = None # Use forward reference
    current_focus_objective: Optional['FocusObjective'] = None # NEW: Store the current focus from Planner
    user_model_state: UserModelState = Field(default_factory=UserModelState)
    last_interaction_summary: Optional[str] = None # What did the tutor just do? What did user respond?
    current_teaching_topic: Optional[str] = None # Which topic is the Teacher actively explaining?
    # Add for session resume:
    last_event: Optional[dict] = None # Store the last event for session resume
    pending_interaction_type: Optional[str] = None # Store pending interaction type for resume
    # Add other relevant session state as needed
    # e.g., current_lesson_progress: Optional[str] = None 

# Utility helper to check if a concept is mastered
def is_mastered(c: UserConceptMastery) -> bool:
    """Return true if mastery probability > 0.8 and confidence (observations) >= 5."""
    return c.mastery > 0.8 and c.confidence >= 5 