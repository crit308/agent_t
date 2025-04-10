from __future__ import annotations

import json
from uuid import UUID

# Store original default method from the JSONEncoder class
_original_json_encoder_default = json.JSONEncoder.default

def _custom_json_encoder_default(self, obj):
    """Custom default method for JSONEncoder to handle UUIDs."""
    if isinstance(obj, UUID):
        # If the object is a UUID, convert it to its string representation
        return str(obj)
    # For any other object type, fall back to the original default method
    # This ensures that standard JSON types and any other custom handling
    # registered elsewhere are still processed correctly.
    return _original_json_encoder_default(self, obj)

# Monkey-patch the default method of json.JSONEncoder
# Now, any part of the application using standard json.dumps
# (including libraries like httpx used by the tracer) will use this logic.
json.JSONEncoder.default = _custom_json_encoder_default

from pydantic import BaseModel, ConfigDict
from datetime import datetime # Keep datetime import for conversion if needed
from typing import Optional, List, Any, Dict, Literal, TYPE_CHECKING
from pydantic import Field

# Use TYPE_CHECKING to prevent runtime circular imports for type hints
if TYPE_CHECKING:
    from ai_tutor.agents.models import LessonPlan, QuizQuestion, LearningObjective, FocusObjective
    from ai_tutor.agents.analyzer_agent import AnalysisResult

class UserConceptMastery(BaseModel):
    """Tracks user's mastery of a specific concept."""
    mastery_level: float = 0.0 # e.g., 0-1 scale, assessed by quizzes/summaries
    # Add more detail on outcomes
    last_interaction_outcome: Optional[str] = None # e.g., 'correct', 'incorrect', 'asked_question'
    attempts: int = 0
    # Add tracking for specific struggles
    confusion_points: List[str] = Field(default_factory=list, description="Specific points user struggled with on this topic")
    # Change datetime to string to avoid schema validation issues with optional format
    last_accessed: Optional[str] = Field(None, description="ISO 8601 timestamp of when the concept was last accessed")

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
    # user_id: Optional[str] = None
    user_id: UUID # User ID from Supabase Auth, now mandatory UUID
    session_id: UUID # Use UUID
    folder_id: Optional[UUID] = None # Link to the folder ID
    vector_store_id: Optional[str] = None
    uploaded_file_paths: List[str] = Field(default_factory=list)
    analysis_result: Optional['AnalysisResult'] = None # Use forward reference
    knowledge_base_path: Optional[str] = None # Add path to KB file
    lesson_plan: Optional['LessonPlan'] = None # Use forward reference
    current_quiz_question: Optional['QuizQuestion'] = None # Use forward reference
    current_focus_objective: Optional['FocusObjective'] = None # NEW: Store the current focus from Planner
    user_model_state: UserModelState = Field(default_factory=UserModelState)
    last_interaction_summary: Optional[str] = None # What did the tutor just do? What did user respond?
    current_teaching_topic: Optional[str] = None # Which topic is the Teacher actively explaining?
    # Add other relevant session state as needed
    # e.g., current_lesson_progress: Optional[str] = None 