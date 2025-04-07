from __future__ import annotations
from pydantic import BaseModel
from datetime import datetime # Keep datetime import for conversion if needed
from typing import Optional, List, Any, Dict, Literal, TYPE_CHECKING
from pydantic import Field
from uuid import UUID # Import UUID

# Use TYPE_CHECKING to prevent runtime circular imports for type hints
if TYPE_CHECKING:
    from ai_tutor.agents.models import LessonPlan, QuizQuestion
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
    pending_interaction_type: Optional[Literal['checking_question', 'summary_prompt']] = None
    pending_interaction_details: Optional[Dict[str, Any]] = None # e.g., {'question_text': '...', 'interaction_id': 'xyz'}

class TutorContext(BaseModel):
    """Context object for an AI Tutor session."""
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
    user_model_state: UserModelState = Field(default_factory=UserModelState)
    last_interaction_summary: Optional[str] = None # What did the tutor just do? What did user respond?
    current_teaching_topic: Optional[str] = None # Which topic is the Teacher actively explaining?
    # Add other relevant session state as needed
    # e.g., current_lesson_progress: Optional[str] = None 