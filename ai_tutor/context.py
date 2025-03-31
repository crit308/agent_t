from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, List, Any, Dict, Literal, TYPE_CHECKING
from pydantic import Field

# Use TYPE_CHECKING to prevent runtime circular imports for type hints
if TYPE_CHECKING:
    from ai_tutor.agents.models import LessonPlan, QuizQuestion
    from ai_tutor.agents.analyzer_agent import AnalysisResult

class UserConceptMastery(BaseModel):
    """Tracks user's mastery of a specific concept."""
    mastery_level: float = 0.0 # e.g., 0-1 scale, assessed by quizzes/summaries
    last_interaction_outcome: Optional[str] = None # e.g., 'correct', 'incorrect', 'asked_question'
    attempts: int = 0

class UserModelState(BaseModel):
    """Represents the AI's understanding of the user's knowledge state."""
    concepts: Dict[str, UserConceptMastery] = Field(default_factory=dict)
    overall_progress: float = 0.0 # e.g., percentage of lesson plan covered
    current_topic: Optional[str] = None
    # Future: Add learning style, confusion points, etc.

class TutorContext(BaseModel):
    """Context object for an AI Tutor session."""
    user_id: Optional[str] = None
    session_id: str
    vector_store_id: Optional[str] = None
    uploaded_file_paths: List[str] = Field(default_factory=list)
    analysis_result: Optional['AnalysisResult'] = None # Use forward reference
    lesson_plan: Optional['LessonPlan'] = None # Use forward reference
    current_quiz_question: Optional['QuizQuestion'] = None # Use forward reference
    user_model_state: UserModelState = Field(default_factory=UserModelState)
    last_interaction_summary: Optional[str] = None # What did the tutor just do? What did user respond?
    # Add other relevant session state as needed
    # e.g., current_lesson_progress: Optional[str] = None 