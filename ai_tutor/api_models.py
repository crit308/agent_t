from pydantic import BaseModel
from typing import List, Optional, Union

from ai_tutor.agents.models import (
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis
)
from ai_tutor.agents.analyzer_agent import AnalysisResult # Import the specific model

# --- Request Models ---
# (QuizUserAnswers is already defined and can be reused)

# --- Response Models ---

class SessionResponse(BaseModel):
    session_id: str

class DocumentUploadResponse(BaseModel):
    vector_store_id: Optional[str]
    files_received: List[str]
    analysis_status: str # e.g., "completed", "failed", "pending" (if async)
    message: str

class AnalysisResponse(BaseModel):
    status: str # "pending", "completed", "failed", "not_started"
    analysis: Optional[AnalysisResult] = None # Use the specific model here
    error: Optional[str] = None 