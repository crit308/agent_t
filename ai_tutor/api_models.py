from pydantic import BaseModel
from typing import List, Optional, Union, Literal

from ai_tutor.agents.models import (
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis, QuizQuestion, QuizFeedbackItem
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

# --- Interaction Response Models ---

class ExplanationResponse(BaseModel):
    """Response containing a detailed explanation of a topic."""
    response_type: Literal["explanation"] = "explanation"
    text: str
    topic: str
    references: Optional[List[str]] = None

class QuestionResponse(BaseModel):
    """Response containing a quiz question."""
    response_type: Literal["question"] = "question"
    question: QuizQuestion
    topic: str
    context: Optional[str] = None

class FeedbackResponse(BaseModel):
    """Response containing feedback on a student's answer."""
    response_type: Literal["feedback"] = "feedback"
    feedback: QuizFeedbackItem
    topic: str
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None

class MessageResponse(BaseModel):
    """For general messages from the tutor."""
    response_type: Literal["message"] = "message"
    text: str
    message_type: Optional[str] = None  # e.g., "info", "success", "warning"

class ErrorResponse(BaseModel):
    """For error messages and exceptional cases."""
    response_type: Literal["error"] = "error"
    message: str
    error_code: Optional[str] = None
    details: Optional[dict] = None

TutorInteractionResponse = Union[
    ExplanationResponse,
    QuestionResponse, 
    FeedbackResponse,
    MessageResponse,
    ErrorResponse
]
"""Union type for all possible structured responses from the /interact endpoint.""" 