from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal, Dict, Any
from uuid import UUID # For folder ID typing

from ai_tutor.agents.models import (
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis, QuizQuestion, QuizFeedbackItem
)
from ai_tutor.agents.analyzer_agent import AnalysisResult # Import the specific model
from ai_tutor.context import UserModelState # Import UserModelState

# --- Request Models ---
# (QuizUserAnswers is already defined and can be reused)

# --- Folder Models ---
class FolderCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name of the new folder.")

class FolderResponse(BaseModel):
    id: UUID
    name: str
    created_at: str # ISO 8601 timestamp

# --- Interaction Models ---
class InteractionRequestData(BaseModel):
    """Model for data sent TO the /interact endpoint."""
    type: Literal['start', 'next', 'answer', 'question', 'summary', 'previous']
    data: Optional[Dict[str, Any]] = None # e.g., {"answer_index": 1} or {"question_text": "..."}

# --- Response Models ---

# --- Folder List Response ---
class FolderListResponse(BaseModel):
    folders: List[FolderResponse]

class SessionResponse(BaseModel):
    session_id: UUID # Use UUID type

class DocumentUploadResponse(BaseModel):
    # vector_store_id: Optional[str] # Remove field as it's no longer used
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
    response_type: Literal["explanation"]
    text: str
    topic: str
    segment_index: int # Make segment_index required
    is_last_segment: bool # Indicate if more segments follow
    references: Optional[List[str]] = None

class QuestionResponse(BaseModel):
    """Response containing a quiz question."""
    response_type: Literal["question"]
    question: QuizQuestion
    topic: str
    context: Optional[str] = None

class FeedbackResponse(BaseModel):
    """Response containing feedback on a student's answer."""
    response_type: Literal["feedback"]
    feedback: QuizFeedbackItem
    topic: str
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None

class MessageResponse(BaseModel):
    """For general messages from the tutor."""
    response_type: Literal["message"]
    text: str
    message_type: Optional[str] = None  # e.g., "info", "success", "warning"

class ErrorResponse(BaseModel):
    """For error messages and exceptional cases."""
    response_type: Literal["error"]
    message: str
    error_code: Optional[str] = None
    details: Optional[dict] = None

# --- Interaction Response Data Wrapper ---
class InteractionResponseData(BaseModel):
    """Wrapper for data sent FROM the /interact endpoint."""
    content_type: str # Matches InteractionContentType in frontend (e.g., 'explanation', 'question')
    data: Any # The actual content (ExplanationResponse, QuestionResponse, etc.)
    user_model_state: UserModelState # Include the updated user model state

TutorInteractionResponse = Union[
    ExplanationResponse,
    QuestionResponse, 
    FeedbackResponse,
    MessageResponse,
    ErrorResponse
]
"""Union type for all possible structured responses from the /interact endpoint.""" 