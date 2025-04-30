from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union, Literal, Dict, Any
from uuid import UUID # For folder ID typing

from ai_tutor.agents.models import (
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis, QuizQuestion, QuizFeedbackItem
)
from ai_tutor.agents.analyzer_agent import AnalysisResult # Import the specific model
from ai_tutor.context import UserModelState # Import UserModelState

# Placeholder type for backend model
WhiteboardAction = Dict[str, Any]

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
    vector_store_id: Optional[str]
    files_received: List[str]
    analysis_status: str # e.g., "completed", "failed", "pending" (if async)
    message: str
    job_id: Optional[str] = None  # Add job_id for async embedding jobs

class AnalysisResponse(BaseModel):
    status: str # "pending", "completed", "failed", "not_started"
    analysis: Optional[AnalysisResult] = None # Use the specific model here
    error: Optional[str] = None

# --- Interaction Response Models ---

class ExplanationResponse(BaseModel):
    """Response containing a detailed explanation segment."""
    response_type: Literal["explanation"] = "explanation"
    text: str
    topic: Optional[str] = None
    segment_index: Optional[int] = None
    is_last_segment: Optional[bool] = None

class QuestionResponse(BaseModel):
    """Response containing a quiz question to present to the user."""
    response_type: Literal["question"] = "question"
    question: QuizQuestion
    topic: Optional[str] = None

class FeedbackResponse(BaseModel):
    """Response containing feedback on a student's answer."""
    response_type: Literal["feedback"] = "feedback"
    item: QuizFeedbackItem  # Changed 'feedback' to 'item' and removed redundant fields
    # topic: Optional[str] = None # Removed
    # correct_answer: Optional[str] = None # Removed
    # explanation: Optional[str] = None # Removed
    # model_config = {"extra": "forbid"} # Can be kept or removed based on strictness needs

class MessageResponse(BaseModel):
    """For general messages from the tutor."""
    response_type: Literal["message"] = "message"
    text: str
    message_type: Optional[str] = None  # e.g., "info", "success", "warning"
    model_config = {"extra": "forbid"}

class ErrorResponse(BaseModel):
    """For error messages and exceptional cases."""
    response_type: Literal["error"] = "error"
    message: str
    error_code: Optional[str] = None
    details: Optional[dict] = None
    model_config = {"extra": "forbid"}

# --- Interaction Response Data Wrapper ---
class InteractionResponseData(BaseModel):
    """Wrapper for data sent FROM the /interact endpoint."""
    model_config = ConfigDict(
        json_encoders={
            UUID: str
        }
    )
    content_type: str # Matches InteractionContentType in frontend (e.g., 'explanation', 'question')
    data: Union[ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse, Dict[str, Any]] # Explicitly allow Dict for fallbacks
    user_model_state: UserModelState # Include the updated user model state
    whiteboard_actions: Optional[List[WhiteboardAction]] = Field(None, description="Optional list of actions for the whiteboard.")

TutorInteractionResponse = Union[
    ExplanationResponse,
    QuestionResponse,
    FeedbackResponse,
    MessageResponse,
    ErrorResponse
]
"""Union type for all possible structured responses from the /interact endpoint.""" 