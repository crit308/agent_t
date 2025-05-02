from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union, Literal, Dict, Any
from uuid import UUID # For folder ID typing

# --- Import necessary models ---
# Import UserModelState from its new location
from ai_tutor.core_models import UserModelState # <--- IMPORT FROM core_models

# Import models likely needed by specific responses
from ai_tutor.agents.models import (
    LessonContent, QuizQuestion, QuizFeedbackItem, LearningObjective, FocusObjective
)
# Import AnalysisResult if needed by any response
from ai_tutor.agents.analyzer_agent import AnalysisResult

# --- Whiteboard / Canvas Types ---
# Define more specific types if known, otherwise use Dict for flexibility initially
CanvasObjectSpec = Dict[str, Any] # Example: { "type": "circle", "x": 100, "y": 100, "radius": 20, "fill": "red" }
WhiteboardAction = Dict[str, Any] # Example: { "action": "create", "object_id": "obj123", "spec": CanvasObjectSpec }

# --- Request Models (If any specific ones are needed for requests) ---
# Example:
class UserInteractionRequest(BaseModel):
    session_id: UUID
    user_input: str
    interaction_type: Literal["message", "answer", "request_clarification"] # etc.
    # Add other fields as needed

# --- Folder Models (Keep existing if used) ---
class FolderCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name of the new folder.")

class FolderUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="New name for the folder.")

class FolderResponse(BaseModel):
    id: UUID
    user_id: str # Assuming user_id is string from Supabase Auth
    name: str
    created_at: str # ISO 8601 string
    updated_at: str # ISO 8601 string
    deleted: bool = False # Soft delete flag

# --- Individual Interaction Response Data Models ---

class ExplanationResponse(BaseModel):
    """Data for when the tutor provides an explanation."""
    response_type: Literal["explanation"] = Field(default="explanation", description="The type of response this model represents.")
    explanation_text: str = Field(description="The main explanation content (Markdown supported).")
    explanation_title: Optional[str] = Field(None, description="Optional title for the explanation section.")
    # Example: Link explanation to specific learning objective(s)
    related_objectives: Optional[List[LearningObjective]] = Field(None, description="Learning objectives this explanation covers.")
    # Example: Include raw LessonContent if useful
    lesson_content: Optional[LessonContent] = Field(None, description="Raw LessonContent object if applicable.")
    whiteboard_actions: Optional[List[WhiteboardAction]] = Field(None, description="Optional list of actions for the whiteboard.")

class QuestionResponse(BaseModel):
    """Data for when the tutor asks the user a question (e.g., checking understanding)."""
    response_type: Literal["question"] = Field(default="question", description="The type of response this model represents.")
    question_type: Literal["multiple_choice", "free_response", "reflection"] = Field(description="Type of question being asked.")
    question_data: QuizQuestion = Field(description="The actual question object.") # Reuse existing QuizQuestion
    # Example: context for the question
    context_summary: Optional[str] = Field(None, description="Brief summary of the topic the question relates to.")

class FeedbackResponse(BaseModel):
    """Data for providing feedback on a user's answer or action."""
    feedback_items: List[QuizFeedbackItem] = Field(description="List of feedback items (reusing existing model).")
    overall_assessment: Optional[str] = Field(None, description="Overall summary of the feedback.")
    # Example: Suggest next steps based on feedback
    suggested_next_step: Optional[str] = Field(None, description="Recommendation for what the user should do next.")

class MessageResponse(BaseModel):
    """Generic message from the tutor (e.g., greetings, transition statements)."""
    message_text: str = Field(description="The text message from the tutor.")
    message_type: Literal["greeting", "transition", "summary", "status_update", "clarification"] = Field(default="status_update", description="The type or purpose of the message.")

class ErrorResponse(BaseModel):
    """Data for reporting an error to the frontend."""
    error_code: Optional[str] = Field(None, description="A unique code identifying the type of error.")
    error_message: str = Field(description="A user-friendly error message.")
    technical_details: Optional[str] = Field(None, description="Optional technical details (for logging/debugging on FE).")

class RawMessageResponse(BaseModel):
    """For sending arbitrary JSON data when other schemas don't fit (use sparingly!)."""
    raw_data: Dict[str, Any] = Field(description="The raw JSON data being sent.")

# --- Union of all possible response data types ---
InteractionResponseDataType = Union[
    ExplanationResponse,
    QuestionResponse,
    FeedbackResponse,
    MessageResponse,
    ErrorResponse,
    RawMessageResponse,
    Dict[str, Any] # Allow Dict as a fallback, but prefer specific types
]

# --- Interaction Response Data Wrapper (Updated) ---
class InteractionResponseData(BaseModel):
    """Wrapper for all data sent FROM the backend /interact endpoint or WebSocket."""
    model_config = ConfigDict(
        json_encoders={
            UUID: str # Ensure UUIDs are serialized as strings
        }
    )
    schema_version: int = Field(default=1, description="Version number for this response schema.")
    content_type: Literal[
        "explanation",
        "question",
        "feedback",
        "message",
        "error",
        "raw"
    ] = Field(description="Indicates the type of data contained in the 'data' field.")
    data: InteractionResponseDataType = Field(description="The actual payload, matching the content_type.")
    user_model_state: UserModelState = Field(description="The latest snapshot of the user's model state.")
    whiteboard_actions: Optional[List[WhiteboardAction]] = Field(None, description="Optional list of actions for the whiteboard.")

# --- Other API Models (Keep if relevant) ---

# Example: Request to start a new session
class SessionStartRequest(BaseModel):
    user_id: str # Or UUID if preferred
    folder_id: Optional[UUID] = None
    session_goal: Optional[str] = None
    # Maybe add initial user preferences here

# Example: Response confirming session start
class SessionStartResponse(BaseModel):
    session_id: UUID
    message: str = "Session started successfully."

# Example: Response for /sessions creation endpoint (alias of SessionStartResponse for clarity)
class SessionResponse(BaseModel):
    session_id: UUID

# Ensure other models like AnalysisResult, LessonPlan etc. are correctly defined/imported
# if they are used within the response types above.

# --- Folder List Response ---
class FolderListResponse(BaseModel):
    folders: List[FolderResponse]

# --- Document Upload Response ---
class DocumentUploadResponse(BaseModel):
    vector_store_id: Optional[str] = None
    files_received: List[str]
    analysis_status: str
    message: Optional[str] = None

# --- Analysis Response ---
class AnalysisResponse(BaseModel):
    status: str
    analysis: Optional[Any] = None
    error: Optional[str] = None

# --- Interaction Request ---
class InteractionRequestData(BaseModel):
    """Data sent from the client to interact with the tutor."""
    user_input: str
    additional_context: Optional[Dict[str, Any]] = None

# --- Legacy Tutor Interaction Response (deprecated in favour of InteractionResponseData) ---
class TutorInteractionResponse(BaseModel):
    content_type: str
    data: Any
    user_model_state: Optional[UserModelState] = None