from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Union

# --- User Model State --- #
class UserModelState(BaseModel):
    """Represents the tutor's model of the user's knowledge, preferences, etc."""
    # Example fields - customize as needed
    knowledge_level: float = Field(default=0.5, description="Estimated knowledge level (0-1)")
    preferences: Dict[str, Any] = Field(default={}, description="User learning preferences")
    history: list = Field(default=[], description="History of interactions or concepts covered")

# --- Specific Content Data Payloads --- #
class ExplanationResponse(BaseModel):
    explanation_text: str

class DialogueResponse(BaseModel):
    dialogue_text: str

class QuestionResponse(BaseModel):
    question_text: str
    question_type: Literal["multiple_choice", "short_answer", "coding_challenge"] # Example types
    options: list[str] | None = None # For multiple choice

class FeedbackResponse(BaseModel):
    feedback_text: str
    is_correct: bool | None = None # Optional correctness flag

# --- Union of Possible Data Payloads --- #
# This allows the 'data' field to hold different structures based on 'content_type'
ResponseType = Union[
    ExplanationResponse, 
    DialogueResponse, 
    QuestionResponse, 
    FeedbackResponse
]

# --- Main Interaction Data Structures --- #

class InteractionRequest(BaseModel):
    """Data received from the frontend via WebSocket."""
    session_id: str
    user_input_text: str | None = None
    # Add other relevant fields from FE if needed (e.g., button clicks)

class InteractionResponseData(BaseModel):
    """The structured response sent back to the frontend via WebSocket."""
    content_type: Literal["explanation", "dialogue", "question", "feedback", "error"] # Extend as needed
    data: ResponseType # The actual content, structure depends on content_type
    user_model_state: UserModelState # Always include the latest user model state
    # NOTE: No top-level 'status' field here. Status is inferred by FE/WS handler based on content_type. 