from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

class ToolCall(BaseModel):
    """Unified envelope produced by the lean Executor LLM each turn."""
    name: Literal[
        "ask_question",  # Renders MCQ/input UI
        "explain",       # Renders text explanation
        "draw",          # Renders SVG on whiteboard
        "reflect",       # Internal thought process, no UI change
        "summarise_context",  # Internal history management
        "end_session",   # Signals session completion
    ] = Field(..., description="Tool/function the tutor wants to invoke")
    args: Dict[str, Any] = Field(default_factory=dict, description="JSON args for the tool") 