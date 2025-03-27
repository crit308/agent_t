from pydantic import BaseModel
from typing import Optional, List

class TutorContext(BaseModel):
    """Context object for an AI Tutor session."""
    user_id: Optional[str] = None
    session_id: str
    vector_store_id: Optional[str] = None
    uploaded_file_paths: List[str] = []
    # Add other relevant session state as needed
    # e.g., current_lesson_progress: Optional[str] = None 