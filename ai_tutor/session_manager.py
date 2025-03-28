import uuid
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import os
import json
import time

from ai_tutor.agents.models import (
    LessonPlan, LessonContent, Quiz, QuizFeedback, SessionAnalysis
)
from ai_tutor.agents.analyzer_agent import DocumentAnalysis # Assuming DocumentAnalysis is the output type

# In-memory storage for session data.
# WARNING: This will lose state on server restart and doesn't scale horizontally.
# Consider using Redis or a database for production.
_sessions: Dict[str, Dict[str, Any]] = {}

class SessionManager:
    """Manages AI Tutor sessions (in-memory implementation)."""

    def create_session(self) -> str:
        """Creates a new session and returns its ID."""
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "session_id": session_id,
            "start_time": time.time(),
            "vector_store_id": None,
            "uploaded_files": [],
            "document_analysis": None, # Store the AnalysisResult object or its text
            "knowledge_base_path": None, # Path to the generated Knowledge Base file for this session
            "lesson_plan": None,
            "lesson_content": None,
            "quiz": None,
            "quiz_feedback": None,
            "session_analysis": None,
            "raw_agent_outputs": {}, # To store raw outputs if needed for session analysis
        }
        print(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the state for a given session ID."""
        return _sessions.get(session_id)

    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Updates the state for a given session ID."""
        if session_id in _sessions:
            _sessions[session_id].update(data)
            return True
        return False

    def session_exists(self, session_id: str) -> bool:
        """Checks if a session exists."""
        return session_id in _sessions 