from __future__ import annotations
from typing import Dict, Any, Optional
import json
import time
import uuid
from pathlib import Path

from ai_tutor.context import TutorContext # Import the main context model

# Models might still be needed if SessionManager directly interacts with them.
from ai_tutor.agents.models import LessonPlan, LessonContent, Quiz, QuizFeedback, SessionAnalysis
from ai_tutor.agents.analyzer_agent import AnalysisResult # Import AnalysisResult from its correct location

# In-memory storage for session data.
# WARNING: This will lose state on server restart and doesn't scale horizontally.
# Consider using Redis or a database for production.
_sessions: Dict[str, Dict[str, Any]] = {}

class SessionManager:
    """Manages session state for AI Tutor sessions."""

    def __init__(self):
        """Initialize the session manager."""
        pass

    def create_session(self) -> str:
        """Creates a new session and returns its ID."""
        session_id = str(uuid.uuid4())
        # Initialize session with a TutorContext object's dict representation
        # Agents SDK will receive the TutorContext object itself via the runner.
        context = TutorContext(session_id=session_id)
        _sessions[session_id] = context.model_dump(mode='json') # Store as JSON-compatible dict
        # Add internal start time tracking if needed, separate from context sent to agents
        _sessions[session_id]["_internal_start_time"] = time.time()
        # Keep track of raw agent outputs if needed for later analysis separately
        _sessions[session_id]["_raw_agent_outputs"] = {}
        print(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the state for a given session ID."""
        # Returns the raw dict. The API layer will parse this back into TutorContext.
        return _sessions.get(session_id)

    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Updates the state for a given session ID.
           Expects `data` to contain fields matching TutorContext or internal fields.
        """
        if session_id in _sessions:
            _sessions[session_id].update(data)
            return True
        return False

    def session_exists(self, session_id: str) -> bool:
        """Checks if a session exists."""
        return session_id in _sessions 