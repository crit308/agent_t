from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
import json
import time
import uuid
from pathlib import Path
from supabase import Client, PostgrestAPIResponse
from fastapi import HTTPException # For raising errors

from ai_tutor.context import TutorContext # Import the main context model

# Models might still be needed if SessionManager directly interacts with them.
from ai_tutor.agents.models import LessonPlan, LessonContent, Quiz, QuizFeedback, SessionAnalysis
from ai_tutor.agents.analyzer_agent import AnalysisResult # Import AnalysisResult from its correct location

if TYPE_CHECKING:
    from supabase import Client

# In-memory storage for session data.
# WARNING: This will lose state on server restart and doesn't scale horizontally.
# Consider using Redis or a database for production.
_sessions: Dict[str, Dict[str, Any]] = {}

class SessionManager:
    """Manages session state for AI Tutor sessions."""

    def __init__(self):
        """Initialize the session manager."""
        pass

    async def create_session(self, supabase: Client, user_id: str) -> str:
        """Creates a new session in Supabase DB and returns its ID."""
        session_id = str(uuid.uuid4())
        context = TutorContext(session_id=session_id, user_id=user_id) # Include user_id
        context_dict = context.model_dump(mode='json')

        try:
            response: PostgrestAPIResponse = supabase.table("sessions").insert({
                "id": session_id,
                "user_id": user_id,
                "context_data": context_dict,
                # created_at and updated_at likely handled by DB defaults
            }).execute()

            if response.data:
                print(f"Created session {session_id} for user {user_id} in Supabase.")
                return session_id
            else:
                print(f"Error creating session {session_id} in Supabase: {response.error}")
                raise HTTPException(status_code=500, detail=f"Failed to create session in database: {response.error.message if response.error else 'Unknown error'}")
        except Exception as e:
            print(f"Exception creating session {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error during session creation: {e}")

    async def get_session_context(self, supabase: Client, session_id: str, user_id: str) -> Optional[TutorContext]:
        """Retrieves the TutorContext for a given session ID and user ID from Supabase."""
        try:
            response: PostgrestAPIResponse = supabase.table("sessions").select("context_data").eq("id", session_id).eq("user_id", user_id).maybe_single().execute()

            if response.data and response.data.get("context_data"):
                # Parse the JSONB data back into TutorContext
                return TutorContext(**response.data["context_data"])
            elif response.data is None: # No matching session found for user
                return None
            else: # Data exists but context_data might be missing/null
                 print(f"Warning: Session {session_id} found but context_data is missing or null.")
                 return None
        except Exception as e:
            print(f"Error fetching session {session_id} context from Supabase: {e}")
            raise HTTPException(status_code=500, detail=f"Database error fetching session context: {e}")

    async def update_session_context(self, supabase: Client, session_id: str, user_id: str, context: TutorContext) -> bool:
        """Updates the TutorContext for a given session ID in Supabase."""
        context_dict = context.model_dump(mode='json')
        try:
            response: PostgrestAPIResponse = supabase.table("sessions").update({"context_data": context_dict}).eq("id", session_id).eq("user_id", user_id).execute()
            # For simplicity, just checking for errors.
            # We removed the check for response.error as it caused an AttributeError
            # Now relying on the execute() call to raise an exception on failure.
            print(f"Updated session {session_id} context successfully.")
            return True
        except Exception as e:
            # Check if the exception is directly from Supabase and has details
            # This depends on the exception types raised by supabase-py
            # Example check (might need adjustment):
            # if hasattr(e, 'details'):
            #     print(f"Supabase API Error updating session {session_id}: {e.details}")
            # else:
            print(f"Exception updating session {session_id} context: {e}")
            raise HTTPException(status_code=500, detail=f"Database error updating session context: {e}")

    # session_exists might not be needed if get_session handles the lookup failure

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