from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING
import json
import re # Import re for parsing KB
import time
import uuid
from pathlib import Path
from supabase import Client, PostgrestAPIResponse
from fastapi import HTTPException # For raising errors
from uuid import UUID # Import UUID

from ai_tutor.context import TutorContext # Import the main context model

# Needed for context creation
from ai_tutor.context import UserModelState

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

    async def create_session(self, supabase: Client, user_id: UUID, folder_id: UUID) -> UUID:
        """Creates a new session in Supabase DB and returns its ID."""
        session_id = uuid.uuid4()
        print(f"Creating new session {session_id} linked to folder {folder_id} for user {user_id}")

        # --- Fetch Folder Data ---
        folder_data = None
        initial_vector_store_id = None
        initial_analysis_result = None

        try:
            folder_response: PostgrestAPIResponse = supabase.table("folders").select("knowledge_base, vector_store_id, name").eq("id", str(folder_id)).eq("user_id", user_id).maybe_single().execute()
            if folder_response.data:
                folder_data = folder_response.data
                initial_vector_store_id = folder_data.get("vector_store_id")
                kb_text = folder_data.get("knowledge_base")
                print(f"Found existing folder data. VS_ID: {initial_vector_store_id}, KB Length: {len(kb_text) if kb_text else 0}")

                # Attempt to parse knowledge_base text into AnalysisResult
                if kb_text:
                    try:
                        # Basic parsing logic - needs to match analyzer_agent output format
                        concepts = re.findall(r"KEY CONCEPTS:\n(.*?)\nCONCEPT DETAILS:", kb_text, re.DOTALL)
                        terms_match = re.findall(r"KEY TERMS GLOSSARY:\n(.*)", kb_text, re.DOTALL) # Assume rest is terms
                        files_match = re.findall(r"FILES:\n(.*?)\nFILE METADATA:", kb_text, re.DOTALL)

                        key_concepts = [c.strip() for c in concepts[0].strip().split('\n')] if concepts else []
                        key_terms = dict(re.findall(r"^\s*([^:]+):\s*(.+)$", terms_match[0].strip(), re.MULTILINE)) if terms_match else {}
                        file_names = [f.strip() for f in files_match[0].strip().split('\n')] if files_match else []

                        initial_analysis_result = AnalysisResult(analysis_text=kb_text, key_concepts=key_concepts, key_terms=key_terms, file_names=file_names, vector_store_id=initial_vector_store_id or "")
                        print("Successfully parsed Knowledge Base into AnalysisResult object.")
                    except Exception as parse_error:
                        print(f"Warning: Failed to parse Knowledge Base text for folder {folder_id}: {parse_error}. Proceeding without parsed analysis.")
                        initial_analysis_result = None # Ensure it's None if parsing fails

            else:
                print(f"No existing folder data found for folder {folder_id}. Creating fresh context.")
                # Folder might be newly created, context will be minimal initially

        except Exception as folder_exc:
            print(f"Error fetching folder data for {folder_id}: {folder_exc}")
            # Decide whether to proceed with empty context or raise error
            # Proceeding with empty for now

        # --- Initialize TutorContext ---
        context = TutorContext(
            session_id=session_id,
            user_id=user_id,
            folder_id=folder_id,
            vector_store_id=initial_vector_store_id,
            analysis_result=initial_analysis_result,
            # Keep default UserModelState unless loading from a *previous session's* context
            user_model_state=UserModelState()
        )
        context_dict = context.model_dump(mode='json')

        try:
            response: PostgrestAPIResponse = supabase.table("sessions").insert({
                "id": str(session_id),
                "folder_id": str(folder_id),
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

    async def get_session_context(self, supabase: Client, session_id: UUID, user_id: UUID) -> Optional[TutorContext]:
        """Retrieves the TutorContext for a given session ID and user ID from Supabase."""
        try:
            response: PostgrestAPIResponse = supabase.table("sessions").select("context_data").eq("id", str(session_id)).eq("user_id", user_id).maybe_single().execute()

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

    async def update_session_context(self, supabase: Client, session_id: UUID, user_id: UUID, context: TutorContext) -> bool:
        """Updates the TutorContext for a given session ID in Supabase."""
        context_dict = context.model_dump(mode='json')
        try:
            response: PostgrestAPIResponse = supabase.table("sessions").update({"context_data": context_dict}).eq("id", str(session_id)).eq("user_id", user_id).execute()
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