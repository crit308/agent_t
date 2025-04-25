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
import structlog # Add structlog

from ai_tutor.context import TutorContext, UserModelState # Import UserModelState
from pydantic import ValidationError # Import ValidationError

# Models might still be needed if SessionManager directly interacts with them.
from ai_tutor.agents.models import LessonPlan, LessonContent, Quiz, QuizFeedback, SessionAnalysis
from ai_tutor.agents.analyzer_agent import AnalysisResult # Import AnalysisResult from its correct location

# use TutorFSM directly
# from ai_tutor.fsm import TutorFSM

if TYPE_CHECKING:
    from supabase import Client

# In-memory storage for session data.
# WARNING: This will lose state on server restart and doesn't scale horizontally.
# Consider using Redis or a database for production.
_sessions: Dict[str, Dict[str, Any]] = {}

log = structlog.get_logger(__name__) # Add logger

class SessionManager:
    """Manages session state for AI Tutor sessions."""

    def __init__(self):
        """Initialize the session manager."""
        pass

    async def create_session(self, supabase: Client, user_id: UUID, folder_id: Optional[UUID] = None) -> UUID:
        """Creates a new session in Supabase DB and returns its ID.

        If folder_id is provided, attempts to load initial context from the folder.
        """
        session_id = uuid.uuid4()
        print(f"Creating new session {session_id} for user {user_id}. Linked folder: {folder_id if folder_id else 'None'}")

        # --- Fetch Folder Data (Only if folder_id is provided) ---
        folder_data = None
        initial_vector_store_id = None
        initial_analysis_result = None
        folder_name = "Untitled Session" # Default name if no folder

        if folder_id:
            try:
                folder_response: PostgrestAPIResponse = supabase.table("folders").select("knowledge_base, vector_store_id, name").eq("id", str(folder_id)).eq("user_id", user_id).maybe_single().execute()
                if folder_response.data:
                    folder_data = folder_response.data
                    initial_vector_store_id = folder_data.get("vector_store_id")
                    kb_text = folder_data.get("knowledge_base")
                    folder_name = folder_data.get("name", folder_name) # Use folder name if available
                    print(f"Found existing folder data for folder {folder_id}. VS_ID: {initial_vector_store_id}, KB Length: {len(kb_text) if kb_text else 0}")

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
                    # This case should ideally be handled by the ownership check in the router
                    # If we reach here, it means the folder_id exists but doesn't belong to the user or doesn't exist
                    # Raising an error might be more appropriate than creating a default context silently.
                    print(f"Warning: Folder {folder_id} not found or not owned by user {user_id}. Creating session without folder context.")
                    # Consider raising HTTPException(status_code=404, detail="Folder not found or access denied") here
                    folder_id = None # Treat as if no folder_id was provided

            except Exception as folder_exc:
                print(f"Error fetching folder data for {folder_id}: {folder_exc}")
                # Decide whether to proceed with empty context or raise error
                # Proceeding with empty context for now, but clearing folder_id
                folder_id = None

        # --- Initialize TutorContext ---
        context = TutorContext(
            session_id=session_id,
            user_id=user_id,
            folder_id=folder_id, # This will be None if not provided or fetch failed
            vector_store_id=initial_vector_store_id,
            analysis_result=initial_analysis_result,
            user_model_state=UserModelState()
        )
        context_dict = context.model_dump(mode='json')

        # --- Insert into Supabase ---
        try:
            insert_data = {
                "id": str(session_id),
                "user_id": user_id,
                "context_data": context_dict,
                "folder_id": str(folder_id) if folder_id else None # Store folder_id or NULL
                # "name": folder_name # Optionally add a name field to sessions table
            }
            response: PostgrestAPIResponse = supabase.table("sessions").insert(insert_data).execute()

            if response.data:
                print(f"Created session {session_id} for user {user_id} in Supabase.")
                return session_id
            else:
                print(f"Error creating session {session_id} in Supabase: {response.error}")
                raise HTTPException(status_code=500, detail=f"Failed to create session in database: {response.error.message if response.error else 'Unknown error'}")
        except Exception as e:
            print(f"Exception creating session {session_id}: {e}")
            # Check if the error is due to nullable constraint on folder_id if it's not nullable
            if "violates not-null constraint" in str(e) and "folder_id" in str(e):
                 raise HTTPException(status_code=500, detail="Database configuration error: sessions.folder_id cannot be null.")
            raise HTTPException(status_code=500, detail=f"Database error during session creation: {e}")

    async def get_session_context(self, supabase: Client, session_id: UUID, user_id: UUID) -> Optional[TutorContext]:
        """Retrieves and validates the TutorContext from Supabase."""
        log.debug("Attempting to fetch session context from DB", session_id=str(session_id), user_id=str(user_id))
        try:
            response: PostgrestAPIResponse = supabase.table("sessions").select("context_data").eq("id", str(session_id)).eq("user_id", str(user_id)).maybe_single().execute()

            if not response.data:
                log.info("No session found or context_data is null", session_id=str(session_id), user_id=str(user_id))
                return None # No session found for this user/id

            context_data = response.data.get("context_data")
            if not context_data:
                 log.warning("Session found but context_data field is null or empty.", session_id=str(session_id))
                 # Decide if this should return None or initialize a default context
                 # Returning None for now, assuming FE/caller handles initialization
                 return None

            # --- Parse and Validate --- #
            try:
                if isinstance(context_data, str):
                    try:
                        context_dict = json.loads(context_data)
                    except json.JSONDecodeError as json_err:
                        log.error("Failed to decode JSON from context_data", session_id=str(session_id), error=str(json_err))
                        # Raise a specific exception or return None, depending on desired handling
                        # Raising HTTPException might be too much here, maybe a custom error or None
                        raise ValueError("Invalid JSON in context_data") from json_err
                elif isinstance(context_data, dict):
                    context_dict = context_data
                else:
                    log.error("context_data is neither string nor dict", session_id=str(session_id), type=type(context_data).__name__)
                    raise ValueError("Invalid type for context_data")

                # Ensure core IDs are present before Pydantic validation (Supabase might not guarantee this)
                context_dict.setdefault('session_id', str(session_id))
                context_dict.setdefault('user_id', str(user_id))

                # Validate with Pydantic
                ctx = TutorContext.model_validate(context_dict)
                log.info("Session context successfully fetched and validated", session_id=str(session_id))
                return ctx

            except (ValidationError, ValueError, TypeError) as validation_err:
                 log.error("Failed to validate context_data against TutorContext model", session_id=str(session_id), error=str(validation_err), exc_info=True)
                 # Return None or raise a specific error? Returning None for now.
                 # This indicates corrupted data in the DB.
                 return None # Treat validation failure as if context doesn't exist or is unusable

        except APIError as api_err:
            log.error("Supabase APIError fetching session context", session_id=str(session_id), code=api_err.code, message=api_err.message, exc_info=True)
            # Propagate a generic error upwards
            raise HTTPException(status_code=503, detail=f"Database error fetching session: {api_err.code}") # 503 Service Unavailable might be appropriate
        except Exception as e:
            log.error("Unexpected error fetching session context from Supabase", session_id=str(session_id), error=str(e), exc_info=True)
            # Propagate a generic error upwards
            raise HTTPException(status_code=500, detail=f"Internal error fetching session context")

    async def update_session_context(self, supabase: Client, session_id: UUID, user_id: UUID, context: TutorContext) -> bool:
        """Updates the TutorContext for a given session ID in Supabase."""
        log.debug("Attempting to update session context in DB", session_id=str(session_id), user_id=str(user_id))
        try:
            # Ensure context can be serialized before attempting DB update
            context_dict = context.model_dump(mode='json')
            # Basic check: ensure it's a dictionary
            if not isinstance(context_dict, dict):
                 log.error("Context serialization failed, did not produce a dictionary.", session_id=str(session_id), type=type(context_dict).__name__)
                 raise TypeError("Failed to serialize context to dictionary")
            log.debug("Context serialized successfully for update.", session_id=str(session_id))

        except (ValidationError, TypeError) as serialization_err:
             log.error("Failed to serialize TutorContext before update", session_id=str(session_id), error=str(serialization_err), exc_info=True)
             # Don't raise HTTPException here, return False or raise a specific internal error?
             # Returning False indicates update failure without stopping WebSocket potentially.
             return False

        try:
            update_data = {
                "context_data": context_dict,
                "folder_id": str(context.folder_id) if context.folder_id else None
                # Add updated_at? Supabase trigger should handle this automatically.
            }
            log.info("Executing Supabase update for session context", session_id=str(session_id))
            response: PostgrestAPIResponse = supabase.table("sessions").update(update_data).eq("id", str(session_id)).eq("user_id", str(user_id)).execute()

            # Check if the update actually affected any rows (although .execute() might raise error on failure)
            # Supabase python client v1/v2 behaviour might differ here. Assuming execute raises on DB error.
            # if not response.data: # This check might be unreliable depending on Supabase version/return
            #    log.warning("Supabase update command executed but reported no data change.", session_id=str(session_id))
            #    # Consider this a soft failure?
            #    # return False

            log.info("Session context updated successfully in DB", session_id=str(session_id))
            return True
        except APIError as api_err:
            log.error("Supabase APIError updating session context", session_id=str(session_id), code=api_err.code, message=api_err.message, data_sent=str(update_data)[:200], exc_info=True)
            # Raise a specific exception to be caught by the caller (e.g., tutor_ws)
            raise HTTPException(status_code=503, detail=f"Database error updating session: {api_err.code}")
        except Exception as e:
            log.error("Unexpected error updating session context in Supabase", session_id=str(session_id), error=str(e), exc_info=True)
            # Raise a specific exception
            raise HTTPException(status_code=500, detail=f"Internal error updating session context")

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