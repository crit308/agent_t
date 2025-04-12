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
import logging
from datetime import datetime

from ai_tutor.context import TutorContext # Import the main context model

# Needed for context creation
from ai_tutor.context import UserModelState

# Models might still be needed if SessionManager directly interacts with them.
from ai_tutor.agents.models import LessonPlan, LessonContent, Quiz, QuizFeedback, SessionAnalysis
from ai_tutor.agents.analyzer_agent import AnalysisResult # Import AnalysisResult from its correct location

# Import ADK SessionService classes
from google.adk.sessions.base_session_service import BaseSessionService, Session, ListSessionsResponse, GetSessionConfig, ListEventsResponse
from google.adk.events import Event

if TYPE_CHECKING:
    from supabase import Client

# In-memory storage for session data.
# WARNING: This will lose state on server restart and doesn't scale horizontally.
# Consider using Redis or a database for production.
_sessions: Dict[str, Dict[str, Any]] = {}

logger = logging.getLogger(__name__)

class SupabaseSessionService(BaseSessionService):
    """A session service that uses Supabase for storage."""

    def __init__(self, supabase_client: Client):
        """Initializes the SupabaseSessionService.

        Args:
            supabase_client: An initialized Supabase client instance.
        """
        self.supabase = supabase_client
        logger.info("SupabaseSessionService initialized.")

    def create_session(
        self,
        app_name: str,
        user_id: str, # ADK expects str, convert UUID in API layer if needed
        state: Optional[Dict[str, Any]] = None, # Initial state (should contain folder_id)
        session_id: Optional[str] = None # ADK expects str
    ) -> Session:
        """Creates a new session record in Supabase."""
        session_uuid = UUID(session_id) if session_id else uuid.uuid4()
        user_uuid = UUID(user_id)

        # Initial state must contain folder_id.
        if not state or not state.get('folder_id'):
             logger.error(f"folder_id missing in initial state for session creation. State: {state}")
             raise ValueError("folder_id is required in the initial state dict.")

        try:
            folder_uuid = UUID(state['folder_id'])
        except ValueError:
            logger.error(f"Invalid folder_id format provided in state: {state.get('folder_id')}")
            raise ValueError("Invalid folder_id format.")

        # Create a default TutorContext and serialize it for initial storage
        try:
            initial_context = TutorContext(session_id=session_uuid, user_id=user_uuid, folder_id=folder_uuid)
            context_dict = initial_context.model_dump(mode='json') # Use Pydantic dump for consistency
            logger.info(f"Creating session {session_uuid} for user {user_id}, folder {folder_uuid} with initial context.")
        except Exception as model_error:
            logger.exception(f"Failed to create initial TutorContext model: {model_error}")
            raise ValueError(f"Failed to initialize context state: {model_error}")

        # Persist to Database
        try:
            response: PostgrestAPIResponse = self.supabase.table("sessions").insert({
                "id": str(session_uuid),
                "user_id": str(user_uuid),
                "app_name": app_name,
                "context_data": context_dict, # Store the serialized valid context
                "folder_id": str(folder_uuid) # Ensure folder_id column is populated
            }).execute()

            if not response.data:
                error_info = response.error
                error_msg = f"Failed to create session record in DB: {error_info.message if error_info else 'Unknown DB error'}"
                logger.error(error_msg)
                # Raise HTTPException here because this is an internal server error
                raise HTTPException(status_code=500, detail="Failed to save session.")

            logger.info(f"Session {session_uuid} created successfully in Supabase.")

            # Return ADK Session object with the *actual initial state used*
            # Using the passed 'state' might be incorrect if create_session modified/created it
            return Session(
                id=str(session_uuid),
                app_name=app_name,
                user_id=user_id,
                state=context_dict, # Return the validated, serialized context dict
                events=[],
                last_update_time=time.time() # Use current time as last update
            )

        except HTTPException as http_exc: # Re-raise HTTPExceptions
            raise http_exc
        except Exception as db_error:
            error_msg = f"Database error during session creation: {db_error}"
            logger.exception(error_msg)
            raise HTTPException(status_code=500, detail="Database error during session creation.")

    def get_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None
    ) -> Optional[Session]:
        """Gets a session from Supabase."""
        session_uuid = UUID(session_id)
        user_uuid = UUID(user_id)
        logger.debug(f"Attempting to get session {session_uuid} for user {user_uuid}")

        try:
            # Select all necessary fields for session reconstruction
            response: PostgrestAPIResponse = self.supabase.table("sessions").select(
                "context_data, updated_at, folder_id"
            ).eq("id", str(session_uuid)).eq("user_id", str(user_uuid)).maybe_single().execute()

            if response.data:
                logger.debug(f"Session {session_uuid} found.")
                context_data = response.data.get("context_data", {})
                updated_at_str = response.data.get("updated_at")
                folder_id = response.data.get("folder_id")

                # Validate required fields
                if not folder_id:
                    logger.error(f"Session {session_uuid} missing folder_id.")
                    return None

                # Ensure context_data has minimum required fields
                if not isinstance(context_data, dict):
                    logger.error(f"Session {session_uuid} has invalid context_data type: {type(context_data)}")
                    return None

                # Ensure folder_id is in context
                if "folder_id" not in context_data:
                    context_data["folder_id"] = folder_id

                # Parse timestamp
                try:
                    # updated_at_str_no_tz = updated_at_str.split('+')[0] # Remove timezone offset if present
                    # last_update_time = datetime.fromisoformat(updated_at_str_no_tz).timestamp()
                    # More robust parsing: handle potential fractional seconds and timezone
                    if '+' in updated_at_str:
                         # Handle timezone offset (e.g., +00)
                        dt_obj = datetime.strptime(updated_at_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                    else:
                        # Assume no timezone, parse just the time part
                        dt_obj = datetime.strptime(updated_at_str, "%Y-%m-%dT%H:%M:%S.%f")
                    last_update_time = dt_obj.timestamp()
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid timestamp format for session {session_uuid}: '{updated_at_str}'. Error: {e}. Using current time.")
                    last_update_time = time.time()

                # Optional: Validate context structure matches TutorContext
                try:
                    # This validates the structure but we keep the dict form
                    _ = TutorContext.model_validate(context_data)
                    logger.debug(f"Session {session_uuid} context validated successfully.")
                except Exception as e:
                    logger.error(f"Session {session_uuid} has invalid context structure: {e}")
                    # Consider whether to return None or continue with partial data
                    # For now, continue but log the error
                    logger.warning("Continuing with potentially invalid context data.")

                # Return ADK Session object
                return Session(
                    id=session_id,
                    app_name=app_name,
                    user_id=user_id,
                    state=context_data, # Store validated context dict in state
                    events=[], # Events not persisted in this simple version
                    last_update_time=last_update_time
                )
            else:
                logger.warning(f"Session {session_uuid} not found for user {user_uuid}.")
                return None

        except Exception as e:
            logger.exception(f"Error fetching session {session_uuid}: {e}")
            # Don't raise HTTPException here, return None as per BaseSessionService expectation
            return None

    def list_sessions(
        self,
        app_name: str,
        user_id: str,
        config: Optional[GetSessionConfig] = None
    ) -> ListSessionsResponse:
        """Lists sessions for a user from Supabase."""
        user_uuid = UUID(user_id)
        logger.debug(f"Listing sessions for user {user_uuid}")
        try:
            # Fetch only necessary fields for listing
            response: PostgrestAPIResponse = self.supabase.table("sessions").select("id, updated_at").eq("user_id", str(user_uuid)).order("updated_at", desc=True).execute()

            if response.data:
                sessions = []
                for item in response.data:
                    updated_at_str = item.get("updated_at")
                    last_update_time = datetime.fromisoformat(updated_at_str).timestamp() if updated_at_str else time.time()
                    sessions.append(Session(
                        id=item['id'],
                        app_name=app_name,
                        user_id=user_id,
                        state={}, # Don't load full state for listing
                        events=[],
                        last_update_time=last_update_time
                    ))
                return ListSessionsResponse(sessions=sessions)
            else:
                return ListSessionsResponse(sessions=[]) # Return empty list if none found
        except Exception as e:
            logger.exception(f"Error listing sessions for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error listing sessions: {e}")

    def delete_session(self, session_id: str) -> None:
        """Deletes a session and its associated data from Supabase.
        
        Args:
            session_id: String representation of session UUID
            
        Raises:
            HTTPException: If session deletion fails or session not found
        """
        try:
            session_uuid = UUID(session_id)
            logger.info(f"Attempting to delete session {session_uuid}")

            # First verify session exists and get folder_id
            response = self.supabase.table("sessions").select("folder_id").eq("id", str(session_uuid)).execute()
            
            if not response.data:
                error_msg = f"Session {session_uuid} not found"
                logger.warning(error_msg)
                raise HTTPException(status_code=404, detail=error_msg)

            folder_id = response.data[0].get('folder_id')
            if not folder_id:
                logger.error(f"Session {session_uuid} found but missing folder_id")
                
            # Delete session record
            delete_response = self.supabase.table("sessions").delete().eq("id", str(session_uuid)).execute()
            
            if not delete_response.data:
                error_msg = f"Failed to delete session {session_uuid}: {delete_response.error.message if delete_response.error else 'Unknown error'}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

            logger.info(f"Successfully deleted session {session_uuid}")

            # Attempt to cleanup associated storage files if folder_id exists
            if folder_id:
                try:
                    storage_path = f"sessions/{folder_id}"
                    self.supabase.storage.from_("tutoring").remove([storage_path])
                    logger.info(f"Cleaned up storage for session {session_uuid} at path {storage_path}")
                except Exception as storage_error:
                    # Log but don't fail if storage cleanup fails
                    logger.warning(f"Failed to cleanup storage for session {session_uuid}: {storage_error}")

        except ValueError as ve:
            # Invalid UUID format
            error_msg = f"Invalid session ID format: {ve}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error deleting session {session_id}: {e}"
            logger.exception(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    def append_event(self, session: Session, event: Event) -> Event:
        """Appends an event's state delta to the session's context_data in Supabase."""
        session_id = UUID(session.id)
        user_id = UUID(session.user_id)
        
        # ADK Runner should have already merged event.actions.state_delta
        # into the session.state dictionary before calling this method.
        # We just need to persist the updated session.state back to Supabase.
        updated_context_dict = session.state # Get the already updated state dict
        
        # Ensure state is JSON serializable (Pydantic's model_dump in get/create helps)
        logger.debug(f"Persisting updated context for session {session_id}")
        try: # Ensure `updated_at` trigger is working correctly in Supabase
            response: PostgrestAPIResponse = self.supabase.table("sessions").update({
                "context_data": updated_context_dict,
                "updated_at": datetime.now().isoformat() # Update timestamp explicitly
            }).eq("id", str(session_id)).eq("user_id", str(user_id)).execute()

            if not response.data:
                error_detail = getattr(response, 'error', None) or getattr(response, 'message', 'Unknown error')
                logger.error(f"Error persisting event/state for session {session_id}: {error_detail}")
            else:
                logger.debug(f"Successfully persisted event {event.id} for session {session_id}")

            # Update the timestamp on the in-memory event object (optional, depends if caller uses it)
            # Fetching the actual update_time from DB would require another query.
            # Using current time is a reasonable approximation.
            event.timestamp = time.time()
            session.last_update_time = event.timestamp # Also update session timestamp
            return event # Return the event (potentially updated)
        except Exception as e:
            logger.exception(f"Database exception persisting event/state for session {session_id}: {e}")
            # Handle exception - maybe raise or just log and continue
            return event # Return original event on error

    # --- Methods below are not strictly required if events aren't persisted ---
    # --- Implement if full event history storage in DB is needed           ---

    def list_events(self, session: Session, config: Optional[GetSessionConfig] = None) -> ListEventsResponse:
        """Lists events for a session (Not implemented for Supabase persistence)."""
        logger.warning("list_events not implemented for SupabaseSessionService (events not persisted). Returning empty.")
        return ListEventsResponse(events=[])

    # def close_session(self, *, session: Session): # Optional implementation
    #     logger.info(f"Closing session {session.id} (no specific action in SupabaseSessionService).")
    #     pass

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