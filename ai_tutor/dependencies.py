# ai_tutor/dependencies.py
import os
from fastapi import HTTPException, status
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import Optional
from fastapi import Depends

# Load environment variables specifically for dependencies if needed,
# though they should be loaded by the main app process already.
load_dotenv()

# --- Supabase Client Initialization ---
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") # Use Service Key for backend operations

SUPABASE_CLIENT: Client | None = None
if supabase_url and supabase_key:
    try:
        SUPABASE_CLIENT = create_client(supabase_url, supabase_key)
        print("Supabase client initialized successfully in dependencies module.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Supabase client: {e}") # Simplified log
        # Depending on severity, you might want to prevent app startup
else:
    print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set for Supabase client.")


# --- Dependency Function ---
async def get_supabase_client() -> Client:
    """FastAPI dependency to get the initialized Supabase client."""
    if SUPABASE_CLIENT is None:
        # This condition should ideally not be met if env vars are set correctly
        # and initialization succeeded above.
        print("ERROR: get_supabase_client called but client is not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase client is not available. Check backend configuration and logs."
        )
    # Simple check if the client seems initialized - replace with a proper health check if needed
    # if not hasattr(SUPABASE_CLIENT, 'table'): # Example basic check
    #    raise HTTPException(
    #        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    #        detail="Supabase client appears uninitialized."
    #    )
    return SUPABASE_CLIENT

# --- NEW: Dependency for SupabaseSessionService ---
from ai_tutor.session_manager import SupabaseSessionService # Adjust import path if needed

_supabase_session_service_instance: Optional[SupabaseSessionService] = None

async def get_session_service(supabase: Client = Depends(get_supabase_client)) -> SupabaseSessionService:
    """FastAPI dependency to get the SupabaseSessionService instance."""
    global _supabase_session_service_instance
    if _supabase_session_service_instance is None:
        if supabase:
            _supabase_session_service_instance = SupabaseSessionService(supabase_client=supabase)
        else:
            # This should not happen if get_supabase_client works
            raise HTTPException(status_code=503, detail="Supabase client unavailable for SessionService.")
    return _supabase_session_service_instance 

# --- Dependency for TutorOutputLogger ---
from ai_tutor.output_logger import TutorOutputLogger # Corrected import path

_tutor_output_logger_instance: Optional[TutorOutputLogger] = None

def get_tutor_output_logger() -> TutorOutputLogger:
    """FastAPI dependency to get the TutorOutputLogger instance."""
    global _tutor_output_logger_instance
    if _tutor_output_logger_instance is None:
        # Initialize the logger instance here. Adjust if it needs parameters.
        _tutor_output_logger_instance = TutorOutputLogger()
        print("TutorOutputLogger instance created.")
    return _tutor_output_logger_instance 