# ai_tutor/dependencies.py
import os
from fastapi import HTTPException, status
from supabase import create_client, Client
from dotenv import load_dotenv

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
        print(f"ERROR: Failed to initialize Supabase client in dependencies module: {e}")
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
    return SUPABASE_CLIENT 