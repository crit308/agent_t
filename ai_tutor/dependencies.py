# ai_tutor/dependencies.py
import os
from fastapi import HTTPException, status
from supabase import create_client, Client
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai._base_client import AsyncHttpxClientWrapper  # type: ignore

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

# Create a single global AsyncOpenAI client and share it with the `agents`
# package so that all model providers use the exact same instance.
openai_client = AsyncOpenAI()

# Note: we register this client with the agents SDK in `ai_tutor.api` *after*
# `agents` has been fully imported, to avoid a circular‑import crash.

def get_openai():
    """FastAPI dependency returning the shared AsyncOpenAI client."""
    return openai_client 

# Monkey‑patch OpenAI client wrapper to avoid AttributeError at program exit
# Preserve original __del__
_orig_del = AsyncHttpxClientWrapper.__del__


def _safe_del(self):  # noqa: D401
    """A safer __del__ that swallows the httpx ClientState cleanup error.

    When Python shuts down, httpx's `ClientState` enum may already be
    garbage‑collected. Accessing `self._state` then raises AttributeError,
    which pollutes stderr with tracebacks.  We ignore that specific case.
    """

    try:
        _orig_del(self)
    except AttributeError as exc:
        if "_state" not in str(exc):
            raise  # Unexpected attribute error, re‑raise
        # else: swallow – it's the known shutdown race condition.


# Install the patch only once
if getattr(AsyncHttpxClientWrapper.__del__, "_patched", False) is False:  # type: ignore[attr-defined]
    AsyncHttpxClientWrapper.__del__ = _safe_del  # type: ignore[assignment]
    AsyncHttpxClientWrapper.__del__._patched = True  # type: ignore[attr-defined] 