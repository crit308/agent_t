import os, sys
# Prepend project src directory to sys.path to load local 'agents' package before pip-installed one
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, HTTPException, status
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

from ai_tutor.context import TutorContext, UserModelState, UserConceptMastery  # Import context models
from ai_tutor.routers import sessions, tutor, folders, tutor_ws # Import folders router
from ai_tutor.dependencies import get_supabase_client, openai_client # Import dependency from new location
from agents import set_default_openai_key, set_default_openai_api, Agent # Import Agent
from ai_tutor.auth import verify_token # Assume auth.py exists for JWT verification
from ai_tutor.metrics import metrics_endpoint # Prometheus endpoint

# Import models needed for resolving forward references in TutorContext and other API models
from ai_tutor.agents.models import (
    LessonPlan, LessonSection, LearningObjective, QuizQuestion, QuizFeedbackItem, FocusObjective # Add FocusObjective
)
from ai_tutor.agents.analyzer_agent import AnalysisResult
from ai_tutor.api_models import TutorInteractionResponse, InteractionResponseData # Add InteractionResponseData

# Import shared openai_client then register as default now that `agents` is fully imported.
from ai_tutor.dependencies import get_supabase_client, openai_client
from agents import set_default_openai_client

# Register the singleton client so all model providers reuse it (done here
# to avoid circular import during module initialization).
set_default_openai_client(openai_client)

# --- SDK Configuration ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    # Decide if the app should exit or continue with limited functionality
    # exit(1) # Or raise an exception
else:
    set_default_openai_key(api_key)
    set_default_openai_api("responses") # Ensure using API needed for models like o3-mini
    print("OpenAI API key configured for agents SDK.")

# --- Supabase Client Initialization is now handled in dependencies.py ---
# You might still want a check here to ensure it *was* initialized successfully if critical

# --- Rebuild Pydantic models after all imports ---
# Ensure all models potentially using forward refs are rebuilt
UserConceptMastery.model_rebuild()
UserModelState.model_rebuild()
# LessonPlan and LessonSection might depend on LearningObjective, rebuild them first if so.
LearningObjective.model_rebuild() # Rebuild LearningObjective if it uses forward refs (unlikely but safe)
AnalysisResult.model_rebuild() # Rebuild if it uses forward refs
LessonPlan.model_rebuild() # Rebuild if it uses forward refs
QuizQuestion.model_rebuild() # Rebuild if it uses forward refs
QuizFeedbackItem.model_rebuild() # Add if used in TutorInteractionResponse directly or indirectly
FocusObjective.model_rebuild() # Rebuild the newly added model if it uses forward refs (likely not, but safe)
TutorContext.model_rebuild() # Now this should work

app = FastAPI(
    title="AI Tutor API",
    description="API for generating lessons and quizzes using AI agents.",
    version="1.0.0",
)

# --- CORS Configuration ---
# Adjust origins based on your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend origin
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allow all headers (including Authorization, Content-Type)
)

# --- Mount Routers ---
# Add dependency for authentication to all routers needing it
app.include_router(sessions.router, prefix="/api/v1", dependencies=[Depends(verify_token)])
app.include_router(tutor.router, prefix="/api/v1", dependencies=[Depends(verify_token)])
app.include_router(folders.router, prefix="/api/v1", dependencies=[Depends(verify_token)]) # Include folder routes
app.include_router(tutor_ws.router, prefix="/api/v1")  # Mount WebSocket router for streaming tutor sessions

# Prometheus scrape endpoint
app.add_route("/metrics", metrics_endpoint, methods=["GET"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the AI Tutor API!"}

# Global exception handlers to return JSON ErrorResponse
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from ai_tutor.api_models import ErrorResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Return structured JSON error response
    err = ErrorResponse(response_type="error", message=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=err.dict())

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log the exception with traceback
    print(f"!!! UNHANDLED EXCEPTION: {exc} !!!")
    import traceback
    traceback.print_exc() # Print traceback to console/logs

    # Return generic internal server error
    err = ErrorResponse(response_type="error", message="Internal server error")
    return JSONResponse(status_code=500, content=err.dict())

@app.on_event("shutdown")
async def _shutdown_async_clients():
    """Ensure globally‑instantiated async clients are closed gracefully.
    This prevents OpenAI's AsyncHttpxClientWrapper __del__ from running during
    interpreter finalisation, which triggered `AttributeError: _state` in httpx.
    """
    try:
        if openai_client and not openai_client.is_closed():
            await openai_client.close()
    except Exception as exc:  # noqa: BLE001
        # Log and swallow – shutting down should continue even if close fails.
        import logging
        logging.getLogger("ai_tutor").warning("Failed to close openai client on shutdown: %s", exc)

# To run the API: uvicorn ai_tutor.api:app --reload 