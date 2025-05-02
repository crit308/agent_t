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
from dotenv import load_dotenv
from supabase import Client # Keep Client if used directly, otherwise remove if only through dependency

# Load environment variables from .env file
load_dotenv()

# --- Import Dependencies & SDK Config FIRST ---
from ai_tutor.dependencies import get_supabase_client, openai_client
from agents import set_default_openai_key, set_default_openai_api, Agent, set_default_openai_client # Import Agent and SDK setters
from ai_tutor.auth import verify_token # Assume auth.py exists for JWT verification
from ai_tutor.metrics import metrics_endpoint # Prometheus endpoint

# --- THEN Import Core Models/Context ---
# Import core models first
from ai_tutor.core_models import UserModelState, UserConceptMastery # <--- IMPORT FROM NEW LOCATION
# Import context (which now depends on core_models)
from ai_tutor.context import TutorContext # <--- Import context models (now without UserModelState/Mastery)
# Import agent models
from ai_tutor.agents.models import ( # Models from agents/models
    LessonPlan, LessonSection, LearningObjective, QuizQuestion, QuizFeedbackItem, FocusObjective
)
from ai_tutor.agents.analyzer_agent import AnalysisResult # Analyzer model
# Import API models (which now depend on core_models)
from ai_tutor.api_models import ( # *** Import your API models ***
    ErrorResponse, # Needed for exception handlers
    SessionResponse, # Needed for sessions router
    FolderCreateRequest, FolderResponse, FolderListResponse, # For folders router
    DocumentUploadResponse, AnalysisResponse, # For tutor router
    TutorInteractionResponse, InteractionResponseData, # For tutor_ws and maybe tutor router
)

# --- Configure SDK ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    # Decide if the app should exit or continue with limited functionality
    # exit(1) # Or raise an exception
else:
    set_default_openai_key(api_key)
    # set_default_openai_api("responses") # Ensure using API needed for models like o3-mini - Uncomment if needed
    print("OpenAI API key configured for agents SDK.")

# --- Register OpenAI Client ---
# Register the singleton client so all model providers reuse it.
set_default_openai_client(openai_client) # Register singleton after potential key setting
print("Default OpenAI client registered with agents SDK.")

# Supabase client is initialized lazily via dependency injection, no explicit init here needed.
# print("Supabase client initialized successfully in dependencies module.") # Or log inside dependency

# --- Rebuild Pydantic Models (AFTER all model imports) ---
# Rebuild models from core_models
UserConceptMastery.model_rebuild() # <--- FROM core_models
UserModelState.model_rebuild() # <--- FROM core_models
# Rebuild models from agents.models
LearningObjective.model_rebuild()
AnalysisResult.model_rebuild()
LessonPlan.model_rebuild()
QuizQuestion.model_rebuild()
QuizFeedbackItem.model_rebuild()
FocusObjective.model_rebuild()
# Rebuild models from api_models
TutorInteractionResponse.model_rebuild()
InteractionResponseData.model_rebuild()
SessionResponse.model_rebuild()
FolderResponse.model_rebuild()
# Rebuild context model (should be last)
TutorContext.model_rebuild() # <--- FROM context

# --- Define FastAPI App ---
app = FastAPI(
    title="AI Tutor API",
    description="API for generating lessons and quizzes using AI agents.",
    version="1.0.0",
)

# --- Add Middleware ---
# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend origin
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

# --- NOW Import and Include Routers ---
from ai_tutor.routers import sessions, tutor, folders, tutor_ws # <--- Import routers HERE

app.include_router(sessions.router, prefix="/api/v1", dependencies=[Depends(verify_token)])
app.include_router(tutor.router, prefix="/api/v1", dependencies=[Depends(verify_token)])
app.include_router(folders.router, prefix="/api/v1", dependencies=[Depends(verify_token)]) # Include folder routes
app.include_router(tutor_ws.router, prefix="/api/v1")  # Mount WebSocket router

# --- Add Root Endpoint & Metrics ---
app.add_route("/metrics", metrics_endpoint, methods=["GET"]) # Prometheus scrape endpoint

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the AI Tutor API!"}

# --- Global Exception Handlers ---
from fastapi import Request # Already imported, but good practice to keep near usage
from fastapi.responses import JSONResponse # Already imported
# ErrorResponse already imported above with other models

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Return structured JSON error response
    err = ErrorResponse(response_type="error", message=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=err.model_dump()) # Use model_dump()

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log the exception with traceback
    print(f"!!! UNHANDLED EXCEPTION: {exc} !!!")
    import traceback
    traceback.print_exc() # Print traceback to console/logs

    # Return generic internal server error
    err = ErrorResponse(response_type="error", message="Internal server error")
    return JSONResponse(status_code=500, content=err.model_dump()) # Use model_dump()

# --- Shutdown Event ---
@app.on_event("shutdown")
async def _shutdown_async_clients():
    """Ensure globally‑instantiated async clients are closed gracefully."""
    try:
        if openai_client and hasattr(openai_client, 'is_closed') and not openai_client.is_closed(): # Check if closable
            await openai_client.close()
            print("OpenAI client closed.")
    except Exception as exc:
        import logging
        logging.getLogger("ai_tutor").warning("Failed to close openai client on shutdown: %s", exc)

# To run the API: uvicorn ai_tutor.api:app --reload --port 8001 