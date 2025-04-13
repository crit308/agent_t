from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, HTTPException, status
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from contextlib import asynccontextmanager # Import asynccontextmanager
import logging
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

from ai_tutor.context import TutorContext, UserModelState, UserConceptMastery  # Import context models
from ai_tutor.routers import sessions, tutor, folders # Import folders router
from ai_tutor.dependencies import get_supabase_client, get_session_service # Import new service dependency
from google.adk.agents import LlmAgent # Use ADK imports
from google.adk.runners import Runner, RunConfig # Use ADK imports
from ai_tutor.auth import verify_token # Assume auth.py exists for JWT verification
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent, generate_quiz # Assuming these exist

# Import models needed for resolving forward references in TutorContext and other API models
from ai_tutor.agents.models import (
    LessonPlan, LessonSection, LearningObjective, QuizQuestion, QuizFeedbackItem, FocusObjective # Add FocusObjective
)
from ai_tutor.agents.analyzer_agent import AnalysisResult
from ai_tutor.api_models import TutorInteractionResponse, InteractionResponseData # Add InteractionResponseData
from ai_tutor.session_manager import SupabaseSessionService # Needed for Runner init

# Import models and context to allow for rebuild
from ai_tutor.agents import models # Ensure models are imported

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini client (ensure GOOGLE_API_KEY is set in .env)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.warning("GOOGLE_API_KEY not found in environment variables.")
    # Decide how to handle missing key: raise error, exit, or proceed with limited functionality
    # raise ValueError("GOOGLE_API_KEY must be set in the environment.")
else:
    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini client configured successfully.")
    except Exception as e:
         logger.error(f"Failed to configure Gemini client: {e}")
         # Handle configuration error appropriately

# --- FastAPI Lifespan for Runner Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: Initialize the runner cache
    app.state.adk_runners = {} # Dictionary to hold Runner instances per session_id
    print("Initialized ADK Runner cache on app startup.")
    
    # Startup logic
    logger.info("Application startup...")
    # Initialize database connections, load models, etc.
    # Example: await database.connect()
    
    # --- Rebuild Pydantic models with forward references resolved ---
    # Ensure all necessary models are defined before calling this
    try:
        logger.info("Rebuilding Pydantic models (TutorContext)...")
        TutorContext.model_rebuild(force=True) # Force rebuild might help
        logger.info("TutorContext model rebuild complete.")
    except Exception as e:
        logger.exception(f"Error rebuilding Pydantic models: {e}")
        # Decide if this is a fatal error for startup
    
    yield
    # On shutdown: Clean up runners if needed (optional)
    app.state.adk_runners.clear()
    print("Cleared ADK Runner cache on app shutdown.")
    # Shutdown logic
    logger.info("Application shutdown...")
    # Example: await database.disconnect()

# --- REMOVE Redundant Rebuild Calls --- 
# Ensure all models potentially using forward refs are rebuilt
# UserConceptMastery.model_rebuild() # REMOVED
# UserModelState.model_rebuild() # REMOVED
# LessonPlan and LessonSection might depend on LearningObjective, rebuild them first if so.
# LearningObjective.model_rebuild() # REMOVED
# AnalysisResult.model_rebuild() # REMOVED
# LessonPlan.model_rebuild() # REMOVED
# QuizQuestion.model_rebuild() # REMOVED
# QuizFeedbackItem.model_rebuild() # REMOVED
# FocusObjective.model_rebuild() # REMOVED
# TutorContext.model_rebuild() # REMOVED - Keep the one inside lifespan

app = FastAPI(
    title="AI Tutor API (ADK Backend)",
    description="API for generating lessons and quizzes using AI agents.",
    version="1.0.0",
    lifespan=lifespan # Add the lifespan context manager
)

# --- CORS Configuration ---
# Adjust origins based on your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for now, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize ADK Runner (conceptually here, might move) ---
# from google.adk.runners import Runner
# session_service = get_session_service() # Need to manage instance lifecycle
# adk_runner = Runner(app_name="ai_tutor", agent=orchestrator_agent, session_service=session_service)

# --- Mount Routers ---
# Add dependency for authentication to all routers needing it
app.include_router(sessions.router, prefix="/api/v1", dependencies=[Depends(verify_token)])
app.include_router(tutor.router, prefix="/api/v1", dependencies=[Depends(verify_token)])
app.include_router(folders.router, prefix="/api/v1", dependencies=[Depends(verify_token)]) # Include folder routes

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the AI Tutor API!"}

# To run the API: uvicorn ai_tutor.api:app --reload 