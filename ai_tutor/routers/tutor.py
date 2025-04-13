from __future__ import annotations
import logging # Add standard logging import
import os
import json
import traceback  # Add traceback import
import shutil # Make sure shutil is imported
import time
import asyncio
from typing import Optional, List, Dict, Any, Literal, Union, TypeVar, Callable, Awaitable
from uuid import UUID
from datetime import datetime, timedelta
from collections import defaultdict
import io # Import io for BytesIO

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form, Body, Request
from supabase import Client
from gotrue.types import User
from pydantic import BaseModel
from pydantic_core import ValidationError # Import Pydantic validation error

# ADK and Agent related imports
from google.adk.runners import Runner, RunConfig
from google.adk import Agent # Use top-level Agent alias
from google.adk.events import Event # Import Event from correct module
from ai_tutor.manager import AITutorManager
# Import Content and Part from the content_types submodule
from google.genai.types import GenerateContentResponse, FunctionCall # Corrected import
# Consolidate imports from google.genai.types
from google.genai.types import Content, Part, FunctionResponse 
from ai_tutor.output_logger import TutorOutputLogger # Corrected import
from ai_tutor.dependencies import get_supabase_client, get_session_service, get_tutor_output_logger # Import the new dependency factory
from ai_tutor.auth import verify_token
from google.api_core import exceptions as google_exceptions
from google.adk.tools import FunctionTool, ToolContext
import google.genai.errors # Add this import
import google.generativeai as genai # Import Gemini library

# Project specific imports
from ai_tutor.session_manager import SessionManager, SupabaseSessionService, Session # Use Session from ADK session_manager
from ai_tutor.tools.file_upload import FileUploadManager
from ai_tutor.agents.analyzer_agent import analyze_documents, AnalysisResult
from ai_tutor.agents.session_analyzer_agent import analyze_teaching_session
from ai_tutor.agents.orchestrator_agent import create_orchestrator_agent
from ai_tutor.agents.models import (
    FocusObjective,
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis, QuizQuestion, QuizFeedbackItem
)
from ai_tutor.api_models import (
    DocumentUploadResponse, AnalysisResponse, TutorInteractionResponse,
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse,
    InteractionRequestData, InteractionResponseData
)
from ai_tutor.context import TutorContext, UserModelState
import uuid

router = APIRouter()
# session_manager = SessionManager() # Keep for create_session? Or fully replace with service? Consider removing.

# Setup standard Python logger for this module
logger = logging.getLogger(__name__)

# Directory for temporary file uploads
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Add type variable for generic return type
T = TypeVar('T')

# --- Dependency to get TutorContext from DB ---
async def get_tutor_context(
    session_id: UUID, # Expect UUID
    request: Request, # Access user from request state
    session_service: SupabaseSessionService = Depends(get_session_service) # Use ADK Service
) -> TutorContext:
    user: User = request.state.user # Get authenticated user
    # ADK session service expects string IDs
    session = session_service.get_session(app_name="ai_tutor", user_id=str(user.id), session_id=str(session_id))
    if not session or not session.state:
        logger.warning(f"Session not found or state missing for session_id: {session_id}, user_id: {user.id}")
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or not authorized for user.")
    
    # --- DEBUGGING ---
    logger.debug(f"Raw session state fetched for {session_id}: {session.state}")
    try:
        # Deserialize TutorContext from session state dictionary
        validated_context = TutorContext.model_validate(session.state)
        logger.debug(f"Successfully validated TutorContext for session {session_id}")
        return validated_context
    except ValidationError as e:
        # Log the detailed Pydantic errors
        error_details = e.errors()
        logger.error(f"Pydantic validation failed for TutorContext in session {session_id}: {error_details}")
        # It might be helpful to log the raw state again alongside the error
        logger.error(f"Raw state causing validation error: {session.state}") 
        # Raise 500 to avoid 422 if it's a server-side data consistency issue
        raise HTTPException(status_code=500, detail=f"Internal server error: Session context data is invalid. Details: {error_details}")
    except Exception as e:
        logger.exception(f"Unexpected error during TutorContext validation for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error validating session context.")

# --- Helper to get logger ---
def get_session_logger(session_id: UUID) -> TutorOutputLogger:
    # Customize logger per session if needed, e.g., different file path
    log_file = os.path.join("logs", f"session_{session_id}.log") # Example path
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Use the renamed import for clarity
    return get_tutor_output_logger(output_file=log_file)

# --- Define Models for New Endpoints ---
class MiniQuizLogData(BaseModel):
    question: str
    selectedOption: str # Match frontend naming
    correctOption: str # Match frontend naming
    isCorrect: bool    # Match frontend naming
    relatedSection: Optional[str] = None
    topic: Optional[str] = None

# --- Endpoints ---

@router.post("/sessions/{session_id}/documents", response_model=DocumentUploadResponse)
async def upload_documents(
    session_id: UUID,
    request: Request,
    files: List[UploadFile] = File(...),
    supabase: Client = Depends(get_supabase_client),
    # genai_client: genai.Client = Depends(get_genai_client) # Add dependency later
):
    """ # noqa: D415
    Uploads one or more documents to the specified session.
    Uploads to Google File API using genai.upload_file with path, stores the file URI in the database,
    and triggers analysis.
    """
    user: User = request.state.user
    logger.info(f"\n=== Received /documents for session {session_id} ===")

    # --- Manually fetch context & Ensure GenAI Config --- 
    try:
        session_service: SupabaseSessionService = await get_session_service(supabase)
        tutor_context: TutorContext = await get_tutor_context(session_id, request, session_service)
        
        # Ensure GenAI is configured (assuming configure was called at startup in api.py)
        if not os.getenv("GOOGLE_API_KEY"):
             raise ValueError("GOOGLE_API_KEY not configured.")
        # REMOVED client = genai.Client()
             
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Error manually fetching context or checking GenAI config for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error preparing session.")
    # --- End Manual Fetch & Setup ---

    folder_id = tutor_context.folder_id
    if not folder_id:
        logger.error(f"Session context is missing folder_id for session {session_id}.")
        raise HTTPException(status_code=400, detail="Session context is invalid (missing folder ID).")
    
    logger.info(f"Starting Google File API upload for session {session_id}, folder {folder_id}")
    uploaded_file_details = [] # Store tuples of (original_filename, gemini_file_name)
    upload_errors = []
    temp_file_paths_to_clean = [] # Keep track of temp paths for cleanup

    # 1. Save locally and Upload files to Google File API using file path
    for file in files:
        filename = os.path.basename(file.filename or f"upload_{uuid.uuid4()}") 
        temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")
        temp_file_paths_to_clean.append(temp_path) # Add to cleanup list
        logger.debug(f"Saving {filename} temporarily to {temp_path}")
        try:
            # Save locally first
            content = await file.read()
            with open(temp_path, "wb") as temp_file:
                temp_file.write(content)

            # --- Use Google File API with Path --- 
            logger.debug(f"Uploading {filename} from path {temp_path} to Google File API...")
            google_file = genai.upload_file(
                path=temp_path, # Pass the file path string
                display_name=f"{session_id}_{filename}"
            )
            logger.info(f"Successfully uploaded {filename} to Google File API. Name: {google_file.name}")
            uploaded_file_details.append((filename, google_file.name))
            # ----------------------------------- 
        except Exception as api_err:
             logger.exception(f"Error saving or uploading {filename} to Google File API: {api_err}")
             upload_errors.append(f"Failed to upload {filename}: {api_err}")
             # Don't add to uploaded_file_details if upload failed

    if not uploaded_file_details:
         # If all uploads failed
         error_detail = "Failed to upload any files to backend service. " + "; ".join(upload_errors)
         # Clean up any temp files that were created
         for path in temp_file_paths_to_clean:
              if os.path.exists(path): os.remove(path)
         raise HTTPException(status_code=500, detail=error_detail)
    
    message = f"{len(uploaded_file_details)} file(s) uploaded successfully to backend. "
    if upload_errors:
        message += f"Errors occurred for some files: {'; '.join(upload_errors)}. "

    # 2. Update Database with Gemini File Names
    # Store the list of Gemini file names. Adjust schema if needed.
    # Option 1: Store as JSON list in a single column (e.g., 'gemini_file_names')
    # Option 2: Store in a separate related table (documents table)
    # Assuming Option 1 for simplicity here.
    gemini_names_list = [name for _, name in uploaded_file_details]
    try:
        logger.debug(f"Updating gemini_file_names in folders table for folder {folder_id}")
        db_response = supabase.table("folders").update({
            "gemini_file_names": gemini_names_list, # Assuming a column of type text[] or jsonb
            "updated_at": datetime.now().isoformat(),
            "knowledge_base": None # Explicitly clear old knowledge_base if it exists
        }).eq("id", str(folder_id)).eq("user_id", str(user.id)).execute()

        if hasattr(db_response, 'error') and db_response.error:
             logger.error(f"Supabase DB update failed for folder {folder_id}: {db_response.error}")
             raise Exception(f"Database update failed: {db_response.error}")
        elif not db_response.data:
             logger.warning(f"Supabase DB update for gemini_file_names didn't affect any rows for folder {folder_id}.")
             # This might be an error depending on workflow
             
        logger.info(f"Successfully updated database for folder {folder_id} with Gemini file names.")
        message += "Database updated with file references. "
        # Update context in memory
        tutor_context.gemini_file_names = gemini_names_list
        # tutor_context.knowledge_base = None # REMOVED - Field no longer exists
        # if tutor_context.analysis_result: # REMOVED - Clear old analysis text if needed elsewhere
        #      tutor_context.analysis_result.analysis_text = None 

    except Exception as db_err:
        logger.exception(f"Error updating database with Gemini file names for folder {folder_id}: {db_err}")
        # Note: Files are uploaded to Google File API but DB link failed.
        # Might need cleanup logic for Google files later.
        raise HTTPException(status_code=500, detail=f"Failed to update database with file references: {str(db_err)}")

    # 3. Update and Save Context (includes gemini_file_names)
    try:
        tutor_context.uploaded_file_paths = [fname for fname, _ in uploaded_file_details] # Store original names if needed
        logger.info("Explicitly saving context with Gemini file names BEFORE analysis.") 
        success = await _update_context_in_db(session_id, user.id, tutor_context, supabase)
        if not success:
            logger.error(f"Failed to persist context update for session {session_id} before analysis.")
    except Exception as ctx_err:
        logger.exception(f"Error updating/saving context before analysis for session {session_id}: {ctx_err}")

    # 4. Clean up temp files 
    logger.debug(f"Cleaning up temporary files for session {session_id}")
    for path in temp_file_paths_to_clean:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as rm_err:
                 logger.warning(f"Failed to remove temporary file {path}: {rm_err}")

    # 5. Trigger analysis synchronously (Now potentially uses Gemini on the file URIs)
    # NOTE: The analyze_documents function needs to be updated to use the Gemini File API names from context
    analysis_status = "pending"
    try:
        logger.info(f"Starting synchronous analysis for session {session_id} using Gemini file names...")
        # Pass the updated context object
        analysis_result: Optional[AnalysisResult] = await analyze_documents(
            context=tutor_context, # Pass context containing gemini_file_names
            supabase=supabase 
        )
        # ... (rest of analysis handling: update context, set status, update message)
        if analysis_result:
            analysis_status = "completed"
            tutor_context.analysis_result = analysis_result
            logger.info(f"Analysis completed for session {session_id}. Persisting context with analysis result.")
            # Persist context again after analysis
            success = await _update_context_in_db(session_id, user.id, tutor_context, supabase)
            if not success:
                logger.error(f"Failed to persist context update for session {session_id} after analysis.")
            message += "Analysis completed."
        else:
            message += "Analysis completed but no results were generated."
            analysis_status = "completed_empty"

    except Exception as analysis_err:
        logger.exception(f"Error during document analysis execution for session {session_id}: {analysis_err}")
        message += f"Analysis trigger failed: {str(analysis_err)}" # Append specific error
        analysis_status = "failed"
        # The function will now continue to the response construction below

    # Construct and return the final response (Do this regardless of analysis outcome)
    try:
        # Extract original filenames from the details list to ensure availability
        original_filenames = [fname for fname, _ in uploaded_file_details]
        
        response_payload = DocumentUploadResponse(
            files_received=original_filenames, # Use the extracted list of original filenames
            analysis_status=analysis_status, # Will be 'failed' if analysis exception occurred
            message=message # Will contain error info if analysis exception occurred
        )
        logger.info(f"Constructed DocumentUploadResponse for session {session_id}: Status={analysis_status}")
        return response_payload
    except Exception as validation_error:
        # This specifically catches errors during Pydantic model creation
        logger.exception(f"Pydantic validation failed for DocumentUploadResponse on session {session_id}: {validation_error}")
        # Return a generic server error if response creation fails
        raise HTTPException(status_code=500, detail=f"Internal server error creating response: {str(validation_error)}")

@router.get(
    "/sessions/{session_id}/analysis",
    response_model=AnalysisResponse,
    summary="Get Document Analysis Results",
    tags=["Tutoring Workflow"]
)
async def get_session_analysis_results(
    session_id: UUID, # Expect UUID
    tutor_context: TutorContext = Depends(get_tutor_context) # Use parsed context
):
    """Retrieves the results of the document analysis for the session."""
    analysis_obj = tutor_context.analysis_result # Access directly from context
    if analysis_obj:
        try:
            return AnalysisResponse(status="completed", analysis=analysis_obj)
        except Exception as e:
             return AnalysisResponse(status="error", error=f"Failed to parse analysis data: {e}")
    else:
        return AnalysisResponse(status="not_found", analysis=None)

@router.get(
    "/sessions/{session_id}/lesson",
    response_model=Optional[LessonContent], # Allow null if not generated yet
    dependencies=[Depends(verify_token)], # Add auth dependency
    summary="Retrieve Generated Lesson Content",
    tags=["Tutoring Workflow"]
)
async def get_session_lesson_content(session_id: UUID, tutor_context: TutorContext = Depends(get_tutor_context)):
    """Retrieves the generated lesson content for the session."""
    logger = get_session_logger(session_id)
    # Fetch lesson content directly from the parsed context
    content_data = tutor_context.lesson_content # Example assuming it's stored directly

    if not content_data:
        logger.log_error("GetLessonContent", f"Lesson content not found in session state for {session_id}")
        # Return None or 404 - let's return None first to see how frontend handles it
        # raise HTTPException(status_code=404, detail="Lesson content not yet generated or available.")
        return None # Or return an empty LessonContent structure if preferred

    try:
        # Assuming content_data is stored as a dict (from model_dump())
        lesson_content = content_data # If it's already parsed by TutorContext
        return lesson_content
    except Exception as e:
         logger.log_error("GetLessonContentParse", f"Failed to parse stored lesson content: {e}")
         raise HTTPException(status_code=500, detail="Failed to retrieve/parse lesson content.")

@router.get(
    "/sessions/{session_id}/quiz",
    response_model=Optional[Quiz], # Return Quiz or null
    dependencies=[Depends(verify_token)], # Add auth dependency
    summary="Retrieve Generated Quiz",
    tags=["Tutoring Workflow"]
)
async def get_session_quiz(session_id: UUID, tutor_context: TutorContext = Depends(get_tutor_context)):
    """Retrieves the generated quiz for the session."""
    logger = get_session_logger(session_id)
    quiz_data = tutor_context.quiz # Get 'quiz' from context object if stored there

    if not quiz_data:
        logger.log_error("GetQuiz", f"Quiz not found in session state for {session_id}")
        # Return None - frontend useEffect should handle retries/errors
        return None

    try:
        # Assuming quiz_data is stored as a dict (from model_dump())
        quiz = quiz_data # If already parsed by TutorContext
        return quiz
    except Exception as e:
         logger.log_error("GetQuizParse", f"Failed to parse stored quiz: {e}")
         raise HTTPException(status_code=500, detail="Failed to retrieve/parse quiz.")

@router.post(
    "/sessions/{session_id}/log/mini-quiz",
    status_code=204, # No content to return
    dependencies=[Depends(verify_token)], # Add auth dependency
    summary="Log Mini-Quiz Attempt",
    tags=["Logging"]
)
async def log_mini_quiz_event(
    session_id: UUID, # Expect UUID
    attempt_data: MiniQuizLogData = Body(...), # Removed request: Request, not needed if just logging
    tutor_context: TutorContext = Depends(get_tutor_context) # Use context to ensure session exists for user
):
    """Logs a user's attempt on an in-lesson mini-quiz question."""
    logger = get_session_logger(session_id)

    # You can expand this: store in session state, DB, etc.
    # For now, just log it using the TutorOutputLogger
    try:
        logger.log_mini_quiz_attempt(
            question=attempt_data.question,
            selected_option=attempt_data.selectedOption,
            correct_option=attempt_data.correctOption,
            is_correct=attempt_data.isCorrect
        )
        # Maybe append to a list in the session state if you want to access it later
        # mini_quiz_attempts = session.get("mini_quiz_attempts", [])
        # mini_quiz_attempts.append(attempt_data.model_dump())
        # session_manager.update_session(session_id, {"mini_quiz_attempts": mini_quiz_attempts})
        return # FastAPI handles 204 No Content
    except Exception as e:
        logger.log_error("LogMiniQuiz", f"Failed to log mini-quiz attempt: {e}")
        # Don't fail the request for logging, but maybe log internally
        # Consider returning 500 if logging is critical
        return

# --- Define other missing endpoints if needed (e.g., /log/summary) ---
class UserSummaryLogData(BaseModel):
    section: str
    topic: str
    summary: str

@router.post(
    "/sessions/{session_id}/log/summary",
    status_code=204, # No content to return
    dependencies=[Depends(verify_token)], # Add auth dependency
    summary="Log User Summary Attempt",
    tags=["Logging"]
)
async def log_user_summary_event(
    session_id: UUID, # Expect UUID
    summary_data: UserSummaryLogData = Body(...),
    tutor_context: TutorContext = Depends(get_tutor_context) # Use context to ensure session exists
):
    """Logs a user's summary attempt during the lesson."""
    logger = get_session_logger(session_id)
    try:
        logger.log_user_summary(
            section_title=summary_data.section,
            topic=summary_data.topic,
            summary_text=summary_data.summary
        )
        return
    except Exception as e:
        logger.log_error("LogUserSummary", f"Failed to log user summary: {e}")
        return

# --- New Interaction Endpoint ---
@router.post(
    "/sessions/{session_id}/interact",
    response_model=InteractionResponseData,
    dependencies=[Depends(verify_token)],
    summary="Interact with the AI Tutor",
    tags=["Tutoring Workflow"]
)
async def interact_with_tutor(
    session_id: UUID,
    interaction_input: InteractionRequestData, # Expects type: 'start' or 'message'
    request: Request, # Added request dependency
    user: User = Depends(verify_token),
    supabase: Client = Depends(get_supabase_client),
    session_service: SupabaseSessionService = Depends(get_session_service),
    # Get logger dependency if configured
    logger_dep: TutorOutputLogger = Depends(get_tutor_output_logger)
) -> InteractionResponseData:
    """Handles user interaction, initiating or continuing the agent workflow."""
    log_prefix = f"[Interact Session: {session_id}]"
    logger.info(f"{log_prefix} === Endpoint Start ===")
    logger.info(f"{log_prefix} Received input type='{interaction_input.type}', data={interaction_input.data}")

    # --- 1. Get Context & Initialize --- #
    try:
        tutor_context = await get_tutor_context(session_id, request, session_service)
        if not tutor_context:
            logger.error(f"{log_prefix} Context not found or failed to load.")
            raise HTTPException(status_code=404, detail="Session context not found.")
        logger.info(f"{log_prefix} Context loaded successfully.")
    except HTTPException as e:
        logger.error(f"{log_prefix} Error loading context: {e.detail}")
        raise e # Re-raise FastAPI HTTPException
    except Exception as e:
        logger.exception(f"{log_prefix} Unexpected error loading context.")
        raise HTTPException(status_code=500, detail="Internal server error loading session.")

    # --- Initialize ADK Runner --- #
    try:
        orchestrator_agent = create_orchestrator_agent() # Create the main orchestrator
        adk_runner = Runner("ai_tutor", orchestrator_agent, session_service)
        run_config = RunConfig(workflow_name="Tutor_Interaction", group_id=str(session_id))
        logger.info(f"{log_prefix} ADK Runner initialized.")
    except Exception as e:
        logger.exception(f"{log_prefix} Failed to initialize ADK agent or runner.")
        raise HTTPException(status_code=500, detail="Internal error initializing agent.")

    # --- 2. Prepare ADK Input --- #
    new_message_content = None
    if interaction_input.type == 'start' and interaction_input.data is None:
        # Initial message often handled by agent instruction or first tool call
        # Sending a generic start message might be useful sometimes.
        new_message_content = "Start the tutoring session."
        logger.info(f"{log_prefix} Preparing generic 'start' message.")
    elif interaction_input.type == 'message' and isinstance(interaction_input.data, dict) and 'text' in interaction_input.data:
        new_message_content = interaction_input.data['text']
        logger.info(f"{log_prefix} Preparing user message: '{new_message_content[:100]}...'")
    else:
        logger.warning(f"{log_prefix} Invalid interaction input type/data: {interaction_input}. Cannot proceed.")
        raise HTTPException(status_code=400, detail="Invalid interaction input for this endpoint.")

    # --- 3. Run ADK Agent --- #
    final_response_data: Optional[TutorInteractionResponse] = None
    last_agent_event: Optional[Event] = None
    question_for_user: Optional[QuizQuestion] = None
    paused_tool_call_id: Optional[str] = None # Store the ID needed for resume
    final_context: TutorContext = tutor_context # Initialize with initial context

    try:
        logger.info(f"{log_prefix} Calling ADK Runner run_async...")
        async for event in adk_runner.run_async(
            user_id=str(user.id),
            session_id=str(session_id),
            new_message=new_message_content,
            run_config=run_config
        ):
            # Improved Event Logging
            event_type = "Unknown"
            event_details = ""
            if event.content:
                if event.content.parts:
                    part = event.content.parts[0]
                    if part.text:
                        event_type = "Text"
                        event_details = f": '{part.text[:100]}...'"
                    elif part.function_call:
                        event_type = "Tool Call"
                        event_details = f": {part.function_call.name} (Args: {str(part.function_call.args)[:100]}...)"
                    elif part.function_response:
                        event_type = "Tool Response"
                        event_details = f": {part.function_response.name} (Result: {str(part.function_response.response)[:100]}...)"
                elif event.content.role == "model" and not event.content.parts:
                    event_type = "Model Content (Empty?)"
            elif event.actions:
                 event_type = "Action"
                 event_details = f": {event.actions}"

            logger.info(f"{log_prefix} Event Received: ID={event.id}, Author={event.author}, Type={event_type}{event_details}")

            logger_dep.log_orchestrator_output(event.content) # Log content using dependency
            last_agent_event = event # Keep track of the last event processed

            # --- Check for Pause/Input Request --- #
            if event.actions and event.actions.custom_action:
                 custom_action = event.actions.custom_action
                 if isinstance(custom_action, dict) and custom_action.get("type") == "WAIT_FOR_USER_INPUT":
                     # Extract question and paused_tool_call_id from custom_action
                     question_data = custom_action.get("details", {}).get("question")
                     paused_id = custom_action.get("details", {}).get("paused_tool_call_id")

                     if question_data and paused_id:
                         try:
                             question_for_user = QuizQuestion.model_validate(question_data)
                             paused_tool_call_id = paused_id
                             logger.info(f"{log_prefix} Pausing for user input. Question: '{question_for_user.question[:50]}...', Pause ID: {paused_tool_call_id}")
                             break # Exit the loop, we need user input
                         except ValidationError as q_val_err:
                             logger.error(f"{log_prefix} Failed to validate question data from custom action: {q_val_err}")
                             # Don't break, maybe the agent can recover or send a text response
                     else:
                         logger.warning(f"{log_prefix} Received WAIT_FOR_USER_INPUT action but missing question or paused_id in details.")

    except Exception as agent_err:
        logger.exception(f"{log_prefix} Exception during ADK agent execution: {agent_err}")
        raise HTTPException(status_code=500, detail="Internal error during agent execution.")

    # --- 4. Process Result & Update Final Context --- #
    logger.info(f"{log_prefix} Agent run finished or paused. Processing results...")
    try:
        # Fetch the latest state after the run/pause using the established dependency logic
        final_context = await get_tutor_context(session_id, request, session_service)
        logger.info(f"{log_prefix} Fetched final context state.")

        # Update context with pause details if needed
        if paused_tool_call_id and question_for_user: # Update only if we actually paused for a question
            if final_context.user_model_state.pending_interaction_type != 'checking_question':
                final_context.user_model_state.pending_interaction_type = 'checking_question'
                final_context.user_model_state.pending_interaction_details = {"paused_tool_call_id": paused_tool_call_id}
                # Persist this update immediately so /answer can retrieve it
                saved = await _update_context_in_db(session_id, user.id, final_context, supabase)
                if saved:
                    logger.info(f"{log_prefix} Stored pause state (paused_tool_call_id={paused_tool_call_id}) in context DB.")
                else:
                    logger.error(f"{log_prefix} Failed to store pause state in context DB!")
                    # Raise error? Or rely on next request potentially fixing it?
                    raise HTTPException(status_code=500, detail="Failed to save session pause state.")
        else:
             logger.info(f"{log_prefix} No pause detected or pause info already set.")

    except Exception as context_err:
        logger.exception(f"{log_prefix} Error fetching or updating final context: {context_err}")
        # If context fetch fails, we can't reliably return state. Return error.
        raise HTTPException(status_code=500, detail="Internal error retrieving final session state.")

    # --- 5. Format Response --- #
    logger.info(f"{log_prefix} Formatting API response...")
    # --- Logic to handle the PAUSED state --- #
    if question_for_user and paused_tool_call_id:
        # We broke the loop because we need user input
        final_response_data = QuestionResponse(
            response_type="question",
            question=question_for_user,
            topic=question_for_user.related_section or "General Knowledge" # Use related section if available
        )
        logger.info(f"{log_prefix} Responding with QuestionResponse. Session is now paused.")

    # --- Logic to handle the COMPLETED state (no pause detected) --- #
    elif last_agent_event and last_agent_event.content:
        logger.info(f"{log_prefix} Run completed normally. Processing final event from agent '{last_agent_event.author}'.")
        # Process the event content to get the core payload
        processed_payload = _process_adk_event_for_api(last_agent_event, logger_dep) # Use logger dependency
        if isinstance(processed_payload, ErrorResponse):
             final_response_data = processed_payload
             logger.error(f"{log_prefix} Orchestrator returned an error payload: {processed_payload.message}")
        elif isinstance(processed_payload, TutorInteractionResponse):
             final_response_data = processed_payload
             logger.info(f"{log_prefix} Orchestrator returned TutorInteractionResponse type: {final_response_data.response_type}")
        else: # Fallback if helper returned unexpected type
             fallback_text = str(processed_payload) # Convert to string as fallback
             logger.warning(f"{log_prefix} Orchestrator returned unexpected payload type. Falling back to MessageResponse. Payload: {fallback_text[:100]}...")
             final_response_data = MessageResponse(response_type="message", text=fallback_text)

    else:
        # No event received or last event had no content
        error_msg = f"Agent interaction finished without a final response event."
        logger.error(f"{log_prefix} Error: {error_msg}")
        final_response_data = ErrorResponse(
            error=error_msg,
            message="There was an internal error processing your request."
        )

    # Ensure final_context is available (should be due to error handling above)
    if not final_context:
         logger.error(f"{log_prefix} CRITICAL: Final context is unexpectedly None before returning response.")
         final_context = tutor_context # Use initial as last resort, likely stale

    # Return the structured response
    response = InteractionResponseData(
        content_type=getattr(final_response_data, 'response_type', 'error'), # Safely get type
        data=final_response_data, # Send the response from the final agent run
        user_model_state=final_context.user_model_state # Send final updated state
    )
    logger.info(f"{log_prefix} === Endpoint End === Returning {response.content_type}")
    return response

# --- Endpoint to handle user answers to questions --- #
@router.post(
    "/sessions/{session_id}/answer",
    response_model=InteractionResponseData,
    summary="Submit an answer to a pending question."
)
async def answer_tutor_question(
    session_id: UUID,
    interaction_input: InteractionRequestData, # Expects type: 'answer'
    request: Request,
    user: User = Depends(verify_token),
    supabase: Client = Depends(get_supabase_client),
    session_service: SupabaseSessionService = Depends(get_session_service),
    logger_dep: TutorOutputLogger = Depends(get_tutor_output_logger)
) -> InteractionResponseData:
    """Handles the user submitting an answer to a question posed by the agent."""
    log_prefix = f"[Answer Session: {session_id}]"
    logger.info(f"{log_prefix} === Endpoint Start ===")
    logger.info(f"{log_prefix} Received input type='{interaction_input.type}', data={interaction_input.data}")

    # --- 1. Validate Input & Retrieve Context/Pause State --- #
    if interaction_input.type != 'answer' or not isinstance(interaction_input.data, dict) or 'answer_index' not in interaction_input.data:
        logger.error(f"{log_prefix} Invalid input for /answer endpoint. Expected type='answer' and data={{'answer_index': int}}.")
        raise HTTPException(status_code=400, detail="Invalid input type or data for /answer endpoint. Expected type='answer' and data={'answer_index': number}.")

    try:
        tutor_context = await get_tutor_context(session_id, request, session_service)
        if not tutor_context:
            logger.error(f"{log_prefix} Context not found or failed to load.")
            raise HTTPException(status_code=404, detail="Session context not found.")
        logger.info(f"{log_prefix} Context loaded successfully.")
    except HTTPException as e:
        raise e # Re-raise known exceptions
    except Exception as e:
        logger.exception(f"{log_prefix} Unexpected error loading context.")
        raise HTTPException(status_code=500, detail="Internal server error loading session.")

    # --- Retrieve details about the paused interaction --- #
    if tutor_context.user_model_state.pending_interaction_type != 'checking_question' or \
       not tutor_context.user_model_state.pending_interaction_details or \
       'paused_tool_call_id' not in tutor_context.user_model_state.pending_interaction_details:
        pending_type = tutor_context.user_model_state.pending_interaction_type
        logger.warning(f"{log_prefix} Received answer, but no valid pending 'checking_question' interaction found in context. Pending type: {pending_type}")
        raise HTTPException(status_code=400, detail=f"No pending question found for this session {session_id}. Current pending type: {pending_type}")

    paused_tool_call_id = tutor_context.user_model_state.pending_interaction_details['paused_tool_call_id']
    user_answer_index = interaction_input.data['answer_index']
    logger.info(f"{log_prefix} Found pending interaction. Pause ID: {paused_tool_call_id}, User Answer Index: {user_answer_index}")

    # --- Clear the pending state immediately (before agent run) --- #
    logger.info(f"{log_prefix} Clearing pending interaction state in context...")
    tutor_context.user_model_state.pending_interaction_type = None
    tutor_context.user_model_state.pending_interaction_details = None
    # Persist this change immediately
    saved = await _update_context_in_db(session_id, user.id, tutor_context, supabase)
    if saved:
         logger.info(f"{log_prefix} Cleared and saved pending state in context DB.")
    else:
         logger.error(f"{log_prefix} Failed to clear and save pending state in context DB!")
         raise HTTPException(status_code=500, detail="Failed to update session state before resuming.")

    # --- 2. Prepare Resume Event --- #
    # Create the FunctionResponse event that the long-running tool expects
    logger.info(f"{log_prefix} Creating FunctionResponse event for tool 'ask_user_question_and_get_answer' with pause ID {paused_tool_call_id}")
    resume_event = Event(
        author="user", # Or maybe system? Let's use user for now
        content=adk_types.Content(
            parts=[
                # ADK expects the response payload here
                adk_types.Part.from_dict({
                     'function_response': {
                         'name': 'ask_user_question_and_get_answer',
                         # 'id': paused_tool_call_id, # ID might not be needed here, ADK matches by name/call context
                         'response': {'answer_index': user_answer_index}
                     }
                 })
            ],
            # tool_code_parts is for *requests*, not responses AFAIK
            # tool_code_parts=[adk_types.FunctionResponse(name='ask_user_question_and_get_answer', id=paused_tool_call_id, response={'answer_index': user_answer_index})], # Pass ID here for ADK matching
        ),
        invocation_id=tutor_context.last_interaction_summary or f"resume_{session_id}", # Find appropriate invocation ID? Use last one?
        # Need to manually set ID and timestamp?
        id=Event.new_id(),
        timestamp=time.time(),
    )
    # Append this synthesized event to the session history
    try:
        logger.info(f"{log_prefix} Appending resume event to session service.")
        session_service.append_event(session=tutor_context, event=resume_event)
        logger.info(f"{log_prefix} Resume event appended.")
    except Exception as append_err:
        logger.exception(f"{log_prefix} Failed to append resume event to session service: {append_err}")
        raise HTTPException(status_code=500, detail="Failed to record resume event.")


    # --- 3. Resume the ADK Runner --- #
    logger.info(f"{log_prefix} Resuming ADK Runner...")
    final_response_data_after_resume: Optional[TutorInteractionResponse] = None
    last_agent_event_after_resume: Optional[Event] = None
    question_after_resume: Optional[QuizQuestion] = None
    paused_id_after_resume: Optional[str] = None
    final_context_after_resume: TutorContext = tutor_context # Start with current context

    try:
        # Re-initialize runner and run_async.
        # ADK's Runner should pick up from the paused state using the session history.
        orchestrator_agent = create_orchestrator_agent()
        adk_runner = Runner("ai_tutor", orchestrator_agent, session_service)
        run_config = RunConfig(workflow_name="Tutor_Interaction_Resume", group_id=str(session_id))
        logger.info(f"{log_prefix} ADK Runner re-initialized for resume.")

        # Run async again. It should process the appended resume_event and continue.
        logger.info(f"{log_prefix} Calling ADK Runner run_async for resume...")
        async for event in adk_runner.run_async(
            user_id=str(user.id),
            session_id=str(session_id),
            new_message=None, # Crucial: No new message, resume from history
            run_config=run_config
        ):
            # Improved Event Logging
            event_type = "Unknown"
            event_details = ""
            if event.content:
                if event.content.parts:
                    part = event.content.parts[0]
                    if part.text:
                        event_type = "Text"
                        event_details = f": '{part.text[:100]}...'"
                    elif part.function_call:
                        event_type = "Tool Call"
                        event_details = f": {part.function_call.name} (Args: {str(part.function_call.args)[:100]}...)"
                    elif part.function_response:
                        event_type = "Tool Response"
                        event_details = f": {part.function_response.name} (Result: {str(part.function_response.response)[:100]}...)"
                elif event.content.role == "model" and not event.content.parts:
                    event_type = "Model Content (Empty?)"
            elif event.actions:
                 event_type = "Action"
                 event_details = f": {event.actions}"

            logger.info(f"{log_prefix} Event Received (Resume): ID={event.id}, Author={event.author}, Type={event_type}{event_details}")

            logger_dep.log_orchestrator_output(event.content)
            last_agent_event_after_resume = event

            # Check if it paused AGAIN
            if event.actions and event.actions.custom_action:
                custom_action = event.actions.custom_action
                if isinstance(custom_action, dict) and custom_action.get("type") == "WAIT_FOR_USER_INPUT":
                    question_data = custom_action.get("details", {}).get("question")
                    paused_id = custom_action.get("details", {}).get("paused_tool_call_id")
                    if question_data and paused_id:
                        try:
                            question_after_resume = QuizQuestion.model_validate(question_data)
                            paused_id_after_resume = paused_id
                            logger.info(f"{log_prefix} Paused AGAIN after resume. Question: '{question_after_resume.question[:50]}...', New Pause ID: {paused_id_after_resume}")
                            break # Exit loop
                        except ValidationError as q_val_err:
                            logger.error(f"{log_prefix} Failed to validate question data from custom action (after resume): {q_val_err}")
                    else:
                        logger.warning(f"{log_prefix} Received WAIT_FOR_USER_INPUT action (after resume) but missing details.")

    except Exception as agent_err:
        logger.exception(f"{log_prefix} Exception during ADK agent execution (resume): {agent_err}")
        raise HTTPException(status_code=500, detail="Internal error during agent execution after answer.")

    # --- 4. Process Result & Update Final Context (After Resume) --- #
    logger.info(f"{log_prefix} Agent run (resume) finished or paused again. Processing results...")
    try:
        # Reload latest context AFTER the resume run
        final_context_after_resume = await get_tutor_context(session_id, request, session_service)
        logger.info(f"{log_prefix} Fetched final context state after resume.")

        # Update context with pause details if needed (moved for clarity)
        if paused_id_after_resume and question_after_resume:
            if final_context_after_resume.user_model_state.pending_interaction_type != 'checking_question':
                final_context_after_resume.user_model_state.pending_interaction_type = 'checking_question'
                final_context_after_resume.user_model_state.pending_interaction_details = {"paused_tool_call_id": paused_id_after_resume}
                saved = await _update_context_in_db(session_id, user.id, final_context_after_resume, supabase)
                if saved:
                    logger.info(f"{log_prefix} Stored new pause state (paused_tool_call_id={paused_id_after_resume}) in context DB after resume.")
                else:
                    logger.error(f"{log_prefix} Failed to store new pause state in context DB after resume!")
                    raise HTTPException(status_code=500, detail="Failed to save session pause state after resume.")
        else:
            logger.info(f"{log_prefix} No new pause detected after resume.")

    except Exception as context_err:
        logger.exception(f"{log_prefix} Error fetching or updating final context after resume: {context_err}")
        raise HTTPException(status_code=500, detail="Internal error retrieving final session state after resume.")

    # --- 5. Format Response (After Resume) --- #
    logger.info(f"{log_prefix} Formatting API response after resume...")
    # --- Logic to handle PAUSED AGAIN state --- #
    if question_after_resume and paused_id_after_resume:
        final_response_data_after_resume = QuestionResponse(
            response_type="question",
            question=question_after_resume,
            topic=question_after_resume.related_section or "General Knowledge"
        )
        logger.info(f"{log_prefix} Responding with NEW QuestionResponse. Session paused again.")

    # --- Logic to handle COMPLETED state (after resume) --- #
    elif last_agent_event_after_resume and last_agent_event_after_resume.content:
        logger.info(f"{log_prefix} Run completed normally after resume. Processing final event from agent '{last_agent_event_after_resume.author}'.")
        processed_payload = _process_adk_event_for_api(last_agent_event_after_resume, logger_dep)
        if isinstance(processed_payload, ErrorResponse):
            final_response_data_after_resume = processed_payload
            logger.error(f"{log_prefix} Orchestrator returned an error payload after resume: {processed_payload.message}")
        elif isinstance(processed_payload, TutorInteractionResponse):
            final_response_data_after_resume = processed_payload
            logger.info(f"{log_prefix} Orchestrator returned TutorInteractionResponse type after resume: {final_response_data_after_resume.response_type}")
        else:
            fallback_text = str(processed_payload)
            logger.warning(f"{log_prefix} Orchestrator returned unexpected payload type after resume. Falling back to MessageResponse. Payload: {fallback_text[:100]}...")
            final_response_data_after_resume = MessageResponse(response_type="message", text=fallback_text)

    else:
        # No event or content after resume
        error_msg = f"Agent interaction finished after resume without a final response event."
        logger.error(f"{log_prefix} Error: {error_msg}")
        final_response_data_after_resume = ErrorResponse(error=error_msg, message="Internal processing error after submitting answer.")

    # Ensure final_context_after_resume exists
    if not final_context_after_resume:
        logger.error(f"{log_prefix} CRITICAL: Final context after resume is unexpectedly None before returning response.")
        final_context_after_resume = tutor_context # Fallback, likely stale

    response = InteractionResponseData(
        content_type=getattr(final_response_data_after_resume, 'response_type', 'error'),
        data=final_response_data_after_resume,
        user_model_state=final_context_after_resume.user_model_state # Return the LATEST state
    )
    logger.info(f"{log_prefix} === Endpoint End === Returning {response.content_type}")
    return response

# --- Helper function to process ADK event content into API response payload (not InteractionResponseData) --- #
def _process_adk_event_for_api(event: Event, logger: TutorOutputLogger) -> Union[TutorInteractionResponse, dict]:
    """Processes an ADK event and formats it for the API response payload.

    Returns the specific Pydantic response model (ExplanationResponse, etc.) or a dict for errors/fallbacks.
    """
    log_prefix = f"[_process_adk_event_for_api Event: {event.id}]"
    logger.info(f"{log_prefix} Processing event from author '{event.author}'...")
    response_payload: Union[TutorInteractionResponse, dict] = MessageResponse(response_type="message", text="Processing...") # Default/fallback

    # Handle error events first
    if event.error_code or event.error_message:
        error_text = event.error_message or f"Agent error occurred (Code: {event.error_code})"
        response_payload = ErrorResponse(response_type="error", message=error_text, error=f"AGENT_ERROR_{event.error_code}")
        logger.log_error("AgentEventError", f"Error in event {event.id}: {error_text}")
        logger.error(f"{log_prefix} Processed as ErrorResponse.")
        return response_payload

    # Process event content if available
    if event.content:
        logger.info(f"{log_prefix} Event has content. Processing parts...")
        for part in event.content.parts:
            # 1. Text Part
            if part.text:
                text = part.text.strip()
                if text: # Ensure there's text content
                    logger.info(f"{log_prefix} Processing text part: '{text[:100]}...'")
                    response_payload = MessageResponse(response_type="message", text=text)
                    logger.info(f"{log_prefix} Processed as MessageResponse.")
                    return response_payload # Text content takes precedence for direct response
                else:
                    logger.info(f"{log_prefix} Text part is empty.")

            # 2. Function Call Part (Agent requesting tool use)
            elif part.function_call:
                tool_name = part.function_call.name
                tool_args = part.function_call.args or {}
                logger.info(f"{log_prefix} Processing function call part: {tool_name}")

                # Check if this is the specific call to ask the user a question
                if tool_name == "ask_user_question_and_get_answer":
                    try:
                        # Args should match QuizQuestion schema
                        question = QuizQuestion.model_validate(tool_args)
                        response_payload = QuestionResponse(
                                response_type="question",
                                question=question,
                            topic=tool_args.get("topic", "Unknown Topic") # Assuming topic is passed
                        )
                        logger.info(f"{log_prefix} Processed as QuestionResponse for user.")
                        # Note: The actual pause happens in the main loop based on custom_action
                        # This function just formats what *would* be sent if the tool call itself was the final API response.
                        return response_payload
                    except (ValidationError, KeyError) as e:
                        logger.error(f"{log_prefix} Validation failed for question tool args: {e}")
                        response_payload = ErrorResponse(response_type="error", message=f"Invalid data from question tool: {e}", error="TOOL_ARG_VALIDATION_ERROR")
                        logger.info(f"{log_prefix} Processed as ErrorResponse due to validation failure.")
                        return response_payload
                else:
                    # Other tool calls are typically intermediate steps
                    logger.info(f"{log_prefix} Tool call '{tool_name}' is intermediate, not formatting API response.")
                    # Keep default payload or indicate processing
                    response_payload = {"status": "processing_tool_call", "tool_name": tool_name}
                    # Continue processing other parts if any (though usually only one part)
                    continue

            # 3. Function Response Part (Result from a tool execution)
            elif part.function_response:
                tool_name = part.function_response.name
                tool_response = part.function_response.response if part.function_response.response is not None else {}
                logger.info(f"{log_prefix} Processing function response for tool: {tool_name}")

                # Check if this is the result from the evaluation tool
                if tool_name == "call_quiz_teacher_evaluate":
                    try:
                        feedback_item = QuizFeedbackItem.model_validate(tool_response)
                        response_payload = FeedbackResponse(
                            response_type="feedback",
                            feedback=feedback_item,
                            topic=tool_response.get("topic", "Feedback Topic") # Assuming topic might be in response
                        )
                        logger.info(f"{log_prefix} Processed as FeedbackResponse.")
                        return response_payload
                    except (ValidationError, KeyError) as e:
                        logger.error(f"{log_prefix} Validation failed for feedback tool response: {e}")
                        response_payload = ErrorResponse(response_type="error", message=f"Invalid data from feedback tool: {e}", error="TOOL_RESPONSE_VALIDATION_ERROR")
                        logger.info(f"{log_prefix} Processed as ErrorResponse due to validation failure.")
                        return response_payload
                else:
                    # Other tool responses usually update state internally, don't form final API response
                    logger.info(f"{log_prefix} Tool response for '{tool_name}' is intermediate, not formatting API response.")
                    response_payload = {"status": "processing_tool_response", "tool_name": tool_name}
                    # Continue processing other parts if any
                    continue
            else:
                logger.info(f"{log_prefix} Unknown part type encountered.")
    else:
        logger.info(f"{log_prefix} Event has no content.")

    # Handle case where no specific API response was generated (e.g., only intermediate steps)
    if isinstance(response_payload, dict) and ("status" in response_payload):
        logger.warning(f"{log_prefix} Event processing finished with intermediate status: {response_payload}. Returning generic message.")
        response_payload = MessageResponse(response_type="message", text="Processing complete.")
    elif not isinstance(response_payload, TutorInteractionResponse):
        logger.warning(f"{log_prefix} Event processing finished with unexpected payload type: {type(response_payload).__name__}. Returning generic message.")
        response_payload = MessageResponse(response_type="message", text="Interaction processed.")

    logger.info(f"{log_prefix} Final processed payload type: {getattr(response_payload, 'response_type', 'dict')}")
    return response_payload

# --- Helper to update context in DB (replace direct calls) ---
async def _update_context_in_db(session_id: UUID, user_id: UUID, context: TutorContext, supabase: Client) -> bool: # Added return type hint
    """Helper to persist context via Supabase client.

    Uses optimistic approach: saves the current context state.
    Returns True on success, False on failure.
    """
    log_prefix = f"[_update_context_in_db Session: {session_id}]"
    logger.info(f"{log_prefix} Attempting to save context state to DB.")
    try:
        context_dict = context.model_dump(mode='json')
        response = supabase.table("sessions").update(
            {"context_data": context_dict, "updated_at": datetime.now().isoformat()}
        ).eq("id", str(session_id)).eq("user_id", str(user_id)).execute()

        # Check response (Supabase client >= 2.0.0)
        if response.data:
            logger.info(f"{log_prefix} Context successfully saved to DB.")
            return True
        else:
            # Log potential error details if available in response
            error_details = getattr(response, 'error', None)
            logger.error(f"{log_prefix} Failed to save context to DB. No data returned. Error: {error_details}")
            return False
    except Exception as e:
        logger.exception(f"{log_prefix} Exception during context save: {e}")
        return False

# --- Retry Helper ---
async def retry_with_exponential_backoff(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 32.0,
    logger: Optional[logging.Logger] = None
) -> T:
    """Helper function to retry operations with exponential backoff.
    
    Args:
        operation: Async function to retry
        max_retries: Maximum number of retries (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 32.0)
        logger: Optional logger instance
    
    Returns:
        The result of the operation if successful
    
    Raises:
        The last exception encountered if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            return await operation()
        except google.genai.errors.ClientError as e:
            # Check if the ClientError is due to resource exhaustion by looking at the message string
            if 'RESOURCE_EXHAUSTED' in str(e):
                # *** Fix Indentation Start ***
                last_exception = e
                log_func = logger.warning if logger else print
                log_func(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue # Go to next attempt
                # *** Fix Indentation End ***
            else:
                # For other ClientErrors, raise immediately (or handle differently)
                last_exception = e
                log_func = logger.error if logger else print
                log_func(f"Non-rate-limit ClientError encountered: {e}")
                break # Exit retry loop for non-recoverable client errors
        except google_exceptions.GoogleAPIError as e: # Catch broader Google API errors if needed
            last_exception = e
            log_func = logger.error if logger else print
            log_func(f"Google API Error encountered: {e}")
            # Decide if retry is appropriate for GoogleAPIError subtypes
            break # Exit retry loop for now
        except Exception as e:
            # Catch other unexpected exceptions
            last_exception = e
            log_func = logger.error if logger else print
            log_func(f"Unexpected exception during operation: {e}")
            break # Exit retry loop for unexpected errors

    raise last_exception # Raise the last exception if all retries failed

# TODO: Implement endpoint for session analysis if needed:
# POST /sessions/{session_id}/analyze-session (Full Session Analysis) 