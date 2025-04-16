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
from ai_tutor.agents.session_analyzer_agent import analyze_teaching_session
from ai_tutor.agents.orchestrator_agent import create_orchestrator_agent
# Import models first
from ai_tutor.agents.models import (
    FocusObjective,
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis, QuizQuestion, QuizFeedbackItem, TeacherTurnResult
)
# Then import functions that might use them
from ai_tutor.agents.analyzer_agent import analyze_documents, AnalysisResult

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

    # --- Initialize ADK Runner ---
    try:
        logger.info("[Interact Session: {}] Initializing ADK Runner...".format(session_id))
        # Use keyword arguments for Runner initialization
        adk_runner = Runner(
            app_name="ai_tutor", 
            agent=create_orchestrator_agent(), 
            session_service=session_service
        )
        logger.info("[Interact Session: {}] ADK Runner initialized successfully.".format(session_id))
    except Exception as e:
        logger.exception(f"[Interact Session: {session_id}] Failed to initialize ADK agent or runner.")
        # Construct proper ErrorResponse and InteractionResponseData
        error_response = ErrorResponse(
            response_type='error', # Add required field
            error_code='AGENT_INITIALIZATION_FAILED',
            message=f"Internal server error: Failed to initialize agent. Details: {str(e)}"
        )
        return InteractionResponseData(
            content_type='error',
            data=error_response,
            user_model_state=UserModelState() # Provide default empty state
        )

    # --- 2. Prepare ADK Input --- #
    new_message_for_adk: Optional[Content] = None # Initialize as None, type hint Content
    if interaction_input.type == 'start' and interaction_input.data is None:
        # Initial message often handled by agent instruction or first tool call
        # Sending a generic start message might be useful sometimes.
        start_text = "Start the tutoring session."
        new_message_for_adk = Content(parts=[Part(text=start_text)]) # Wrap in Content/Part
        logger.info(f"{log_prefix} Preparing generic 'start' message as Content object.")
    elif interaction_input.type == 'message' and isinstance(interaction_input.data, dict) and 'text' in interaction_input.data:
        user_text = interaction_input.data['text']
        new_message_for_adk = Content(parts=[Part(text=user_text)]) # Wrap in Content/Part
        logger.info(f"{log_prefix} Preparing user message as Content object: '{user_text[:100]}...'" )
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
            new_message=new_message_for_adk # Pass the Content object
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

            # --- Check for Pause/Input Request (Corrected Logic) --- #
            function_calls = event.get_function_calls()
            if function_calls:
                for func_call in function_calls:
                    if func_call.name == "ask_user_question_and_get_answer":
                        logger.info(f"{log_prefix} Detected function call for long-running tool 'ask_user_question_and_get_answer'.")
                        # Extract question data from args
                        question_data = func_call.args
                        # Extract the ID needed for resuming (Check event.long_running_tool_ids)
                        # The ID might also be implicitly linked via event history, but storing explicitly is safer.
                        # We need *an* ID to link the answer back. Let's use the event ID for now if long_running_tool_ids isn't reliable.
                        # NOTE: ADK documentation isn't perfectly clear on the best ID to use here.
                        # Using the *event ID* of the function call request seems plausible.
                        paused_id = event.id # Use the Event ID as the identifier for the pause
                        
                        if question_data and paused_id:
                            try:
                                question_for_user = QuizQuestion.model_validate(question_data)
                                paused_tool_call_id = paused_id # Store the event ID
                                logger.info(f"{log_prefix} Pausing for user input. Question: '{question_for_user.question[:50]}...', Pause ID (Event ID): {paused_tool_call_id}")
                                break # Exit the inner loop (function calls)
                            except ValidationError as q_val_err:
                                logger.error(f"{log_prefix} Failed to validate question data from function call args: {q_val_err}")
                                # Continue processing other function calls or parts?
                        else:
                            logger.warning(f"{log_prefix} Received 'ask_user...' function call but missing args or couldn't determine pause ID.")
            
            # If we set paused_tool_call_id in the inner loop, break the outer event loop too
            if paused_tool_call_id:
                break
    except google.api_core.exceptions.ResourceExhausted as rate_limit_err:
        logger.error(f"{log_prefix} Google AI Rate Limit Error: {rate_limit_err}")
        retry_delay = "15s" # Default retry delay
        try:
            details = getattr(rate_limit_err, 'details', [])
            for detail in details:
                if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                    retry_delay = detail.get('retryDelay', retry_delay)
                    break
        except Exception:
            pass 
        error_response = ErrorResponse(
            response_type='error',
            error_code='RATE_LIMIT_EXCEEDED',
            message=f"API rate limit exceeded. Please try again in {retry_delay}."
        )
        return InteractionResponseData(
            content_type='error',
            data=error_response,
            user_model_state=tutor_context.user_model_state 
        )
    except google.genai.errors.ClientError as client_err:
        if "RESOURCE_EXHAUSTED" in str(client_err) or "429" in str(client_err):
            logger.error(f"{log_prefix} Google AI Rate Limit Error (via ClientError): {client_err}")
            retry_delay = "15s" # Default retry delay
            try:
                details = getattr(client_err, 'details', [])
                for detail in details:
                    if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                        retry_delay = detail.get('retryDelay', retry_delay)
                        break
            except Exception:
                pass 
            error_response = ErrorResponse(
                response_type='error',
                error_code='RATE_LIMIT_EXCEEDED',
                message=f"API rate limit exceeded. Please try again in {retry_delay}."
            )
            return InteractionResponseData(
                content_type='error',
                data=error_response,
                user_model_state=tutor_context.user_model_state 
            )
        else:
            logger.exception(f"{log_prefix} Non-rate-limit ClientError during ADK agent run: {client_err}")
            error_response = ErrorResponse(
                response_type='error', 
                error_code='AGENT_EXECUTION_ERROR',
                message=f"Internal server error during agent processing: {str(client_err)}"
            )
            return InteractionResponseData(
                content_type='error',
                data=error_response,
                user_model_state=tutor_context.user_model_state 
            )
    except Exception as agent_err:
        logger.exception(f"{log_prefix} Error during ADK agent run: {agent_err}")
        # Return an ErrorResponse within InteractionResponseData
        error_response = ErrorResponse(
            response_type='error', 
            error_code='AGENT_EXECUTION_ERROR',
            message=f"Internal server error during agent processing: {str(agent_err)}"
        )
        return InteractionResponseData(
            content_type='error',
            data=error_response,
            user_model_state=tutor_context.user_model_state 
        )
    # --- 4. Process Result & Update Final Context --- #
    logger.info(f"{log_prefix} Agent run finished or paused. Processing results...")
    # Fetch latest context state AFTER agent run completes
    # This ensures we capture state changes made by the agent/tools/callbacks
    try:
        # Re-fetch context to get the absolute latest state persisted by SessionService
        latest_context = await get_tutor_context(session_id, request, session_service)
        final_context = latest_context # Use the latest fetched context
        logger.info(f"{log_prefix} Fetched final context state.")
    except HTTPException as he:
        # Handle case where session might not be found after run (unlikely)
        logger.error(f"{log_prefix} Failed to fetch final context state: {he.detail}")
        # Fallback to context from before the run? Or error out?
        # Let's error out for now, as state might be inconsistent.
        raise HTTPException(status_code=500, detail="Internal server error retrieving final session state.")

    # Check if the agent paused and needs user input
    if paused_tool_call_id and question_for_user:
         logger.info(f"{log_prefix} Agent paused for user input. Storing pause details.")
         # Store pause details in the *final* context state
         final_context.pending_interaction_details = {
            "status": "paused",
            "paused_tool_call_id": paused_tool_call_id,
            "question_details": question_for_user.model_dump() # Store the question data
         }
         # Save context with pause details
         await _update_context_in_db(session_id, user.id, final_context, supabase)
         # Format response indicating pause
         response_data = QuestionResponse(
             status="requires_answer",
             message="The tutor is waiting for your answer.",
             question=question_for_user
         )
         return InteractionResponseData(status="requires_answer", data=response_data)
    else:
         logger.info(f"{log_prefix} No pause detected or pause info already set.")
         # Clear any potentially stale pending interaction if run completed normally
         if final_context.pending_interaction_details:
              logger.info(f"{log_prefix} Clearing stale pending interaction details.")
              final_context.pending_interaction_details = None
              # Save context with cleared details
              await _update_context_in_db(session_id, user.id, final_context, supabase)

    # --- Format Final Response --- #
    logger.info(f"{log_prefix} Formatting API response...")
    if not last_agent_event:
         logger.error(f"{log_prefix} Agent run finished, but no final agent event captured.")
         return InteractionResponseData(status='error', message="Internal error: Agent did not produce a final output.")

    logger.info(f"{log_prefix} Run completed normally. Processing final event from agent '{last_agent_event.author}'.")
    # Pass the output logger dependency to the processing function
    processed_payload = await retry_with_exponential_backoff(lambda: _process_adk_event_for_api(last_agent_event, output_logger=logger_dep))

    if isinstance(processed_payload, ErrorResponse):
         # For errors, content_type might not be applicable or fixed to 'error'
         # User model state might be stale, but send the latest fetched one
         return InteractionResponseData(
             content_type='error', # Explicitly set for errors
             data=processed_payload, 
             user_model_state=final_context.user_model_state
         )
    else:
         # Ensure processed_payload has response_type before accessing
         content_type = getattr(processed_payload, 'response_type', 'unknown')
         return InteractionResponseData(
             content_type=content_type, # Get type from the payload
             data=processed_payload,
             user_model_state=final_context.user_model_state # Get state from final context
         )

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
    """Receives an answer, resumes the ADK agent, and returns the next interaction."""
    log_prefix = f"[Answer Session: {session_id}]"
    logger.info(f"{log_prefix} === Endpoint Start ===")
    logger.info(f"{log_prefix} Received input: {interaction_input}")

    # --- 1. Validate Input & Context --- #
    if interaction_input.type != 'answer' or not isinstance(interaction_input.data, dict) or 'text' not in interaction_input.data:
        raise HTTPException(status_code=400, detail="Invalid input: Expected type 'answer' with 'text' in data.")

    user_answer_text = interaction_input.data['text']

    try:
        tutor_context = await get_tutor_context(session_id, request, session_service)
        logger.info(f"{log_prefix} Context loaded.")
        
        # Check if context is actually paused
        pause_details = tutor_context.pending_interaction_details
        if not pause_details or pause_details.get("status") != "paused" or not pause_details.get("paused_tool_call_id"):
            logger.warning(f"{log_prefix} Received answer, but session context is not in a valid paused state.")
            return InteractionResponseData(status='error', message="Tutor was not waiting for an answer.")
        
        paused_tool_call_id = pause_details["paused_tool_call_id"]
        logger.info(f"{log_prefix} Session is paused, expecting answer for tool call ID: {paused_tool_call_id}")
        
    except HTTPException as he:
        raise he # Re-raise validation/auth errors
    except Exception as e:
        logger.exception(f"{log_prefix} Error loading session or context: {e}")
        raise HTTPException(status_code=500, detail="Internal server error loading session.")

    # --- 2. Prepare Tool Response for ADK --- #
    # The tool response should mimic what the `ask_user_question_and_get_answer` tool
    # would normally return after getting the answer.
    # Let's assume it returns a simple dictionary with the answer text.
    tool_response_data = {
        "user_answer": user_answer_text
    }
    
    # Create the FunctionResponse part for the ADK event
    tool_response_part = FunctionResponse(
        name="ask_user_question_and_get_answer", # Name of the tool that paused
        response=tool_response_data
    )
    
    # Create the ADK Event to send back to the runner
    # We need to find the original event ID that caused the pause (stored as paused_tool_call_id)
    # and associate this response with it.
    # ADK typically handles this linking automatically if we structure the resume correctly.
    # We likely need to provide the FunctionResponse as the `new_message` to `run_async`
    # when resuming, associated with the original function call.
    
    # ADK needs a google.genai.types.Content object containing the FunctionResponse
    resume_content = Content(parts=[tool_response_part])
    
    # --- 3. Re-initialize Runner & Resume --- #
    final_response_data: Optional[TutorInteractionResponse] = None
    last_agent_event: Optional[Event] = None
    question_for_user: Optional[QuizQuestion] = None # Reset for this turn
    new_paused_tool_call_id: Optional[str] = None # Reset for this turn
    final_context: TutorContext = tutor_context # Start with current context

    try:
        # Re-initialize runner and run_async.
        # ADK's Runner should pick up from the paused state using the session history.
        adk_runner = Runner(
            app_name="ai_tutor", 
            agent=create_orchestrator_agent(), 
            session_service=session_service
        )
        logger.info(f"{log_prefix} ADK Runner re-initialized for resume.")

        logger.info(f"{log_prefix} Calling ADK Runner run_async for resume...")
        async for event in adk_runner.run_async(
            user_id=str(user.id),
            session_id=str(session_id),
            new_message=resume_content # Send the tool response as the message
        ):
            # --- Process Events (Similar to /interact loop) --- #
            event_type = "Unknown"
            event_details = ""
            if event.content:
                 if event.content.parts and event.content.parts[0].text:
                      event_type = "Text"
                      event_details = f": '{event.content.parts[0].text[:100]}...'"
            elif event.get_function_calls():
                 event_type = "Tool Call"
                 event_details = f": {event.get_function_calls()[0].name}"
            elif event.get_function_responses():
                 event_type = "Tool Response"
                 event_details = f": {event.get_function_responses()[0].name} (Result: {str(event.get_function_responses()[0].response)[:100]}...)"
            elif event.actions:
                 event_type = "Action"
                 # Simple action logging, needs refinement
                 actions_summary = []
                 if event.actions.state_delta: actions_summary.append("state_delta")
                 if event.actions.artifact_delta: actions_summary.append("artifact_delta")
                 if event.actions.transfer_to_agent: actions_summary.append(f"transfer({event.actions.transfer_to_agent})")
                 if event.actions.escalate: actions_summary.append("escalate")
                 event_details = f": {', '.join(actions_summary)}"

            logger.info(f"{log_prefix} Event Received: ID={event.id}, Author={event.author}, Type={event_type}{event_details}")

            logger_dep.log_orchestrator_output(event.content) # Log content using dependency
            last_agent_event = event # Keep track of the last event processed

            # --- Check for NEW Pause/Input Request --- #
            function_calls = event.get_function_calls()
            if function_calls:
                for func_call in function_calls:
                    if func_call.name == "ask_user_question_and_get_answer":
                        logger.info(f"{log_prefix} Detected NEW function call for 'ask_user_question_and_get_answer'.")
                        question_data = func_call.args
                        paused_id = event.id # Use new event ID for the new pause
                        
                        if question_data and paused_id:
                            try:
                                question_for_user = QuizQuestion.model_validate(question_data)
                                new_paused_tool_call_id = paused_id # Store the *new* pause ID
                                logger.info(f"{log_prefix} Agent is pausing AGAIN for user input. Question: '{question_for_user.question[:50]}...', Pause ID: {new_paused_tool_call_id}")
                                break # Exit inner loop
                            except ValidationError as q_val_err:
                                logger.error(f"{log_prefix} Failed to validate question data from NEW function call: {q_val_err}")
                        else:
                            logger.warning(f"{log_prefix} Received NEW 'ask_user...' function call but missing args or pause ID.")
            
            # If we set new_paused_tool_call_id, break the outer event loop
            if new_paused_tool_call_id:
                break
            # --- End Event Processing --- #

    except google.api_core.exceptions.ResourceExhausted as rate_limit_err:
        logger.error(f"{log_prefix} Google AI Rate Limit Error on Resume: {rate_limit_err}")
        retry_delay = "15s" # Default retry delay
        try:
            details = getattr(rate_limit_err, 'details', [])
            for detail in details:
                if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                    retry_delay = detail.get('retryDelay', retry_delay)
                    break
        except Exception:
            pass 
        error_response = ErrorResponse(
            response_type='error',
            error_code='RATE_LIMIT_EXCEEDED',
            message=f"API rate limit exceeded during resume. Please try again in {retry_delay}."
        )
        return InteractionResponseData(
            content_type='error',
            data=error_response,
            user_model_state=tutor_context.user_model_state # State before the failed resume call
        )
    except google.genai.errors.ClientError as client_err:
        if "RESOURCE_EXHAUSTED" in str(client_err) or "429" in str(client_err):
            logger.error(f"{log_prefix} Google AI Rate Limit Error (via ClientError) on Resume: {client_err}")
            retry_delay = "15s" # Default retry delay
            try:
                details = getattr(client_err, 'details', [])
                for detail in details:
                    if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                        retry_delay = detail.get('retryDelay', retry_delay)
                        break
            except Exception:
                pass 
            error_response = ErrorResponse(
                response_type='error',
                error_code='RATE_LIMIT_EXCEEDED',
                message=f"API rate limit exceeded during resume. Please try again in {retry_delay}."
            )
            return InteractionResponseData(
                content_type='error',
                data=error_response,
                user_model_state=tutor_context.user_model_state # State before the failed resume call
            )
        else:
             logger.exception(f"{log_prefix} Non-rate-limit ClientError during ADK agent resume run: {client_err}")
             error_response = ErrorResponse(
                response_type='error', 
                error_code='AGENT_EXECUTION_ERROR',
                message=f"Internal server error during agent processing on resume: {str(client_err)}"
             )
             return InteractionResponseData(
                content_type='error',
                data=error_response,
                user_model_state=tutor_context.user_model_state # State before the failed resume call
             )
    except Exception as agent_err:
        logger.exception(f"{log_prefix} Error during ADK agent resume run: {agent_err}")
        error_response = ErrorResponse(
            response_type='error', 
            error_code='AGENT_EXECUTION_ERROR',
            message=f"Internal server error during agent processing on resume: {str(agent_err)}"
        )
        return InteractionResponseData(
            content_type='error',
            data=error_response,
            user_model_state=tutor_context.user_model_state # State before the failed resume call
        )

    # --- 4. Process Final State & Response (Similar to /interact) --- #
    logger.info(f"{log_prefix} Agent resume run finished or paused again. Processing results...")
    try:
        latest_context = await get_tutor_context(session_id, request, session_service)
        final_context = latest_context
        logger.info(f"{log_prefix} Fetched final context state after resume.")
    except HTTPException as he:
        logger.error(f"{log_prefix} Failed to fetch final context state after resume: {he.detail}")
        raise HTTPException(status_code=500, detail="Internal server error retrieving final session state after resume.")

    # Update pending interaction details based on whether it paused again
    if new_paused_tool_call_id and question_for_user:
         logger.info(f"{log_prefix} Agent paused AGAIN. Storing new pause details.")
         final_context.pending_interaction_details = {
            "status": "paused",
            "paused_tool_call_id": new_paused_tool_call_id,
            "question_details": question_for_user.model_dump()
         }
         await _update_context_in_db(session_id, user.id, final_context, supabase)
         response_data = QuestionResponse(
             status="requires_answer",
             message="The tutor has another question.",
             question=question_for_user
         )
         return InteractionResponseData(status="requires_answer", data=response_data)
    else:
         logger.info(f"{log_prefix} Agent run completed after resume. Clearing pause details.")
         # Clear pending interaction details as the answer was processed
         final_context.pending_interaction_details = None
         await _update_context_in_db(session_id, user.id, final_context, supabase)

    # --- Format Final Response (Same logic as /interact) --- #
    logger.info(f"{log_prefix} Formatting final API response after resume...")
    if not last_agent_event:
         logger.error(f"{log_prefix} Agent resume run finished, but no final agent event captured.")
         return InteractionResponseData(status='error', message="Internal error: Agent did not produce a final output after resuming.")

    logger.info(f"{log_prefix} Resume run completed normally. Processing final event from agent '{last_agent_event.author}'.")
    processed_payload = await retry_with_exponential_backoff(lambda: _process_adk_event_for_api(last_agent_event, output_logger=logger_dep))

    if isinstance(processed_payload, ErrorResponse):
         # For errors, content_type might not be applicable or fixed to 'error'
         # User model state might be stale, but send the latest fetched one
         return InteractionResponseData(
             content_type='error', # Explicitly set for errors
             data=processed_payload, 
             user_model_state=final_context.user_model_state
         )
    else:
         # Ensure processed_payload has response_type before accessing
         content_type = getattr(processed_payload, 'response_type', 'unknown')
         return InteractionResponseData(
             content_type=content_type, # Get type from the payload
             data=processed_payload,
             user_model_state=final_context.user_model_state # Get state from final context
         )

# --- Helper to process final event --- #
# Renamed logger parameter to output_logger
# --- Retry Helper ---
async def retry_with_exponential_backoff(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0, # Adjusted max delay slightly
    logger_instance: Optional[logging.Logger] = None
) -> T:
    """Retries an async operation with exponential backoff, specifically for Google AI rate limits."""
    delay = initial_delay
    last_exception = None
    log = logger_instance or logging.getLogger(__name__) # Use passed logger or default
    
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        # Catch the specific ResourceExhausted error if possible, otherwise generic ClientError
        except google.api_core.exceptions.ResourceExhausted as e:
            last_exception = e
            log.warning(
                f"Rate limit hit (ResourceExhausted - attempt {attempt + 1}/{max_retries + 1}). "
                f"Error: {e}. Retrying in {delay:.2f} seconds..."
            )
        except google.genai.errors.ClientError as e:
            # Check if it's a rate limit error based on message/details
            # The specific error might be wrapped
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                last_exception = e
                log.warning(
                    f"Rate limit hit (ClientError - attempt {attempt + 1}/{max_retries + 1}). "
                    f"Error: {e}. Retrying in {delay:.2f} seconds..."
                )
            else:
                # Non-rate-limit ClientError, re-raise immediately
                log.error(f"Non-rate-limit ClientError encountered: {e}")
                raise e 
        except Exception as e:
            # Catch other unexpected exceptions
            last_exception = e
            log.error(f"Unexpected exception during operation: {e}")
            raise e # Re-raise unexpected errors immediately
            
        # If we caught a rate limit error, wait and increase delay
        if attempt < max_retries:
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
        else:
            log.error(f"Max retries ({max_retries}) exceeded for rate limit error.")
            break # Exit loop after max retries

    # If loop finished due to retries, raise the last caught exception
    raise last_exception

def _process_adk_event_for_api(
    event: Event,
    output_logger: Optional[TutorOutputLogger] = None # Make logger optional
) -> Union[ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse, Dict[str, Any]]:
    """Processes an ADK event and formats its content into the relevant API response payload model (or an error dict)."""
    log_prefix = f"[_process_adk_event Session: {event.session_id[:8] if event.session_id else 'N/A'}... Event: {event.id[:8] if event.id else 'N/A'}]" # Add safety checks for IDs
    logger_instance = output_logger or logging.getLogger(__name__) # Use passed logger or default
    logger_instance.info(f"{log_prefix} Processing event from author '{event.author}'...")

    # Initialize response variables
    response_payload: Union[ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse, Dict[str, Any]] = {"status": "processing", "message": "No suitable payload generated from event."}
    response_type = "intermediate" # Default

    # Handle error events first
    if event.error_code or event.error_message:
        error_msg = event.error_message or f"Agent error occurred (Code: {event.error_code})"
        logger_instance.error(f"{log_prefix} Agent Error: {error_msg} (Code: {event.error_code})")
        # CORRECT: Create ErrorResponse with response_type
        try:
            response_payload = ErrorResponse(response_type="error", message=error_msg, code=str(event.error_code))
            response_type = "error"
        except Exception as e:
            # Fallback if ErrorResponse creation fails
             logger_instance.error(f"{log_prefix} Failed to create ErrorResponse object: {e}")
             response_payload = {"error": True, "message": error_msg, "code": str(event.error_code)}
             response_type = "error"
        return response_payload # Return immediately on error
        
    # Process event content if available
    if event.content:
        try:
            # Attempt to process content assuming it might be structured JSON
            if event.content.parts:
                # Handle text content
                text_content = event.content.parts[0].text
                if text_content:
                    logger_instance.info(f"{log_prefix} Processing text part: {text_content[:100]}...")
                    # Check if text is JSON
                    try:
                        json_data = json.loads(text_content)
                        logger_instance.info(f"{log_prefix} Parsed text content as JSON.")
                        # TODO: Add logic here to validate/map JSON to known Pydantic models (ExplanationResponse, etc.)
                        # If it matches a known model, assign it to response_payload and set response_type
                        # Example (needs specific model checks):
                        # if "explanation" in json_data:
                        #     response_payload = ExplanationResponse.model_validate(json_data)
                        #     response_type = "explanation"
                        # else: 
                        response_payload = json_data # For now, return the parsed JSON dict
                        response_type = "json_data" # Indicate it's generic JSON for now
                    except json.JSONDecodeError:
                        # Not JSON, treat as plain message
                        response_payload = MessageResponse(response_type="message", text=text_content)
                        response_type = "message"
                        logger_instance.info(f"{log_prefix} Treated as plain text message.")
                # Handle function calls (often intermediate, maybe not final payload)
                elif event.content.parts[0].function_call:
                    fc = event.content.parts[0].function_call
                    msg = f"{log_prefix} Event ended with tool call: {fc.name}"
                    logger_instance.warning(f"{log_prefix} Event ended with tool call: {fc.name}")
                    # Usually don't return a tool call as the final API response
                    # Keep response_payload as the default processing message
                    response_payload = MessageResponse(response_type="message", text=msg) # Or keep default processing message
                    response_type = "tool_call" # Indicate intermediate state
                    if output_logger: output_logger.log_tool_call(event.author, fc.name, fc.args)
                # Handle function responses (also often intermediate)
                elif event.content.parts[0].function_response:
                    fr = event.content.parts[0].function_response
                    msg = f"{log_prefix} Event ended after receiving tool response for: {fr.name}"
                    logger_instance.info(f"{log_prefix} Event ended after receiving tool response for {fr.name}")
                    # Keep response_payload as the default processing message
                    response_payload = MessageResponse(response_type="message", text=msg) # Or keep default processing message
                    response_type = "tool_response" # Indicate intermediate state
                    if output_logger: output_logger.log_tool_response(event.author, fr.name, fr.response)
            else:
                logger_instance.warning(f"{log_prefix} Event content has no parts.")
        except Exception as e:
             logger_instance.exception(f"{log_prefix} Error processing event content: {e}")
             response_payload = ErrorResponse(response_type="error", message=f"Internal error processing event content: {e}", code="CONTENT_PROCESSING_ERROR")
             response_type = "error"
    else:
        # No error, no content? Empty response?
        logger_instance.warning(f"{log_prefix} Event finished with no error or content.")
        response_payload = MessageResponse(response_type="message", text="Agent finished processing.")
        response_type = "message"

    logger_instance.info(f"{log_prefix} Final processed payload type: {response_type}")
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

# TODO: Implement endpoint for session analysis if needed:
# POST /sessions/{session_id}/analyze-session (Full Session Analysis) 