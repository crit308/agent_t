from __future__ import annotations
import logging # Add standard logging import
import os
import json
import traceback  # Add traceback import
import shutil # Make sure shutil is imported
import time
from typing import Optional, List, Dict, Any, Literal
from uuid import UUID

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
from ai_tutor.output_logger import get_logger as get_session_logger_instance, TutorOutputLogger # Rename import
from ai_tutor.dependencies import get_supabase_client, get_session_service
from ai_tutor.auth import verify_token
from google.api_core import exceptions as google_exceptions
from google.adk.tools import FunctionTool, ToolContext
import google.genai.errors # Add this import

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

router = APIRouter()
# session_manager = SessionManager() # Keep for create_session? Or fully replace with service? Consider removing.

# Setup standard Python logger for this module
logger = logging.getLogger(__name__)

# Directory for temporary file uploads
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

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
    return get_session_logger_instance(output_file=log_file)

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
async def upload_documents( # Function name was already correct
    session_id: UUID,                             # Path Param
    request: Request,                             # Raw Request Dependency
    files: List[UploadFile] = File(...),          # Body (Form Data) Dependency
    supabase: Client = Depends(get_supabase_client), # Dependency 1
    # tutor_context: TutorContext = Depends(get_tutor_context) # Temporarily remove dependency
):
    """ # noqa: D415
    Uploads one or more documents to the specified session.
    Stores files temporarily, uploads them to Supabase Storage,
    and synchronously triggers document analysis.
    """
    user: User = request.state.user # Get user from request state populated by verify_token dependency
    # Use the standard logger for general messages
    # session_logger = get_session_logger(session_id) # Keep if needed for specific TutorOutputLogger methods

    # --- Manually fetch context since dependency is removed ---
    try:
        session_service: SupabaseSessionService = await get_session_service(supabase) # Get service instance
        tutor_context: TutorContext = await get_tutor_context(session_id, request, session_service) # Call dependency logic manually
    except HTTPException as he:
        raise he # Re-raise validation/auth errors
    except Exception as e:
        logger.exception(f"Error manually fetching context for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error fetching session context.")
    # --- End Manual Fetch ---

    folder_id = tutor_context.folder_id
    if not folder_id:
        # Use standard logger here
        logger.error(f"Session context is missing folder_id for session {session_id}.")
        raise HTTPException(status_code=400, detail="Session context is invalid (missing folder ID).")
    
    logger.info(f"Starting document upload for session {session_id}, folder {folder_id}") # Use standard logger
    temp_file_paths = []
    uploaded_filenames = []
    
    # Save uploaded files temporarily
    try:
        for file in files:
            # Sanitize filename to prevent directory traversal or invalid chars
            filename = os.path.basename(file.filename or f"upload_{uuid.uuid4()}") 
            temp_path = os.path.join(TEMP_UPLOAD_DIR, filename)
            # Read file content into memory, then write to temp file
            # Consider chunking for very large files to avoid high memory usage
            content = await file.read()
            with open(temp_path, "wb") as temp_file:
                temp_file.write(content)
            temp_file_paths.append(temp_path)
            uploaded_filenames.append(filename)
            # No need to close file.file when using await file.read() context
    except Exception as e:
        logger.exception(f"Error saving files locally for session {session_id}: {e}")
        # Clean up any partially saved files
        for path in temp_file_paths:
             if os.path.exists(path): os.remove(path)
        raise HTTPException(status_code=500, detail="Failed to process uploaded files")

    if not temp_file_paths:
        raise HTTPException(status_code=400, detail="No files were successfully processed locally.")

    # Upload to Supabase Storage
    message = ""
    try:
        file_upload_manager = FileUploadManager(supabase=supabase)
        
        for i, temp_path in enumerate(temp_file_paths):
            uploaded_file = await file_upload_manager.upload_and_process_file(
                file_path=temp_path,
                user_id=user.id,
                folder_id=folder_id
            )
            message += f"Uploaded {uploaded_filenames[i]} to Supabase at {uploaded_file.supabase_path}. "
        logger.info(f"Supabase upload complete for session {session_id}") # Use standard logger

        # Update context with file paths and save
        if uploaded_filenames:
            tutor_context.uploaded_file_paths.extend(uploaded_filenames) # Keep track of names
            logger.info("Explicitly saving context with updated file paths BEFORE analysis.") # Standard logger
            # Use the helper, ensuring it uses standard logging internally or returns success/failure
            success = await _update_context_in_db(session_id, user.id, tutor_context, supabase)
            if not success:
                # Handle context update failure if critical
                logger.error(f"Failed to persist context update for session {session_id} before analysis.")
                # Decide if this is a fatal error for the upload process

    except Exception as e:
        logger.exception(f"Error during file upload processing for session {session_id}: {e}")
        # Clean up temp files on failure
        for temp_path in temp_file_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")

    # Clean up temp files after successful upload
    for temp_path in temp_file_paths:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Trigger analysis synchronously
    analysis_status = "failed"
    try:
        logger.info(f"Starting synchronous analysis for session {session_id}...") # Standard logger

        analysis_result: Optional[AnalysisResult] = await analyze_documents(
            context=tutor_context,
            supabase=supabase
        )

        if analysis_result:
            analysis_status = "completed"
            tutor_context.analysis_result = analysis_result
            logger.info(f"Analysis completed for session {session_id}. Persisting context with analysis result.") # Standard logger
            success = await _update_context_in_db(session_id, user.id, tutor_context, supabase)
            if not success:
                logger.error(f"Failed to persist context update for session {session_id} after analysis.")
                # Log error but maybe continue to return response to user?
            message += "Analysis completed."
        else:
            message += "Analysis completed but no results were generated."
            analysis_status = "completed_empty"

    except Exception as e:
        logger.exception(f"Error during document analysis execution for session {session_id}: {e}")
        message += f"Analysis trigger failed: {str(e)}"
        analysis_status = "failed"

    # Construct and return the final response
    try:
        response_payload = DocumentUploadResponse(
            files_received=uploaded_filenames,
            analysis_status=analysis_status,
            message=message
        )
        logger.info(f"Successfully constructed DocumentUploadResponse for session {session_id}") # Standard logger
        return response_payload
    except Exception as validation_error:
        logger.exception(f"Pydantic validation failed for DocumentUploadResponse on session {session_id}: {validation_error}")
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
    request: Request,
    interaction_input: InteractionRequestData = Body(...),
    supabase: Client = Depends(get_supabase_client),
    session_service: SupabaseSessionService = Depends(get_session_service),
    tutor_context: TutorContext = Depends(get_tutor_context)
):
    # Use the standard logger for endpoint-level info
    logger.info(f"\n=== Received /interact for session {session_id} ===")
    logger.info(f"Input Type: {interaction_input.type}, Data: {interaction_input.data}")

    # --- ADK Runner Setup ---
    orchestrator_agent = create_orchestrator_agent() # Create the main orchestrator

    logger.info(f"Creating/Using ADK Runner instance for session {session_id}")
    # Note: Assuming runner caching is handled elsewhere or we create a new one per request
    adk_runner = Runner(
        app_name="ai_tutor",
        agent=orchestrator_agent,
        session_service=session_service # Use the injected service
        # Add artifact_service and memory_service if needed by agents
    )

    # Prepare the input message for the ADK Runner
    adk_input_content = None
    if interaction_input.type == "user_message" and interaction_input.data:
        # Convert the input data (string) into ADK Content/Part format
        # Ensure data is treated as a simple string for the text part
        user_text = str(interaction_input.data) 
        adk_input_content = Content(role="user", parts=[Part(text=user_text)])
    elif interaction_input.type == "start":
        logger.info(f"Handling 'start' interaction type for session {session_id}. No specific user message sent to agent.")
        # For 'start', send no message. The orchestrator agent's logic should handle the initial state.
        adk_input_content = None
    elif interaction_input.type == "tool_response":
        # Handle tool responses needed by the ADK flow (e.g., from long-running tools)
        logger.info(f"Handling 'tool_response' interaction type for session {session_id}.")
        try:
            if not isinstance(interaction_input.data, dict) or 'name' not in interaction_input.data or 'response' not in interaction_input.data:
                 raise ValueError("Invalid format for tool_response data.")
            # Construct the ADK FunctionResponse object
            func_response = FunctionResponse(
                    name=interaction_input.data['name'],
                response=interaction_input.data['response']
            )
            adk_input_content = Content(role="function", parts=[Part(function_response=func_response)])
        except Exception as e:
            logger.error(f"Error processing tool_response data for session {session_id}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid tool_response data: {e}")
    else:
        # Log and raise error for truly unhandled types
        logger.warning(f"Unsupported interaction type received: {interaction_input.type}. Data: {interaction_input.data}")
        raise HTTPException(status_code=400, detail=f"Unsupported interaction type: {interaction_input.type}")

    # --- Setup RunConfig ---
    run_config = RunConfig(
        # workflow_name="AI Tutor Interaction", # Optional: for tracing/logging
        # group_id=str(session_id) # Optional: group related runs
    )

    # Stream events back to the client (or collect and return)
    try:
        user_id = str(request.state.user.id) # Get user ID from request state

        # --- Run the ADK agent ---
        logger.debug(f"Calling adk_runner.run_async for session {session_id}, user {user_id}")
        event_generator = adk_runner.run_async(
            user_id=user_id,
            session_id=str(session_id),
            new_message=adk_input_content, # Pass the prepared Content object (or None)
            run_config=run_config,
        )

        # Process the stream of events
        last_processed_event_data = None
        async for event in event_generator:
            logger.debug(f"[Interact] Raw ADK Event: ID={event.id}, Author={event.author}, Type={type(event.content)}, Actions={event.actions}")
            processed_event_data = _process_adk_event_for_api(event, logger) # Use standard logger

            # Check content_type instead of response_type
            if processed_event_data.content_type == "error":
                 # Access message via .data.message if data is ErrorResponse
                 error_message = "Unknown error" # Default message
                 if isinstance(processed_event_data.data, ErrorResponse):
                    error_message = processed_event_data.data.message
                 logger.error(f"ADK Runner returned an error event: {error_message}")

            # Store the last successfully processed event data suitable for the API response
            # Check content_type instead of response_type
            if processed_event_data.content_type != 'intermediate': # Filter out internal/log events
                last_processed_event_data = processed_event_data

        # After the loop, return the data from the *last* relevant API event
        if last_processed_event_data:
            # Log content_type instead of response_type
            logger.info(f"ADK Runner finished. Returning final API event data for session {session_id}: {last_processed_event_data.content_type}")
            return last_processed_event_data
        else:
            logger.warning(f"ADK Runner finished but produced no events suitable for API response for session {session_id}.")
            # Construct a proper InteractionResponseData with ErrorResponse payload
            default_error_payload = ErrorResponse(
                response_type="error",
                error_type="processing_error",
                message="Interaction complete, but no specific output generated."
            )
            return InteractionResponseData(
                content_type="error",
                data=default_error_payload,
                user_model_state=UserModelState() # Use default state here too
            )

    except google_exceptions.ResourceExhausted as e:
        # Existing handling for google.api_core exceptions
        error_message = f"API Rate Limit Exceeded for session {session_id}: {e}"
        logger.error(error_message)
        retry_after = "unknown"
        try:
            if hasattr(e, 'details'):
                for detail in e.details:
                    if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                        retry_after = detail.get('retryDelay', 'unknown')
                        break
        except Exception:
            pass
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Please try again after {retry_after}.",
            headers={"Retry-After": retry_after.replace('s', '') if retry_after != 'unknown' else "60"}
        )
    except google.genai.errors.ClientError as e:
        # Add specific handling for google.genai ClientError (which includes 429)
        if e.status_code == 429:
            error_message = f"API Rate Limit Exceeded (ClientError) for session {session_id}: {e.message}"
            logger.error(error_message)
            retry_after = "unknown"
            try:
                # Attempt to parse retryDelay from the details
                if isinstance(e.details, list):
                    for detail in e.details:
                        if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                            retry_after = detail.get('retryDelay', 'unknown')
                            break
            except Exception:
                pass # Ignore errors parsing details
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please try again after {retry_after}.",
                headers={"Retry-After": retry_after.replace('s', '') if retry_after != 'unknown' else "60"}
            )
        else:
            # Handle other ClientErrors as 500
            tb_str = traceback.format_exc()
            logger.error(f"ClientError during interaction with ADK runner for session {session_id}: {e}\nTraceback:\n{tb_str}")
            raise HTTPException(status_code=500, detail=f"An API client error occurred: {e.message}")

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Error during interaction with ADK runner for session {session_id}: {e}\nTraceback:\n{tb_str}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@router.post(
    "/sessions/{session_id}/answer",
    response_model=InteractionResponseData,
    dependencies=[Depends(verify_token)],
    summary="Submit Answer to Paused Question",
    tags=["Tutoring Workflow"]
)
async def submit_answer_to_tutor(
    session_id: UUID,
    request: Request,
    interaction_input: InteractionRequestData = Body(...),
    supabase: Client = Depends(get_supabase_client),
    session_service: SupabaseSessionService = Depends(get_session_service),
    tutor_context: TutorContext = Depends(get_tutor_context)
):
    # logger = get_session_logger(session_id) # REMOVE THIS LINE
    session_logger = get_session_logger(session_id) # Use session logger for agent-related logging
    user: User = request.state.user
    print(f"\n=== Received /answer for session {session_id} ===")
    # Validate interaction type and data
    if interaction_input.type != "tool_response" or not isinstance(interaction_input.data, dict):
        logger.warning(f"Invalid interaction type/data for answer submission: {interaction_input.type}")
        raise HTTPException(status_code=400, detail="Answer submission must be of type 'tool_response' with data.")

    # Extract tool call ID and response data from the incoming request
    tool_call_id = interaction_input.data.get("tool_call_id") # Assuming frontend sends this
    tool_response_data = interaction_input.data.get("response") # Assuming frontend sends the response part

    if not tool_call_id or tool_response_data is None:
        logger.warning(f"Missing tool_call_id or response in answer submission for session {session_id}")
        raise HTTPException(status_code=400, detail="'tool_call_id' and 'response' are required in data for answer submission.")

    # --- Create the FunctionResponse Event ---
    answer_event = Event(
        author="user", # Or system? Clarify ADK best practice - 'user' role with tool response is common
        content=Content( # Use imported Content
            role="tool", # Response is for a tool
            parts=[
                Part(function_response=FunctionResponse( # Correctly use FunctionResponse
                    name=interaction_input.data['name'],
                    response=tool_response_data
                ))
            ]
        ),
        invocation_id=tutor_context.last_interaction_summary or f"resume_{session_id}",
    )

    # --- Resume the ADK Runner ---
    # Re-initialize runner and run_async, providing the answer event.
    orchestrator_agent = create_orchestrator_agent() # Recreate agent instance
    adk_runner = Runner("ai_tutor", orchestrator_agent, session_service)
    run_config = RunConfig(workflow_name="Tutor_Interaction_Resume", group_id=str(session_id))

    # Process the stream *after* resuming (need to define last_event... here)
    try:
        last_event: Optional[Event] = None # Ensure last_event is initialized
        async for event in adk_runner.run_async(
            user_id=str(user.id),
            session_id=str(session_id),
            new_message=answer_event.content, # Pass the Content object from the event
            run_config=run_config
        ):
            print(f"[Answer] Received Event after resume: ID={event.id}, Author={event.author}")
            session_logger.log_orchestrator_output(event.content) # Use session logger
            last_event = event
            logger.info(f"[Answer] Received event after resume: {event.type} - {event.action}")

        if not last_event:
            logger.error(f"ADK Runner finished without producing events after resuming for session {session_id}.")
            raise HTTPException(status_code=500, detail="Tutor did not respond after answer submission.")

        logger.info(f"ADK Runner finished after answer. Last event action: {last_event.action}")

        # Process the final event for the API response
        api_response_data = _process_adk_event_for_api(last_event, session_logger)
        return api_response_data

    except google_exceptions.GoogleAPIError as e:
        logger.exception(f"Google API Error during answer processing for session {session_id}: {e}")
        error_detail = f"Google API Error: {e.message}" if hasattr(e, 'message') else str(e)
        raise HTTPException(status_code=503, detail=f"AI service unavailable. {error_detail}")
    except Exception as e:
        logger.exception(f"Error during answer submission processing for session {session_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal tutor error: {str(e)}")

# Helper function to process ADK events into API response data
def _process_adk_event_for_api(event: Event, logger: TutorOutputLogger) -> InteractionResponseData:
    """Processes an ADK event and formats it for the API response."""
    # Initialize response variables
    response_type: Literal["explanation", "question", "feedback", "message", "error", "intermediate"] = "intermediate"
    response_data = None

    try:
        # Handle error events first
        if event.error_code or event.error_message:
            error_text = event.error_message or f"An error occurred (Code: {event.error_code})"
            logger.error(f"ADK Event error: {error_text}")
            # Return the error response directly. Need to wrap in InteractionResponseData
            # with appropriate state. Since we don't have context here, use default state.
            error_payload = ErrorResponse(
                    response_type="error",
                    error_type="tutor_error",
                    message=error_text
            )
            return InteractionResponseData(
                content_type="error",
                data=error_payload,
                user_model_state=UserModelState() # Default state for error
            )

        # Process event content if available
        if event.content and event.content.parts:
            for part in event.content.parts:
                # Handle text content
                if part.text:
                    text = part.text.strip()
                    if text:
                        logger.info(f"Processing text content: {text[:100]}...")
                        response_type = "message"
                        # Add missing response_type here
                        response_data = MessageResponse(response_type="message", text=text)
                        break  # Text content takes precedence

                # Handle function calls
                elif part.function_call:
                    tool_name = part.function_call.name
                    tool_args = part.function_call.args or {}
                    logger.info(f"Processing function call: {tool_name}")

                    if tool_name == "present_explanation":
                        try:
                            valid_args = {k: v for k, v in tool_args.items() if v is not None}
                            response_data = ExplanationResponse.model_validate(valid_args)
                            response_type = "explanation"
                        except ValidationError as e:
                            logger.error(f"Validation error in present_explanation: {e}")
                            return InteractionResponseData(
                                type="error",
                                data=ErrorResponse(
                                    response_type="error",
                                    error_type="validation_error",
                                    message=f"Invalid explanation data: {str(e)}"
                                )
                            )

                    elif tool_name == "ask_checking_question":
                        try:
                            if not isinstance(tool_args.get('question'), dict):
                                raise ValueError("Question data must be a dictionary")
                            
                            question = QuizQuestion.model_validate(tool_args['question'])
                            response_data = QuestionResponse(
                                question=question,
                                topic=tool_args.get('topic', 'General')
                            )
                            response_type = "question"
                        except (ValidationError, ValueError) as e:
                            logger.error(f"Validation error in ask_checking_question: {e}")
                            return InteractionResponseData(
                                type="error",
                                data=ErrorResponse(
                                    response_type="error",
                                    error_type="validation_error",
                                    message=f"Invalid question data: {str(e)}"
                                )
                            )

                    else:
                        # Handle other tool calls as intermediate steps
                        response_type = "intermediate"
                        response_data = {"tool_name": tool_name, "args": tool_args}
                    
                    break  # Process one function call at a time

                # Handle function responses
                elif part.function_response:
                    response_type = "intermediate"
                    response_data = {
                        "tool_name": part.function_response.name,
                        "response": part.function_response.response
                    }

        # Handle case where no meaningful response was generated
        if response_type == "intermediate" and not response_data:
            response_data = {"status": "processing"}
        elif not response_data:
            logger.warning("No response data generated from event")
            response_type = "message"
            response_data = MessageResponse(response_type="message", text="Processing your request...")

        # This return was incorrect - It should return InteractionResponseData
        # Find the correct UserModelState to return - this function doesn't have it!
        # For now, return intermediate if not a final type, requires refactor
        if response_type in ["explanation", "question", "feedback", "message", "error"]:
             # Need to get the current UserModelState somehow to return here.
             # This function signature does not have access to the full TutorContext.
             # Placeholder - returning default. THIS NEEDS REFACTORING.
             logger.warning("_process_adk_event_for_api returning default UserModelState - requires refactor!")
             return InteractionResponseData(
                 content_type=response_type,
                 data=response_data, # Already a validated Pydantic model or dict
                 user_model_state=UserModelState()
             )
        else:
             # Return intermediate status without user_model_state? Or default?
             # Let's return the intermediate dict for now, main loop handles state.
             # This also needs clarification on intended flow.
              logger.debug(f"Returning intermediate data: {response_data}")
              # Cannot return a dict, must be InteractionResponseData.
              # Return default state for intermediate for now.
              return InteractionResponseData(
                 content_type="intermediate",
                 data=response_data or {"status": "processing"},
                 user_model_state=UserModelState()
             )

    except Exception as e:
        logger.error(f"Unexpected error processing ADK event: {str(e)}")
        # Construct ErrorResponse first
        error_payload = ErrorResponse(
                response_type="error",
                error_type="processing_error",
                message=f"Internal error processing tutor response: {str(e)}"
            )
        # Now construct the InteractionResponseData wrapper
        return InteractionResponseData(
            content_type="error", # Use 'error' as the content type
            data=error_payload, # Embed the ErrorResponse
            user_model_state=UserModelState() # Provide a default UserModelState
        )

# --- Helper to update context in DB (replace direct calls) ---
async def _update_context_in_db(session_id: UUID, user_id: UUID, context: TutorContext, supabase: Client):
    """Helper to persist context via SupabaseSessionService interface."""
    # This mimics how the SessionService's append_event would work
    try:
        context_dict = context.model_dump(mode='json')
        response: PostgrestAPIResponse = supabase.table("sessions").update(
            {"context_data": context_dict}
        ).eq("id", str(session_id)).eq("user_id", str(user_id)).execute()
        # Check for data presence instead of response.error for success
        if not response.data:
            print(f"Error updating context in DB for session {session_id}: {response.error}")
            return False
        return True
    except Exception as e:
        print(f"Exception updating context in DB for session {session_id}: {e}")
        return False

# TODO: Implement endpoint for session analysis if needed:
# POST /sessions/{session_id}/analyze-session (Full Session Analysis) 