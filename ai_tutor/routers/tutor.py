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
from google.adk.runners import Runner, RunConfig # Remove RunnerError import
from google.adk.agents import LlmAgent, BaseAgent # Correct LlmAgent casing
# Import Content/Part directly from google.ai.generativelanguage
# from google.ai.generativelanguage import Content, Part
# Import Content/Part from google.adk.events
# from google.adk.events import Event, Content, Part # Import Content/Part here
# Import types from google.adk.agents
# Import Content/Part directly from google.generativeai
# from google.ai.generativelanguage import Content, Part # Keep this line (or ensure it's there)
# Import Content/Part directly from google.ai.generativelanguage
# from google.ai.generativelanguage import Content, Part # Ensure this import exists
from google.adk.events import Event # Import Event
from google.generativeai.types import Content, Part # Correct import path

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
from ai_tutor.context import TutorContext
from ai_tutor.output_logger import get_logger as get_session_logger_instance, TutorOutputLogger # Rename import
from ai_tutor.manager import AITutorManager
from ai_tutor.dependencies import get_supabase_client, get_session_service
from ai_tutor.auth import verify_token
from google.api_core import exceptions as google_exceptions


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
    logger.info(f"Received interaction request for session {session_id}: {interaction_input.type}")
    # Keep the custom session logger if needed for specific agent output logging later
    session_logger = get_session_logger(session_id)
    print(f"\n=== Received /interact for session {session_id} ===")
    # Log safely in case data is complex or None
    try:
        data_log = json.dumps(interaction_input.data)
    except TypeError:
        data_log = str(interaction_input.data)
    print(f"Input Type: {interaction_input.type}, Data: {data_log}")

    # --- Get or Create ADK Runner instance for this session ---
    orchestrator_agent = create_orchestrator_agent()
    runner_key = str(session_id) # Use string key for dict

    # Access the runner cache from app state
    if not hasattr(request.app.state, "adk_runners"):
         # This should not happen if lifespan is correctly configured
         logger.error("ADK Runner cache (app.state.adk_runners) not initialized!")
         raise HTTPException(status_code=500, detail="Internal server error: Runner cache missing.")

    # Get existing runner or create a new one for this session
    if runner_key not in request.app.state.adk_runners:
        print(f"Creating new ADK Runner instance for session {session_id}")
        request.app.state.adk_runners[runner_key] = Runner(
            app_name="ai_tutor", # Use keyword arg
            agent=orchestrator_agent,        # Use keyword arg
            session_service=session_service # Use keyword arg
        )
    adk_runner: Runner = request.app.state.adk_runners[runner_key]
    # Use standard logger for endpoint info
    logger.info(f"Using ADK Runner for session {session_id}")

    # Use a central variable for the message to send to the ADK runner
    message_to_runner = None

    # Handle interaction input type
    if interaction_input.type == "user_message":
        user_input_text = interaction_input.message
        # Ensure data is None or a valid JSON string representation
        user_data_str = json.dumps(interaction_input.data) if interaction_input.data is not None else 'None'
        user_input_text = f"User Action Type: {interaction_input.type}. Data: {user_data_str}"
        # Create message using Content/Part objects
        user_event_content = Content(
            role="user",
            parts=[Part(text=user_input_text)]
        )
        session_logger.log_user_input(user_input_text) # Log user input
        print(f"[Interact] Running Agent: {orchestrator_agent.name}")

    elif interaction_input.type == "tool_response":
        # Validate tool_response data structure
        if not isinstance(interaction_input.data, dict) or 'name' not in interaction_input.data or 'response' not in interaction_input.data:
            logger.error(f"Invalid tool_response data format for session {session_id}: {interaction_input.data}")
            raise HTTPException(status_code=400, detail="Invalid format for tool_response data.")

        try:
            # Ensure response data is serializable if needed by FunctionResponse
            response_data = interaction_input.data.get('response', {})
            # Construct function response using Content/Part objects
            message_to_runner = Content(
                role="function",
                parts=[Part.from_function_response(
                    name=interaction_input.data['name'],
                    response=response_data
                )]
            )
            # Log the structured tool response if needed
            session_logger.log_tool_response(interaction_input.data['name'], response_data) 

        except Exception as e:
            logger.exception(f"Error constructing FunctionResponse for tool_response in session {session_id}: {e}")
            raise HTTPException(status_code=500, detail="Error processing tool response.")

    elif interaction_input.type == "system_message": # Example if needed
        message_to_runner = Content(
            role="system",
            parts=[Part(text=interaction_input.message)]
        )
        session_logger.log_system_message(interaction_input.message) # Log system message

    else:
        # Default or error handling for unknown types
        error_text = f"Unknown interaction type received: {interaction_input.type}. Data: {interaction_input.data}"
        logger.error(error_text)
        message_to_runner = Content(
            role="user",
            parts=[Part(text=error_text)]
        )

    if not message_to_runner:
        # This case should ideally not be reached if all interaction_input.type are handled
        logger.error(f"Interaction resulted in no message to send to runner for session {session_id}. Type: {interaction_input.type}")
        raise HTTPException(status_code=500, detail="Internal error processing interaction.")

    # Run the agent with the constructed message
    try:
        run_config = RunConfig() # Remove context parameter
        user: User = request.state.user # Get user from request state

        last_event = None
        async for event in adk_runner.run_async(
            user_id=str(user.id),
            session_id=str(session_id),
            new_message=message_to_runner, # Pass the Content object
            run_config=run_config
        ):
            print(f"[Interact] Received Event: ID={event.id}, Author={event.author}, Actions={event.actions}")
            session_logger.log_orchestrator_output(event.content) # Use the session logger instance
            last_event = event
            # Process events if needed during the run (e.g., log intermediate steps)
            logger.info(f"[Interact] Received event: {event.type} - {event.action}")

        if not last_event:
            logger.error(f"ADK Runner finished without producing any events for session {session_id}.")
            raise HTTPException(status_code=500, detail="Tutor did not produce a response.")

        logger.info(f"ADK Runner finished. Last event action: {last_event.action}")

        # --- Process the final event to create the API response ---
        api_response_data = _process_adk_event_for_api(last_event, session_logger)
        return api_response_data

    except google_exceptions.GoogleAPIError as e:
        logger.exception(f"Google API Error during interaction for session {session_id}: {e}")
        # Extract more specific details if possible
        error_detail = f"Google API Error: {e.message}" if hasattr(e, 'message') else str(e)
        raise HTTPException(status_code=503, detail=f"AI service unavailable. {error_detail}")
    except Exception as e:
        logger.exception(f"Error during interaction with ADK runner for session {session_id}: {e}")
        traceback.print_exc() # Print full traceback to console/logs
        raise HTTPException(status_code=500, detail=f"Internal tutor error: {str(e)}")

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
        author="user", # Or system? Clarify ADK best practice
        content=Content( # Use import from events
            role="tool", # Response is for a tool
            parts=[
                Part.from_function_response( # Use import from events
                    name="ask_user_question_and_get_answer", # Tool name
                    id=tool_call_id, # CRUCIAL: Match the original tool call ID
                    response={"answer_index": tool_response_data} # The data the tool's caller expects
                )
            ] # Pass the response data as a list
        ),
        invocation_id=tutor_context.last_interaction_summary or f"resume_{session_id}"
    )

    # ADK's Runner should handle routing this event correctly based on the function_call_id.
    # Note: We pass the *content* of the event, not the full event object to new_message
    run_config = RunConfig() # Use empty config

    # Process the stream *after* resuming (need to define last_event... here)
    try:
        last_event = None
        async for event in adk_runner.run_async(
            user_id=str(user.id),
            session_id=str(session_id),
            new_message=answer_event, # Pass the dictionary content
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
# (Keep this logic or adapt as needed)
def _process_adk_event_for_api(event: Event, logger: TutorOutputLogger) -> InteractionResponseData:
    """Processes the final ADK event and formats it for the API response."""
    response_data = None
    response_type: Literal["explanation", "question", "feedback", "message", "error"] = "message" # Default

    if event.action == EventActions.ERROR or not event.content or not event.content.parts:
        error_text = "An error occurred in the tutor." # Default error
        if event.content and event.content.parts:
            # Try to get a more specific error from the event content
            error_text = event.content.parts[0].text if event.content.parts[0].text else error_text
        elif event.data and isinstance(event.data, dict) and 'error' in event.data:
             error_text = str(event.data['error'])

        logger.error(f"ADK Event indicates error: {error_text}")
        # Use ErrorResponse model structure within InteractionResponseData
        response_data = ErrorResponse(error_type="tutor_error", message=error_text)
        response_type = "error"
        # Consider raising HTTPException here if it's always fatal
        # raise HTTPException(status_code=500, detail=error_text)
        # Or return the structured error response:
        return InteractionResponseData(type="error", data=response_data)

    # --- Process Successful Event Content ---
    # Extract content (handle potential missing parts)
    part = event.content.parts[0] if event.content.parts else None
    text_content = part.text if part and part.text else None
    function_call = part.function_call if part and part.function_call else None
    function_response = part.function_response if part and part.function_response else None

    # Log raw event data for debugging complex cases
    logger.debug(f"Processing event data: {event.data}")

    # Determine response type based on event action and data
    if event.action == EventActions.CALL_FUNCTION and function_call:
        tool_name = function_call.name
        tool_args = function_call.args
        logger.log_tool_call(tool_name, tool_args)

        # Map tool calls to API response types
        if tool_name == "present_explanation":
            response_type = "explanation"
            response_data = ExplanationResponse.model_validate(tool_args)
        elif tool_name == "ask_checking_question":
            response_type = "question"
            # Ensure args match QuestionResponse fields (QuizQuestion is nested)
            # We might need to construct QuestionResponse from args
            try:
                 # Assuming args directly contain the structure needed by QuestionResponse
                 # This might need adjustment based on actual tool_args content
                 if 'question' in tool_args and isinstance(tool_args['question'], dict):
                      validated_question = QuizQuestion.model_validate(tool_args['question'])
                      response_data = QuestionResponse(question=validated_question, topic=tool_args.get('topic', 'Unknown Topic'))
                 else:
                     # Handle case where args don't match expected structure
                     logger.error(f"Tool args for {tool_name} missing 'question' dict: {tool_args}")
                     response_data = MessageResponse(text=f"Tutor wants to ask a question, but data is missing.")
                     response_type = "message"

            except ValidationError as e:
                logger.error(f"Validation error processing {tool_name} args: {e}. Args: {tool_args}")
                response_data = MessageResponse(text=f"Tutor wants to ask a question, but data format is invalid.")
                response_type = "message"

        # Add other tool mappings (e.g., present_feedback -> feedback)
        # elif tool_name == "present_feedback":
        #     response_type = "feedback"
        #     response_data = FeedbackResponse.model_validate(tool_args)

        else:
            # Default for unmapped tool calls (maybe log as message?)
            logger.warning(f"Unhandled tool call in API response mapping: {tool_name}")
            response_type = "message"
            response_data = MessageResponse(text=f"Tutor performed action: {tool_name}")

    elif event.action == EventActions.PAUSE and event.data:
         # Handle PAUSE specifically, e.g., for the long-running question tool
         logger.info(f"ADK Flow Paused. Data: {event.data}")
         # Expecting question data in event.data for AskUserQuestionTool
         if isinstance(event.data, dict) and 'question' in event.data:
             try:
                 validated_question = QuizQuestion.model_validate(event.data['question'])
                 response_type = "question" # API response indicates a question is being asked
                 response_data = QuestionResponse(
                     question=validated_question,
                     topic=event.data.get('topic', 'Unknown Topic'),
                     # Include tool_call_id if necessary for frontend to resume
                     context={"tool_call_id": event.tool_call_id}
                 )
             except ValidationError as e:
                 logger.error(f"Validation error processing PAUSE event data: {e}. Data: {event.data}")
                 response_data = MessageResponse(text=f"Tutor paused to ask a question, but data format is invalid.")
                 response_type = "message"
         else:
             logger.warning(f"PAUSE event missing expected question data: {event.data}")
             response_data = MessageResponse(text=f"Tutor paused.")
             response_type = "message"

    elif text_content:
        # If it's just text content, treat it as a message
        logger.log_llm_output(text_content)
        response_type = "message"
        response_data = MessageResponse(text=text_content)

    else:
        # Fallback if event content is not text or a mapped tool call
        logger.warning(f"Unhandled event content/action for API response. Action: {event.action}, Content: {event.content}")
        response_type = "message"
        response_data = MessageResponse(text="Tutor provided an update.")

    # Ensure response_data is not None before returning
    if response_data is None:
        logger.error("Failed to determine response data from ADK event.")
        # Return a generic error response instead of None
        response_data = ErrorResponse(error_type="processing_error", message="Could not process tutor's response.")
        response_type = "error"

    return InteractionResponseData(type=response_type, data=response_data)

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