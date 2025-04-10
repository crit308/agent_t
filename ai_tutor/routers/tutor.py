from __future__ import annotations
import logging # Add logging import
from typing import Optional, List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form, Body, Request
import os
import shutil
import time
import json
import traceback
from supabase import Client
from gotrue.types import User # To type hint the user object
from uuid import UUID

from ai_tutor.session_manager import SessionManager, SupabaseSessionService # Import ADK Session Service
from ai_tutor.tools.file_upload import FileUploadManager
from ai_tutor.agents.analyzer_agent import analyze_documents
from ai_tutor.agents.session_analyzer_agent import analyze_teaching_session
from ai_tutor.agents.orchestrator_agent import create_orchestrator_agent
from ai_tutor.agents.analyzer_agent import AnalysisResult
from ai_tutor.agents.models import (
    FocusObjective,
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis, QuizQuestion, QuizFeedbackItem
)
from ai_tutor.api_models import ( # Keep API models
    DocumentUploadResponse, AnalysisResponse, TutorInteractionResponse,
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse,
    InteractionRequestData, InteractionResponseData  # Add InteractionResponseData
)
from ai_tutor.context import TutorContext
from ai_tutor.output_logger import get_logger, TutorOutputLogger
# Use ADK Runner and related components
from google.adk.runners import Runner, RunConfig # Remove RunnerError import
# Import LLMAgent and Event from their specific modules
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events import Event, EventActions
from google.generativeai import types as adk_types # Import core types directly
from ai_tutor.manager import AITutorManager
from pydantic import BaseModel
from ai_tutor.dependencies import get_supabase_client, get_session_service # Get ADK service dependency
from ai_tutor.auth import verify_token # Get auth dependency

router = APIRouter()
session_manager = SessionManager()

# Directory for temporary file uploads
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Setup module logger
logger = logging.getLogger(__name__)

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
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or not authorized for user.")
    # Deserialize TutorContext from session state
    return TutorContext.model_validate(session.state)

# --- Helper to get logger ---
def get_session_logger(session_id: UUID) -> TutorOutputLogger:
    # Customize logger per session if needed, e.g., different file path
    log_file = os.path.join("logs", f"session_{session_id}.log") # Example path
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    return get_logger(output_file=log_file)

# --- Define Models for New Endpoints ---
class MiniQuizLogData(BaseModel):
    question: str
    selectedOption: str # Match frontend naming
    correctOption: str # Match frontend naming
    isCorrect: bool    # Match frontend naming
    relatedSection: Optional[str] = None
    topic: Optional[str] = None

# --- Endpoints ---

@router.post(
    "/sessions/{session_id}/documents",
    response_model=DocumentUploadResponse,
    summary="Upload Documents and Trigger Analysis",
    tags=["Tutoring Workflow"]
)
async def upload_session_documents(
    session_id: UUID, # Expect UUID
    request: Request, # Get request object directly
    files: List[UploadFile] = File(...),
    supabase: Client = Depends(get_supabase_client),
    tutor_context: TutorContext = Depends(get_tutor_context) # Fetch current context
):
    """
    Uploads one or more documents to the specified session.
    Stores files temporarily, uploads them to Supabase Storage & OpenAI, adds to vector store,
    and synchronously triggers document analysis.
    """
    user: User = request.state.user # Get user from request state populated by verify_token dependency
    logger = get_session_logger(session_id)
    folder_id = tutor_context.folder_id
    if not folder_id:
        logger.log_error("UploadError", "Session context is missing folder_id.")
        raise HTTPException(status_code=400, detail="Session context is missing folder information.")
    file_upload_manager = FileUploadManager(supabase) # Pass Supabase client
    uploaded_filenames = []
    temp_file_paths = []
    vector_store_id = tutor_context.vector_store_id
    existing_files = tutor_context.uploaded_file_paths

    for file in files:
        filename = file.filename
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{session_id}_{filename}")
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_file_paths.append(temp_file_path)
            uploaded_filenames.append(filename)
        except Exception as e:
            logger.log_error("FileUpload", e)
            raise HTTPException(status_code=500, detail=f"Failed to save file {filename}: {e}")
        finally:
            file.file.close()

    if not temp_file_paths:
        raise HTTPException(status_code=400, detail="No files were successfully saved.")

    # Upload to OpenAI and add to Vector Store
    message = ""
    try:
        for i, temp_path in enumerate(temp_file_paths):
            # Pass user_id and folder_id to file upload manager
            uploaded_file = await file_upload_manager.upload_and_process_file(
                file_path=temp_path,
                user_id=user.id,
                folder_id=folder_id,
                existing_vector_store_id=vector_store_id # Pass current VS ID
            )
            if not vector_store_id:
                vector_store_id = uploaded_file.vector_store_id
            message += f"Uploaded {uploaded_filenames[i]} (Supabase: {uploaded_file.supabase_path}, OpenAI ID: {uploaded_file.file_id}). "

        if vector_store_id:
            # Update context object and save it
            tutor_context.vector_store_id = vector_store_id
            tutor_context.uploaded_file_paths.extend(uploaded_filenames) # Append new files
            await _update_context_in_db(session_id, user.id, tutor_context, supabase)
            message += f"Vector Store ID: {vector_store_id}. "
        else:
            raise HTTPException(status_code=500, detail="Failed to create or retrieve vector store ID.")

    except Exception as e:
        logger.log_error("VectorStoreUpload", e)
        # Clean up any temp files on failure
        for temp_path in temp_file_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to upload files to OpenAI: {e}")

    # Trigger analysis synchronously
    analysis_status = "failed"
    try:
        print(f"Starting synchronous analysis for session {session_id}, vs_id {vector_store_id}")
        # Create context for analysis call (might be simpler than full TutorContext)
        # context already fetched via Depends
        user: User = request.state.user

        analysis_result: Optional[AnalysisResult] = await analyze_documents(
            vector_store_id,
            context=tutor_context,
            supabase=supabase # Pass supabase client to save KB
        )

        if analysis_result:
            analysis_status = "completed"
            # Store analysis result (as dict or object) - Keep storing the object
            tutor_context.analysis_result = analysis_result # Store the Pydantic object
            await _update_context_in_db(session_id, user.id, tutor_context, supabase)
            message += "Analysis completed."

        else:
            message += "Analysis finished but produced no result."
            analysis_status = "completed_empty" # Or "failed" depending on expectation

    except Exception as e:
        logger.log_error("AnalysisTrigger", e)
        message += f"Analysis trigger failed: {e}"
        analysis_status = "failed"
        # Don't raise HTTPException here, report status in response

    return DocumentUploadResponse(
        vector_store_id=vector_store_id,
        files_received=uploaded_filenames,
        analysis_status=analysis_status,
        message=message
    )

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
    """
    Main interaction endpoint for the AI Tutor.
    Handles both regular interactions and long-running tool pauses.
    """
    user: User = request.state.user
    logger = get_session_logger(session_id)
    print(f"\n=== Received /interact for session {session_id} ===")
    print(f"Input Type: {interaction_input.type}, Data: {interaction_input.data}")

    # Initialize ADK components
    orchestrator_agent = create_orchestrator_agent()
    adk_runner = Runner("ai_tutor", orchestrator_agent, session_service)
    run_config = RunConfig(workflow_name="Tutor_Interaction", group_id=str(session_id))

    last_agent_event: Optional[Event] = None
    question_for_user: Optional[QuizQuestion] = None
    paused_tool_call_id: Optional[str] = None

    try:
        # Process events from the ADK Runner
        async for event in adk_runner.run_async(
            user_id=str(user.id),
            session_id=str(session_id),
            new_message=interaction_input.data.get("message", ""),
            run_config=run_config
        ):
            print(f"[Interact] Received Event: ID={event.id}, Author={event.author}, Actions={event.actions}")
            logger.log_orchestrator_output(event.content)
            last_agent_event = event

            # Check for pause/input request from long-running tool
            if event.actions and event.actions.custom_action:
                custom_action = event.actions.custom_action
                if custom_action.get("signal") == "request_user_input":
                    question_data = custom_action.get("question_data")
                    paused_tool_call_id = custom_action.get("tool_call_id")
                    if question_data and paused_tool_call_id:
                        try:
                            question_for_user = QuizQuestion.model_validate(question_data)
                            logger.info(f"Detected pause signal from tool_call_id {paused_tool_call_id}.")
                            print(f"[Interact] Detected pause request. Sending question to user: {question_for_user.question[:50]}...")
                            break
                        except Exception as parse_err:
                            logger.error(f"PauseSignalParse: Failed to parse question from pause signal event {event.id}: {parse_err}")

    except Exception as run_err:
        error_msg = f"Error during ADK Runner execution: {run_err}"
        logger.log_error("ADKRunnerExecution", error_msg)
        print(f"[Interact] {error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal error during agent execution.")

    # Load final context state after the run/pause
    final_session = session_service.get_session("ai_tutor", str(user.id), str(session_id))
    if final_session and final_session.state:
        final_context = TutorContext.model_validate(final_session.state)
        # Store paused tool call ID if we paused
        if paused_tool_call_id:
            final_context.user_model_state.pending_interaction_type = 'checking_question'
            final_context.user_model_state.pending_interaction_details = {"paused_tool_call_id": paused_tool_call_id}
            await _update_context_in_db(session_id, user.id, final_context, supabase)
            logger.info(f"Stored paused_tool_call_id {paused_tool_call_id} in context for session {session_id}")
    else:
        logger.error(f"ContextFetchAfterRun: Failed to fetch session state after run/pause for {session_id}")
        final_context = tutor_context

    # Handle paused state (if question_for_user was set)
    if question_for_user:
        final_response_data = QuestionResponse(
            response_type="question",
            question=question_for_user,
            topic=final_context.current_teaching_topic or "Current Focus"
        )
        print(f"[Interact] Responding with question. Session {session_id} is now paused waiting for answer.")

    # Handle completed state (no pause detected)
    elif last_agent_event and last_agent_event.content:
        response_text = last_agent_event.content.parts[0].text if last_agent_event.content.parts else "Interaction complete."
        final_response_data = MessageResponse(response_type="message", text=response_text)
        print(f"[Interact] Orchestrator returned response of type: {final_response_data.response_type}")

    else:
        error_msg = "No response received from agent."
        logger.log_error("NoAgentResponse", error_msg)
        final_response_data = ErrorResponse(
            response_type="error",
            error=error_msg,
            message="There was an internal error processing your request."
        )

    return InteractionResponseData(
        content_type=final_response_data.response_type,
        data=final_response_data,
        user_model_state=final_context.user_model_state
    )

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
    """
    Handles submitting an answer when the tutor is paused waiting for user input.
    Creates a FunctionResponse event to resume the paused tool execution.
    """
    user: User = request.state.user
    logger = get_session_logger(session_id)
    print(f"\n=== Received /answer for session {session_id} ===")
    print(f"Input Type: {interaction_input.type}, Data: {interaction_input.data}")

    if interaction_input.type != 'answer' or 'answer_index' not in interaction_input.data:
        raise HTTPException(
            status_code=400,
            detail="Invalid input type or data for /answer endpoint. Expected type='answer' and data={'answer_index': number}."
        )

    # Retrieve details about the paused interaction
    if (tutor_context.user_model_state.pending_interaction_type != 'checking_question' or
        not tutor_context.user_model_state.pending_interaction_details or
        'paused_tool_call_id' not in tutor_context.user_model_state.pending_interaction_details):
        logger.warning(f"Received answer for session {session_id}, but no valid pending interaction found in context.")
        raise HTTPException(status_code=400, detail="No pending question found for this session.")

    paused_tool_call_id = tutor_context.user_model_state.pending_interaction_details['paused_tool_call_id']
    user_answer_index = interaction_input.data['answer_index']
    logger.info(f"Resuming tool call {paused_tool_call_id} with answer index {user_answer_index}")

    # Create the FunctionResponse event
    answer_event = Event(
        author="user",
        content=adk_types.Content(
            role="tool",
            parts=[
                adk_types.Part.from_function_response(
                    name="ask_user_question_and_get_answer",
                    id=paused_tool_call_id,
                    response={"answer_index": user_answer_index}
                )
            ]
        ),
        invocation_id=tutor_context.last_interaction_summary or f"resume_{session_id}"
    )

    # Clear pending state in context BEFORE resuming run
    tutor_context.user_model_state.pending_interaction_type = None
    tutor_context.user_model_state.pending_interaction_details = None
    await _update_context_in_db(session_id, user.id, tutor_context, supabase)

    # Resume the ADK Runner
    orchestrator_agent = create_orchestrator_agent()
    adk_runner = Runner("ai_tutor", orchestrator_agent, session_service)
    run_config = RunConfig(workflow_name="Tutor_Interaction_Resume", group_id=str(session_id))

    try:
        last_agent_event_after_resume: Optional[Event] = None
        question_after_resume: Optional[QuizQuestion] = None
        paused_id_after_resume: Optional[str] = None

        async for event in adk_runner.run_async(
            user_id=str(user.id),
            session_id=str(session_id),
            new_message=answer_event,
            run_config=run_config
        ):
            print(f"[Answer] Received Event after resume: ID={event.id}, Author={event.author}")
            logger.log_orchestrator_output(event.content)
            last_agent_event_after_resume = event

            # Check for another pause signal immediately after resume
            if event.actions and event.actions.custom_action:
                custom_action = event.actions.custom_action
                if custom_action.get("signal") == "request_user_input":
                    question_data = custom_action.get("question_data")
                    paused_id_after_resume = custom_action.get("tool_call_id")
                    if question_data and paused_id_after_resume:
                        try:
                            question_after_resume = QuizQuestion.model_validate(question_data)
                            logger.info(f"Detected another pause signal (ID: {paused_id_after_resume}) immediately after resuming.")
                            break
                        except Exception as parse_err:
                            logger.error(f"PauseSignalParse (Resume): Failed to parse question: {parse_err}")

    except Exception as resume_err:
        error_msg = f"Error during agent run after resume: {resume_err}"
        logger.log_error("ADKRunnerResume", error_msg)
        print(f"[Answer] {error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal error resuming agent execution.")

    # Load final context state after the resume
    final_session_after_resume = session_service.get_session("ai_tutor", str(user.id), str(session_id))
    if final_session_after_resume and final_session_after_resume.state:
        final_context_after_resume = TutorContext.model_validate(final_session_after_resume.state)
        # Store new paused tool call ID if another pause occurred
        if paused_id_after_resume:
            final_context_after_resume.user_model_state.pending_interaction_type = 'checking_question'
            final_context_after_resume.user_model_state.pending_interaction_details = {"paused_tool_call_id": paused_id_after_resume}
            await _update_context_in_db(session_id, user.id, final_context_after_resume, supabase)
            logger.info(f"Stored new paused_tool_call_id {paused_id_after_resume} after resume.")
    else:
        logger.error(f"ContextFetchAfterResume: Failed to fetch session state after resume for {session_id}")
        final_context_after_resume = tutor_context

    # Format response based on whether it paused again or completed
    if question_after_resume:
        final_response_data_after_resume = QuestionResponse(
            response_type="question",
            question=question_after_resume,
            topic=final_context_after_resume.current_teaching_topic or "Current Focus"
        )
        print(f"[Answer] Responding with NEW question. Session {session_id} paused again.")
    elif last_agent_event_after_resume and last_agent_event_after_resume.content:
        response_text = last_agent_event_after_resume.content.parts[0].text if last_agent_event_after_resume.content.parts else "Processing complete."
        final_response_data_after_resume = MessageResponse(response_type="message", text=response_text)
        print(f"[Answer] Interaction completed after resume. Final response type: {final_response_data_after_resume.response_type}")
    else:
        error_msg = "Agent interaction finished unexpectedly after resuming."
        print(f"[Answer] Error: {error_msg}")
        final_response_data_after_resume = ErrorResponse(
            response_type="error",
            error=error_msg,
            message="Internal processing error after submitting answer."
        )

    return InteractionResponseData(
        content_type=final_response_data_after_resume.response_type,
        data=final_response_data_after_resume,
        user_model_state=final_context_after_resume.user_model_state
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
        if response.error:
            print(f"Error updating context in DB for session {session_id}: {response.error}")
            return False
        return True
    except Exception as e:
        print(f"Exception updating context in DB for session {session_id}: {e}")
        return False

# TODO: Implement endpoint for session analysis if needed:
# POST /sessions/{session_id}/analyze-session (Full Session Analysis) 