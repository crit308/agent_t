from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form, Body, Request
import os
import shutil
import time
import json
import traceback  # Add traceback import
import logging  # Add logging import
from supabase import Client
from gotrue.types import User # To type hint the user object
from uuid import UUID

from ai_tutor.session_manager import SessionManager
from ai_tutor.tools.file_upload import FileUploadManager
from ai_tutor.agents.analyzer_agent import analyze_documents
from ai_tutor.agents.models import (
    FocusObjective,
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis
)
from ai_tutor.api_models import (
    DocumentUploadResponse, AnalysisResponse, TutorInteractionResponse,
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse,
    InteractionRequestData, InteractionResponseData  # Add InteractionResponseData
)
from ai_tutor.context import TutorContext
from ai_tutor.output_logger import get_logger, TutorOutputLogger
from pydantic import BaseModel
from ai_tutor.dependencies import get_supabase_client # Get supabase client dependency
from ai_tutor.auth import verify_token # Get auth dependency
from ai_tutor.core.schema import PlannerOutput

router = APIRouter()
session_manager = SessionManager()

# Directory for temporary file uploads
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- Dependency to get TutorContext from DB ---
async def get_tutor_context(
    session_id: UUID, # Expect UUID
    request: Request, # Access user from request state
    supabase: Client = Depends(get_supabase_client)
) -> TutorContext:
    user: User = request.state.user # Get authenticated user
    context = await session_manager.get_session_context(supabase, session_id, user.id)
    if not context:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or not authorized for user.")
    return context

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
    response_model=None,
    summary="Upload Documents and Trigger Analysis",
    tags=["Tutoring Workflow"]
)
async def upload_session_documents(
    session_id: UUID,
    request: Request,
    files: List[UploadFile] = File(...),
    supabase: Client = Depends(get_supabase_client),
    tutor_context: TutorContext = Depends(get_tutor_context)
):
    """
    Uploads documents, embeds them into a vector store, analyzes content,
    and synchronously updates the session context for planning.
    """
    user: User = request.state.user
    logger = get_session_logger(session_id)
    folder_id = tutor_context.folder_id
    if not folder_id:
        logger.log_error("UploadError", "Session context is missing folder_id.")
        raise HTTPException(status_code=400, detail="Missing folder information in session context.")

    # Initialize the file upload manager for embeddings
    file_upload_manager = FileUploadManager(supabase)

    # Save uploaded files temporarily
    temp_paths: List[str] = []
    filenames: List[str] = []
    for upload in files:
        temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{session_id}_{upload.filename}")
        try:
            with open(temp_path, "wb") as buf:
                shutil.copyfileobj(upload.file, buf)
            temp_paths.append(temp_path)
            filenames.append(upload.filename)
        except Exception as e:
            logger.log_error("FileSaveError", e)
            raise HTTPException(status_code=500, detail=f"Failed to save {upload.filename}: {e}")
        finally:
            upload.file.close()
    if not temp_paths:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    # Embed files into vector store synchronously
    vector_store_id = tutor_context.vector_store_id
    messages: List[str] = []
    try:
        for path, name in zip(temp_paths, filenames):
            result = await file_upload_manager.upload_and_process_file(
                file_path=path,
                user_id=user.id,
                folder_id=folder_id,
                existing_vector_store_id=vector_store_id
            )
            if result.vector_store_id:
                vector_store_id = result.vector_store_id
            messages.append(f"{name} embedded into {vector_store_id}")
    except Exception as e:
        logger.log_error("EmbedError", e)
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # Update session context after embedding
    tutor_context.uploaded_file_paths.extend(filenames)
    tutor_context.vector_store_id = vector_store_id
    await session_manager.update_session_context(supabase, session_id, user.id, tutor_context)

    # Perform document analysis synchronously
    try:
        analysis = await analyze_documents(vector_store_id, context=tutor_context, supabase=supabase)
        tutor_context.analysis_result = analysis
        analysis_status = "completed"
    except Exception as e:
        logger.log_error("AnalysisError", e)
        analysis_status = "pending"

    # Persist context after analysis
    await session_manager.update_session_context(supabase, session_id, user.id, tutor_context)

    return DocumentUploadResponse(
        vector_store_id=vector_store_id,
        files_received=filenames,
        analysis_status=analysis_status,
        message="; ".join(messages)
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


@router.post(
    "/sessions/{session_id}/plan",
    response_model=None,
    summary="DEPRECATED: Generate Lesson Plan (use /interact endpoint instead)",
    deprecated=True,
    dependencies=[Depends(verify_token)],
    tags=["Tutoring Workflow"]
)
async def generate_session_lesson_plan(
    session_id: UUID, # Expect UUID
    request: Request, # Add request parameter
    tutor_context: TutorContext = Depends(get_tutor_context) # Use parsed context
):
    """
    DEPRECATED: Use /interact endpoint instead.
    This endpoint triggers the Orchestrator to get the initial focus objective from the Planner.
    The focus objective is stored in the session context and returned.
    """
    logger = get_session_logger(session_id)
    # Get user from request state
    user: User = request.state.user

    # Get vector store ID and analysis result from context
    vector_store_id = tutor_context.vector_store_id
    # analysis_result = tutor_context.analysis_result # No longer directly needed by this endpoint
    # folder_id = tutor_context.folder_id # No longer directly needed by this endpoint

    # Initial checks are good to keep, ensure data looks okay before agent creation
    if not vector_store_id:
        raise HTTPException(status_code=400, detail="Documents must be uploaded first.")
    # Remove KB check, planner tool handles reading it

    # --- Wrap the main logic in a try...except block ---
    try:
        # Use direct planner for next focus objective
        from ai_tutor.agents.planner_agent import run_planner
        planner_output = await run_planner(tutor_context)

        # Log the planner output (list of objectives)
        logger.log_planner_output(planner_output)
        supabase: Client = await get_supabase_client()
        success = await session_manager.update_session_context(supabase, session_id, user.id, tutor_context)
        if not success:
            logger.log_error("SessionUpdate", f"Failed to update session {session_id} with focus objective.")

        return planner_output

    except Exception as e:
        # --- Explicit Exception Catching and Logging ---
        logger.log_error("PlannerAgentRun", e)
        error_type = type(e).__name__
        error_details = str(e)
        error_traceback = traceback.format_exc()

        print("\n!!! EXCEPTION IN /plan Endpoint !!!") # Make log stand out
        print(f"Error Type: {error_type}")
        print(f"Error Details: {error_details}")
        print("Full Traceback:")
        print(error_traceback)
        # -------------------------------------------------

        # Raise a generic 500, but the logs now contain the details. The detail message could be improved.
        raise HTTPException(status_code=500, detail=f"Failed to get initial focus due to internal error: {error_type}")

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
    response_model=None,  # Disable automatic response model to avoid invalid field type
    dependencies=[Depends(verify_token)], # Add auth dependency
    summary="Interact with the AI Tutor",
    tags=["Tutoring Workflow"]
)
async def interact_with_tutor(
    session_id: UUID, # Expect UUID
    request: Request, # Get request directly
    interaction_input: InteractionRequestData = Body(...),
    supabase: Client = Depends(get_supabase_client), # To save context
    tutor_context: TutorContext = Depends(get_tutor_context)
):
    logger = get_session_logger(session_id)
    print(f"\n=== Starting /interact for session {session_id} ===")
    print(f"[Interact] Input Type: {interaction_input.type}, Data: {interaction_input.data}")
    print(f"[Interact] Context BEFORE Orchestrator: pending={tutor_context.user_model_state.pending_interaction_type}, topic='{tutor_context.current_teaching_topic}', segment={tutor_context.user_model_state.current_topic_segment_index}")

    user: User = request.state.user
    # Import TutorFSM here to avoid circular import at module load
    from ai_tutor.fsm import TutorFSM
    # Invoke the new FSM wrapper
    last_event = {"type": interaction_input.type, "data": interaction_input.data or {}}
    fsm = TutorFSM(tutor_context)
    try:
        final_response_data = await fsm.on_user_message(last_event)
    except Exception as exc:
        # Log TutorFSM errors using our output logger
        logger.log_error("TutorFSM", exc)
        # Return a serializable error response using ErrorResponse structure
        error_data = ErrorResponse(response_type="error", message=f"Internal FSM error: {str(exc)}")
        # Need to wrap this in InteractionResponseData
        return InteractionResponseData(content_type="error", data=error_data, user_model_state=tutor_context.user_model_state)

    # --- Save Context AFTER determining the final response ---
    # Persist last_event and pending_interaction_type for session resume
    # TODO: Review if last_event and pending_interaction_type are still the right things to save from TutorContext
    # self.ctx seems to hold the definitive state now.
    # tutor_context.last_event = last_event # Already set in on_user_message
    # tutor_context.pending_interaction_type = tutor_context.user_model_state.pending_interaction_type # State is now in ctx.state
    print(f"[Interact] Saving final context state to Supabase for session {session_id}")
    await session_manager.update_session_context(supabase, session_id, user.id, tutor_context)
    print(f"[Interact] Context saved AFTER run: state={tutor_context.state}, topic='{tutor_context.current_teaching_topic}', segment={tutor_context.user_model_state.current_topic_segment_index}")

    # Build response based on what FSM returned
    if hasattr(final_response_data, 'response_type'):
        # Case 1: FSM returned a Pydantic model (e.g., ExplanationResponse, QuestionResponse from ExecutorAgent.run -> CONTINUE)
        content_type = final_response_data.response_type
        data = final_response_data
    elif isinstance(final_response_data, dict) and 'response_type' in final_response_data:
        # Case 2: FSM returned a dictionary with 'response_type' (e.g., error or message dicts from FSM itself)
        content_type = final_response_data['response_type'] # Get the type ('error', 'message')
        # Try to parse the dict into a known Pydantic model for consistency
        try:
            if content_type == "error":
                 # Use the ErrorResponse model for structure
                 data = ErrorResponse(**final_response_data)
            elif content_type == "message":
                 # Use the MessageResponse model for structure
                 data = MessageResponse(**final_response_data)
            else:
                 # Unknown response_type from FSM dict, pass raw dict (shouldn't happen ideally)
                 print(f"[Interact Warning] Unknown response_type '{content_type}' in dict from FSM. Passing raw dict.")
                 data = final_response_data
        except Exception as parse_exc:
            # Handle cases where the dict doesn't match the Pydantic model structure
            print(f"[Interact Error] Failed to parse FSM dict into {content_type} model: {parse_exc}. Dict: {final_response_data}")
            content_type = "error"
            data = ErrorResponse(response_type="error", message="Internal error processing tutor response.")
    else:
        # Case 3: Fallback for unexpected return type from FSM (e.g., None, str, different dict)
        print(f"[Interact Warning] Unexpected return type from FSM: {type(final_response_data)}. Value: {final_response_data}. Defaulting to message.")
        content_type = "message" # Default to message
        # Use MessageResponse model
        data = MessageResponse(response_type="message", text=f"Unexpected response from tutor: {str(final_response_data)}")

    # Construct the final API response
    return InteractionResponseData(
        content_type=content_type, # Use the determined content_type
        data=data, # Use the (potentially parsed) data
        user_model_state=tutor_context.user_model_state
    )

# --- Remove POST /quiz/submit (Legacy) ---
# Quiz answers are now handled via the /interact endpoint.
# An end-of-session quiz submission could potentially be added back later
# if needed, but the core loop uses /interact.

# TODO: Implement endpoint for session analysis if needed:
# POST /sessions/{session_id}/analyze-session (Full Session Analysis) 