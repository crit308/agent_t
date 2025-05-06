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
from ai_tutor.utils.file_upload import FileUploadManager
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
logger = logging.getLogger(__name__) # Use standard logging

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
    logger.info(f"Starting embedding files for session {session_id}")
    folder_id = tutor_context.folder_id
    if not folder_id:
        logger.error("UploadError", "Session context is missing folder_id.")
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
            logger.error("FileSaveError", e)
            raise HTTPException(status_code=500, detail=f"Failed to save {upload.filename}: {e}")
        finally:
            upload.file.close()
    if not temp_paths:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    # --- Section for Embedding and Analysis ---
    embedding_successful = False
    analysis_status = "pending" # Default status
    try:
        # Embed files into vector store synchronously
        logger.info(f"Starting embedding files for session {session_id}")
        vector_store_id = tutor_context.vector_store_id
        messages: List[str] = []
        for path, name in zip(temp_paths, filenames):
            logger.info(f"Calling upload_and_process_file for {name}")
            # This call now includes polling and can raise TimeoutError or other exceptions
            result = await file_upload_manager.upload_and_process_file(
                file_path=path,
                user_id=user.id,
                folder_id=folder_id,
                existing_vector_store_id=vector_store_id
            )
            logger.info(f"upload_and_process_file returned for {name}: {result}")
            if result.vector_store_id:
                vector_store_id = result.vector_store_id # Update vector store ID if created
            messages.append(f"{name} processed successfully.")

        embedding_successful = True # Mark embedding as successful if loop completes

        # Update session context after successful embedding
        tutor_context.uploaded_file_paths.extend(filenames)
        tutor_context.vector_store_id = vector_store_id
        await session_manager.update_session_context(supabase, session_id, user.id, tutor_context)

        # Perform document analysis only if embedding was successful
        logger.info(f"Calling analyze_documents for vector_store_id {vector_store_id}")
        try:
            analysis = await analyze_documents(vector_store_id, context=tutor_context, supabase=supabase)
            logger.info(f"analyze_documents returned: {analysis}")
            tutor_context.analysis_result = analysis
            analysis_status = "completed"
        except Exception as analysis_exc:
            logger.error("AnalysisError", analysis_exc)
            logger.info(f"Exception during analysis: {analysis_exc}")
            analysis_status = "failed" # Mark analysis as failed
            messages.append(f"Document analysis failed: {analysis_exc}")
            # Optionally re-raise or handle differently

    except TimeoutError as te:
        logger.error("EmbeddingTimeout", te)
        logger.info(f"Timeout during embedding file processing: {te}")
        analysis_status = "timeout"
        messages.append(f"File processing timed out: {te}")
        # Optionally re-raise or handle differently
    except Exception as e:
        logger.error("EmbeddingError", e)
        logger.info(f"Exception during embedding: {e}")
        analysis_status = "failed"
        messages.append(f"File embedding failed: {e}")
        # Optionally re-raise or handle differently

    finally:
        # Clean up temporary files regardless of outcome
        for p in temp_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                    logger.info(f"Removed temporary file: {p}")
                except OSError as remove_err:
                    logger.warning(f"Could not remove temporary file {p}: {remove_err}")

    # Persist context after analysis (or analysis attempt)
    # Ensures vector_store_id and analysis_result (even if None/error) are saved
    try:
        await session_manager.update_session_context(supabase, session_id, user.id, tutor_context)
        logger.info(f"Final session context update successful for {session_id}")
    except Exception as update_exc:
        logger.error(f"Failed to perform final context update for session {session_id}: {update_exc}")
        # If this fails, the state might be inconsistent. Consider implications.

    logger.info(f"Returning DocumentUploadResponse for session {session_id} with analysis_status: {analysis_status}")
    return DocumentUploadResponse(
        vector_store_id=tutor_context.vector_store_id, # Use context value which might have been updated
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
    tutor_context: TutorContext = Depends(get_tutor_context), # Use parsed context
    supabase: Client = Depends(get_supabase_client) # Add supabase dependency
):
    user: User = request.state.user
    print(f"\n=== ENTERING /plan Endpoint for session {session_id} ===") # Log entry
    logger.info(f"Attempting to generate plan for session {session_id}, user {user.id}")

    try:
        # --- Original logic of the function ---
        vector_store_id = tutor_context.vector_store_id
        if not vector_store_id:
            logger.error(f"/plan error: No vector store ID found for session {session_id}")
            raise HTTPException(status_code=400, detail="Documents must be uploaded first.")

        # Import run_planner locally if not already imported at top
        from ai_tutor.agents.planner_agent import run_planner

        print(f"Calling run_planner for session {session_id}")
        logger.info(f"Calling run_planner for session {session_id}")
        planner_output = await run_planner(tutor_context)
        print(f"run_planner completed for session {session_id}")
        logger.info(f"run_planner completed for session {session_id}")

        # Log the planner output (careful with large objects)
        # session_logger.log_planner_output(planner_output) # Use standard logger
        logger.info(f"Planner output generated for {session_id}") # Log success

        # Persist context (ensure supabase client is available)
        print(f"Attempting to update context after planning for {session_id}")
        logger.info(f"Attempting to update context after planning for {session_id}")
        success = await session_manager.update_session_context(supabase, session_id, user.id, tutor_context)
        if not success:
            logger.error(f"Failed to update session {session_id} context after planning.")
            # Decide if this should be a 500 error
            # raise HTTPException(status_code=500, detail="Failed to save session state after planning.")
        else:
             logger.info(f"Context updated successfully after planning for {session_id}")


        print(f"=== EXITING /plan Endpoint SUCCESSFULLY for session {session_id} ===")
        logger.info(f"Successfully exiting /plan endpoint for session {session_id}")
        return planner_output # Return the result

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        logger.error(f"HTTPException in /plan for {session_id}: Status={http_exc.status_code}, Detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        # --- Catch ANY other exception ---
        error_type = type(e).__name__
        error_details = str(e)
        error_traceback = traceback.format_exc()

        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! UNHANDLED EXCEPTION in /plan Endpoint for Session {session_id} !!!")
        print(f"Error Type: {error_type}")
        print(f"Error Details: {error_details}")
        print("Full Traceback:")
        print(error_traceback)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        # Also log using standard logger
        logger.critical(f"Unhandled exception in /plan for session {session_id}: {error_type}: {error_details}\n{error_traceback}", exc_info=True)

        # Return a generic 500 error to the client
        raise HTTPException(status_code=500, detail=f"Internal server error during planning: {error_type}")

@router.get(
    "/sessions/{session_id}/lesson",
    response_model=Optional[LessonContent], # Allow null if not generated yet
    dependencies=[Depends(verify_token)], # Add auth dependency
    summary="Retrieve Generated Lesson Content",
    tags=["Tutoring Workflow"]
)
async def get_session_lesson_content(session_id: UUID, tutor_context: TutorContext = Depends(get_tutor_context)):
    """Retrieves the generated lesson content for the session."""
    logger.info(f"Retrieving lesson content for session {session_id}")
    content_data = tutor_context.lesson_content # Example assuming it's stored directly

    if not content_data:
        logger.error("GetLessonContent", f"Lesson content not found in session state for {session_id}")
        # Return None or 404 - let's return None first to see how frontend handles it
        # raise HTTPException(status_code=404, detail="Lesson content not yet generated or available.")
        return None # Or return an empty LessonContent structure if preferred

    try:
        # Assuming content_data is stored as a dict (from model_dump())
        lesson_content = content_data # If it's already parsed by TutorContext
        return lesson_content
    except Exception as e:
         logger.error("GetLessonContentParse", f"Failed to parse stored lesson content: {e}")
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
    logger.info(f"Retrieving quiz for session {session_id}")
    quiz_data = tutor_context.quiz # Get 'quiz' from context object if stored there

    if not quiz_data:
        logger.error("GetQuiz", f"Quiz not found in session state for {session_id}")
        # Return None - frontend useEffect should handle retries/errors
        return None

    try:
        # Assuming quiz_data is stored as a dict (from model_dump())
        quiz = quiz_data # If already parsed by TutorContext
        return quiz
    except Exception as e:
         logger.error("GetQuizParse", f"Failed to parse stored quiz: {e}")
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
    logger.info(f"Logging mini-quiz attempt for session {session_id}")

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
        logger.error("LogMiniQuiz", f"Failed to log mini-quiz attempt: {e}")
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
    logger.info(f"Logging user summary for session {session_id}")
    try:
        logger.log_user_summary(
            section_title=summary_data.section,
            topic=summary_data.topic,
            summary_text=summary_data.summary
        )
        return
    except Exception as e:
        logger.error("LogUserSummary", f"Failed to log user summary: {e}")
        return

# --- Interaction Endpoint removed (handled by WebSocket now) ---
# @router.post(
#     "/sessions/{session_id}/interact",
#     ...
# )
# async def interact_with_tutor(...):
#     ...
#     # Import TutorFSM here to avoid circular import at module load
#     # from ai_tutor.fsm import TutorFSM # <-- REMOVED IMPORT
#     # Invoke the new FSM wrapper
#     ...
#     # final_response_data = await fsm.on_user_message(last_event)
#     ...
#     # --- Save Context AFTER determining the final response ---
#     ...
#     # await session_manager.update_session_context(supabase, session_id, user.id, tutor_context)
#     ...
#     # Build response based on what FSM returned
#     ...
#     # Construct the final API response
#     ...

# --- Remove POST /quiz/submit (Legacy) ---
# Quiz answers are now handled via the /interact endpoint.
# An end-of-session quiz submission could potentially be added back later
# if needed, but the core loop uses /interact.

# TODO: Implement endpoint for session analysis if needed:
# POST /sessions/{session_id}/analyze-session (Full Session Analysis) 