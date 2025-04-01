from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form, Body
import os
import shutil
import time
import json
import traceback  # Add traceback import
import logging  # Add logging import

from ai_tutor.session_manager import SessionManager
from ai_tutor.tools.file_upload import FileUploadManager
from ai_tutor.agents import (
    analyze_documents,
    create_planner_agent,
    analyze_teaching_session,
    create_orchestrator_agent
)
from ai_tutor.agents.analyzer_agent import AnalysisResult
from ai_tutor.agents.models import (
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis
)
from ai_tutor.api_models import (
    DocumentUploadResponse, AnalysisResponse, TutorInteractionResponse,
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse,
    InteractionRequestData, InteractionResponseData  # Add InteractionResponseData
)
from ai_tutor.context import TutorContext
from ai_tutor.output_logger import get_logger, TutorOutputLogger
from agents import Runner, RunConfig, Agent
from ai_tutor.manager import AITutorManager
from pydantic import BaseModel

router = APIRouter()
session_manager = SessionManager()

# Directory for temporary file uploads
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- Dependency to get session state ---
async def get_session_state(session_id: str) -> Dict[str, Any]:
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

# --- Dependency to get parsed TutorContext ---
async def get_tutor_context(session_id: str) -> TutorContext:
    session_data = session_manager.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    try:
        return TutorContext(**session_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse TutorContext: {e}")

# --- Helper to get logger ---
def get_session_logger(session_id: str) -> TutorOutputLogger:
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
    session_id: str,
    files: List[UploadFile] = File(...),
    session: Dict[str, Any] = Depends(get_session_state) # Keep getting dict here
):
    """
    Uploads one or more documents to the specified session.
    Stores files temporarily, uploads them to OpenAI, adds them to a vector store,
    and synchronously triggers document analysis.
    """
    logger = get_session_logger(session_id)
    file_upload_manager = FileUploadManager() # Instantiates its own OpenAI client
    uploaded_filenames = []
    temp_file_paths = []
    vector_store_id = session.get("vector_store_id")
    existing_files = session.get("uploaded_files", [])

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
            uploaded_file = file_upload_manager.upload_and_process_file(temp_path)
            if not vector_store_id:
                vector_store_id = uploaded_file.vector_store_id
            message += f"Uploaded {uploaded_filenames[i]} (ID: {uploaded_file.file_id}). "

        if vector_store_id:
            session_manager.update_session(session_id, {
                "vector_store_id": vector_store_id,
                "uploaded_files": existing_files + uploaded_filenames
            })
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
        tutor_context = TutorContext(session_id=session_id, vector_store_id=vector_store_id)
        tutor_context.uploaded_file_paths = existing_files + uploaded_filenames

        analysis_result: Optional[AnalysisResult] = await analyze_documents(vector_store_id, context=tutor_context)

        if analysis_result:
            analysis_status = "completed"
            # Store the validated AnalysisResult object directly if SessionManager supports it,
            # otherwise use model_dump()
            analysis_data = analysis_result.model_dump(mode='json')
            session_manager.update_session(session_id, {"analysis_result": analysis_data})
            message += "Analysis completed."

            # --- Remove Knowledge Base file creation ---
            # The Orchestrator/Planner will access analysis via context or a tool
            # Remove kb_path and related logic from session state if not needed elsewhere
            print(f"Document analysis completed and stored in session state for {session_id}.")
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
    session_id: str,
    tutor_context: TutorContext = Depends(get_tutor_context) # Use parsed context
):
    """Retrieves the results of the document analysis for the session."""
    # Access directly from the parsed context object
    analysis_obj = tutor_context.analysis_result
    if analysis_obj:
        try:
            return AnalysisResponse(status="completed", analysis=analysis_obj)
        except Exception as e:
             return AnalysisResponse(status="error", error=f"Failed to parse analysis data: {e}")
    else:
        # Check if analysis failed or just hasn't run
        # This logic depends on how status is tracked (not fully implemented here)
        return AnalysisResponse(status="not_started", analysis=None)


@router.post(
    "/sessions/{session_id}/plan",
    response_model=LessonPlan, # Planner should only return a LessonPlan now
    summary="Generate Lesson Plan",
    tags=["Tutoring Workflow"]
)
async def generate_session_lesson_plan(
    session_id: str,
    tutor_context: TutorContext = Depends(get_tutor_context) # Use parsed context
):
    """
    Generates a lesson plan based on the analyzed documents stored in the session context.
    Stores the plan back into the session context.
    """
    logger = get_session_logger(session_id)

    # Get vector store ID and analysis result from context
    vector_store_id = tutor_context.vector_store_id
    analysis_result = tutor_context.analysis_result

    if not vector_store_id or not analysis_result:
        raise HTTPException(status_code=400, detail="Documents must be uploaded and analysis completed first.")

    try:
        # Ensure the planner agent DOES NOT handoff implicitly
        # Option 1: Modify create_planner_agent to optionally disable handoffs
        # Option 2: Create a separate create_planner_agent_no_handoff
        # Option 3: Assume the Runner call won't trigger handoffs if the agent is designed correctly
        # Let's assume Option 3 for now, but this might need adjustment
        planner_agent: Agent[TutorContext] = create_planner_agent(vector_store_id)

        # The planner agent's instructions need modification to use context/tools for analysis info
        # Pass the full TutorContext to the Runner
        run_config = RunConfig(workflow_name="AI Tutor API - Planning", group_id=session_id)

        # The prompt should guide the planner based on analysis available in context
        # Example: "Create a lesson plan based on the document analysis provided in the context."
        plan_prompt = "Create a lesson plan based on the analyzed documents."

        result = await Runner.run(planner_agent, plan_prompt, run_config=run_config, context=tutor_context)

        lesson_plan_output = result.final_output

        if isinstance(lesson_plan_output, LessonPlan):
            lesson_plan_obj = lesson_plan_output
            logger.log_planner_output(lesson_plan_obj)

            # --- Store generated plan in session state ---
            update_data = {"lesson_plan": lesson_plan_obj.model_dump(mode='json')}
            success = session_manager.update_session(session_id, update_data)
            if not success:
                 logger.log_error("SessionUpdate", f"Failed to update session {session_id} with lesson plan.")

            return lesson_plan_obj # Return the generated plan
        else:
            logger.log_error("PlannerAgentRun", f"Planner agent returned unexpected type: {type(lesson_plan_output)}")
            raise HTTPException(status_code=500, detail="Planner agent failed to return a valid LessonPlan.")

    except Exception as e:
        logger.log_error("PlannerAgentRun", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson plan: {e}")

@router.get(
    "/sessions/{session_id}/lesson",
    response_model=Optional[LessonContent], # Allow null if not generated yet
    summary="Retrieve Generated Lesson Content",
    tags=["Tutoring Workflow"]
)
async def get_session_lesson_content(session_id: str, session: dict = Depends(get_session_state)):
    """Retrieves the generated lesson content for the session."""
    logger = get_session_logger(session_id)
    content_data = session.get("lesson_content") # Get from session state

    if not content_data:
        logger.log_error("GetLessonContent", f"Lesson content not found in session state for {session_id}")
        # Return None or 404 - let's return None first to see how frontend handles it
        # raise HTTPException(status_code=404, detail="Lesson content not yet generated or available.")
        return None # Or return an empty LessonContent structure if preferred

    try:
        # Assuming content_data is stored as a dict (from model_dump())
        lesson_content = LessonContent(**content_data)
        return lesson_content
    except Exception as e:
         logger.log_error("GetLessonContentParse", f"Failed to parse stored lesson content: {e}")
         raise HTTPException(status_code=500, detail="Failed to retrieve/parse lesson content.")

@router.get(
    "/sessions/{session_id}/quiz",
    response_model=Optional[Quiz], # Return Quiz or null
    summary="Retrieve Generated Quiz",
    tags=["Tutoring Workflow"]
)
async def get_session_quiz(session_id: str, session: dict = Depends(get_session_state)):
    """Retrieves the generated quiz for the session."""
    logger = get_session_logger(session_id)
    quiz_data = session.get("quiz") # Get 'quiz' from session state

    if not quiz_data:
        logger.log_error("GetQuiz", f"Quiz not found in session state for {session_id}")
        # Return None - frontend useEffect should handle retries/errors
        return None

    try:
        # Assuming quiz_data is stored as a dict (from model_dump())
        quiz = Quiz(**quiz_data)
        return quiz
    except Exception as e:
         logger.log_error("GetQuizParse", f"Failed to parse stored quiz: {e}")
         raise HTTPException(status_code=500, detail="Failed to retrieve/parse quiz.")

@router.post(
    "/sessions/{session_id}/log/mini-quiz",
    status_code=204, # No content to return
    summary="Log Mini-Quiz Attempt",
    tags=["Logging"]
)
async def log_mini_quiz_event(session_id: str, attempt_data: MiniQuizLogData = Body(...), session: dict = Depends(get_session_state)):
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
    summary="Log User Summary Attempt",
    tags=["Logging"]
)
async def log_user_summary_event(session_id: str, summary_data: UserSummaryLogData = Body(...), session: dict = Depends(get_session_state)):
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
    response_model=InteractionResponseData,  # Change response model
    summary="Interact with the AI Tutor",
    tags=["Tutoring Workflow"]
)
async def interact_with_tutor(
    session_id: str,
    interaction_input: InteractionRequestData = Body(...),
    tutor_context: TutorContext = Depends(get_tutor_context)
):
    """
    Handles user interaction with the AI tutor - can be an explanation request,
    question, or feedback. Updates session state and returns wrapped response.
    """
    logger = get_session_logger(session_id)
    print(f"\n=== Starting /interact for session {session_id} ===")
    print(f"Interaction Type: {interaction_input.type}")
    print(f"Interaction Data: {json.dumps(interaction_input.data, indent=2)}")
    
    logger.log_user_input(f"Type: {interaction_input.type}, Data: {interaction_input.data}")

    # Check prerequisites (plan must exist)
    if not tutor_context.lesson_plan:
        logger.log_error("Prerequisites", "No lesson plan found")
        raise HTTPException(status_code=400, detail="Lesson plan must be generated first")

    # Prepare input for orchestrator
    orchestrator_input_text = f"Type: {interaction_input.type}"
    orchestrator_input_text += f" | Data: {json.dumps(interaction_input.data)}"
    print(f"\nPreparing orchestrator input:\n{orchestrator_input_text}")

    try:
        print("\n--- Creating orchestrator agent ---")
        orchestrator_agent: Agent[TutorContext] = create_orchestrator_agent()
        run_config = RunConfig(workflow_name="AI Tutor API - Interaction", group_id=session_id)
        
        print("\n--- Starting orchestrator run ---")
        result = await Runner.run(
            orchestrator_agent,
            orchestrator_input_text,
            context=tutor_context,
            run_config=run_config
        )

        print(f"\n--- Orchestrator run completed ---")
        print(f"Result type: {type(result.final_output)}")
        print(f"Raw result: {result.final_output}")

        # After the run, save the *modified* context back to the session manager
        print("\n--- Updating session context ---")
        updated_context_data = tutor_context.model_dump(mode='json')
        success = session_manager.update_session(session_id, updated_context_data)
        
        if not success:
            error_msg = f"Failed to save updated context after interaction for session {session_id}"
            print(f"ERROR: {error_msg}")
            logger.log_error("SessionUpdate", error_msg)
            # Decide if this should be a critical error
        else:
            print("Successfully saved updated context") # Log after save
        
        # --- Wrap the Orchestrator's output ---
        # Log orchestrator output
        orchestrator_output = result.final_output
        logger.log_orchestrator_output(orchestrator_output)

        # Determine content_type and data based on orchestrator_output
        # This assumes orchestrator_output is one of the specific response types (ExplanationResponse etc.)
        content_type = getattr(orchestrator_output, 'response_type', 'message') # Default to message if type unknown

        # Return the wrapped response expected by the frontend
        return InteractionResponseData(
            content_type=content_type,
            data=orchestrator_output, # Send the whole object as data
            user_model_state=tutor_context.user_model_state # Send the *updated* user model state
        )

    except Exception as e:
        # --- Explicit Exception Catching and Logging ---
        logger.log_error("OrchestratorRun", e)
        print("\n!!! EXCEPTION IN /interact !!!")
        print("Full traceback:")
        print(traceback.format_exc())
        
        error_msg = f"Error during interaction processing: {str(e)}"
        print(f"\nError details: {error_msg}")
        
        # Include error type in the response for better debugging
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error during interaction",
                "type": type(e).__name__,
                "message": str(e)
            }
        )

# --- Remove POST /quiz/submit ---
# Quiz answers are now handled via the /interact endpoint.
# An end-of-session quiz submission could potentially be added back later
# if needed, but the core loop uses /interact.

# TODO: Implement endpoint for session analysis if needed:
# POST /sessions/{session_id}/analyze-session (Full Session Analysis)

# Response model for /interact - can be complex, start simple
class TutorInteractionResponse(BaseModel):
    response_type: Literal["explanation", "question", "feedback", "error", "message"]
    topic: Optional[str] = None 