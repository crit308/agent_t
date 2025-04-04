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
from ai_tutor.agents.analyzer_agent import analyze_documents
from ai_tutor.agents.planner_agent import create_planner_agent
from ai_tutor.agents.session_analyzer_agent import analyze_teaching_session
from ai_tutor.agents.orchestrator_agent import create_orchestrator_agent
from ai_tutor.agents.teacher_agent import create_interactive_teacher_agent
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

# --- Dependency to get parsed TutorContext (Removed pre-parsing) ---
async def get_tutor_context(session_id: str) -> TutorContext:
    session_data = session_manager.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    try:
        # Pydantic will parse nested models like analysis_result automatically if keys match
        print(f"[Debug] Raw session data for {session_id} before TutorContext init: {json.dumps(session_data, indent=2, default=str)}")
        return TutorContext(**session_data)
    except Exception as e:
        # Log the raw data along with the error for easier debugging
        print(f"[ERROR] Failed to parse session data into TutorContext. Data: {session_data}")
        print(f"[ERROR] Failed to parse session data into TutorContext: {e}") # Log parsing error
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
            # Store analysis result (as dict or object) - Keep storing the object
            analysis_data = analysis_result.model_dump(mode='json')
            session_manager.update_session(session_id, {"analysis_result": analysis_data})
            message += "Analysis completed."

            # --- Reintroduce Knowledge Base file creation ---
            # The planner agent will use a tool to read this.
            kb_filename = f"knowledge_base_{session_id}.txt" # Session-specific KB file
            kb_path = os.path.join(TEMP_UPLOAD_DIR, kb_filename)
            try:
                with open(kb_path, "w", encoding="utf-8") as f:
                    f.write("KNOWLEDGE BASE\n=============\n\n")
                    f.write("DOCUMENT ANALYSIS:\n=================\n\n")
                    f.write(analysis_result.analysis_text) # Write the text part
                session_manager.update_session(session_id, {"knowledge_base_path": kb_path}) # Store the path
                message += f" Knowledge base file created at {kb_path}."
                print(f"Knowledge Base file created for session {session_id} at {kb_path}")
            except Exception as kb_error:
                 logger.log_error("KnowledgeBaseCreation", kb_error)
                 message += f" Warning: Could not create Knowledge Base file: {kb_error}"
            # --- End KB File Creation ---

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
    response_model=LessonPlan,
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
    kb_path = tutor_context.knowledge_base_path # Get KB path from context

    # Initial checks are good to keep, ensure data looks okay before agent creation
    if not vector_store_id:
        raise HTTPException(status_code=400, detail="Documents must be uploaded and analysis completed first.")
    if not kb_path or not os.path.exists(kb_path):
        raise HTTPException(status_code=400, detail="Knowledge base file path not found or file missing.")

    # --- Wrap the main logic in a try...except block ---
    try:
        print(f"[Debug /plan] Creating planner agent for vs_id: {vector_store_id}") # Add log
        planner_agent: Agent[TutorContext] = create_planner_agent(vector_store_id)

        # Pass the full TutorContext to the Runner
        run_config = RunConfig(workflow_name="AI Tutor API - Planning", group_id=session_id)

        print(f"[Debug /plan] Starting Runner.run for planner agent...") # Add log

        # Prompt tells planner to use its tools (read_knowledge_base and file_search)
        plan_prompt = """
        Create a lesson plan. First, use the `read_knowledge_base` tool to understand the document analysis. Then, use the `file_search` tool to clarify details as needed. Finally, generate the `LessonPlan` object based on both sources of information. Follow your detailed agent instructions.
        """
        print(f"[Debug /plan] Planner prompt:\n{plan_prompt[:500]}...") # Log start of prompt

        result = await Runner.run(
            planner_agent,
            plan_prompt,
            run_config=run_config,
            context=tutor_context # Pass the parsed TutorContext object
        )
        print(f"[Debug /plan] Runner.run completed. Result final_output type: {type(result.final_output)}") # Add log

        # --- Check the result and update session ---
        # Use final_output_as for potential validation
        try:
            lesson_plan_obj = result.final_output_as(LessonPlan, raise_if_incorrect_type=True)
            lesson_plan_to_store = lesson_plan_obj
            logger.log_planner_output(lesson_plan_obj)

            update_data = {"lesson_plan": lesson_plan_to_store.model_dump(mode='json')}
            success = session_manager.update_session(session_id, update_data)
            if not success:
                 logger.log_error("SessionUpdate", f"Failed to update session {session_id} with lesson plan.")
            print(f"[Debug /plan] LessonPlan stored in session.") # Add log
            return lesson_plan_obj # Return the generated plan
        except Exception as parse_error: # Catch if final_output isn't a LessonPlan
            error_msg = f"Planner agent returned unexpected output or parsing failed: {parse_error}. Raw output: {result.final_output}"
            logger.log_error("PlannerAgentOutputParse", error_msg)
            print(f"[ERROR /plan] {error_msg}") # Add log
            raise HTTPException(status_code=500, detail="Planner agent failed to return a valid LessonPlan.")

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

        # Raise a generic 500, but the logs now contain the details
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson plan due to internal error: {error_type}")

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
    logger = get_session_logger(session_id)
    print(f"\n=== Starting /interact for session {session_id} ===")
    print(f"[Interact] Input Type: {interaction_input.type}, Data: {interaction_input.data}")
    print(f"[Interact] Context before run: pending_interaction={tutor_context.user_model_state.pending_interaction_type}, current_topic='{tutor_context.current_teaching_topic}', segment={tutor_context.user_model_state.current_topic_segment_index}")

    # --- Agent Execution Logic --- 
    final_response_data: TutorInteractionResponse

    try:
        run_config = RunConfig(workflow_name="AI Tutor API - Interaction", group_id=session_id)

        # Always run the Orchestrator first to decide the next step or handle pending interactions.
        orchestrator_agent = create_orchestrator_agent(tutor_context.vector_store_id)
        
        # Prepare input for the Orchestrator
        if tutor_context.user_model_state.pending_interaction_type:
            # If waiting for user input, provide it clearly
            print("[Interact] Pending interaction detected. Running Orchestrator to evaluate.")
            orchestrator_input = f"User Response to Pending Interaction '{tutor_context.user_model_state.pending_interaction_type}' | Type: {interaction_input.type} | Data: {json.dumps(interaction_input.data)}"
        else:
            # No pending interaction, Orchestrator decides next general step
            print("[Interact] No pending interaction. Running Orchestrator to decide next step.")
            orchestrator_input = f"User Action | Type: {interaction_input.type} | Data: {json.dumps(interaction_input.data)}"

        print(f"[Interact] Running Agent: {orchestrator_agent.name}")
        orchestrator_result = await Runner.run(
            orchestrator_agent,
            orchestrator_input,
            context=tutor_context, # Context is mutable and modified by tools
            run_config=run_config
        )
        orchestrator_output = orchestrator_result.final_output # This is TutorInteractionResponse type
        logger.log_orchestrator_output(orchestrator_output)
        print(f"[Interact] Orchestrator Output Type: {type(orchestrator_output)}, Content: {orchestrator_output}")

        # --- Handle Orchestrator Output ---

        # A) If Orchestrator initiated teaching:
        if isinstance(orchestrator_output, MessageResponse) and orchestrator_output.message_type == 'initiate_teaching':
            print(f"[Interact] Orchestrator signaled to initiate teaching. Running Teacher Agent.")
            if not tutor_context.current_teaching_topic:
                 logger.log_error("TeacherRun", "Orchestrator signaled teaching but current_teaching_topic is not set in context.")
                 raise HTTPException(status_code=500, detail="Internal error: Cannot initiate teaching without a topic set by Orchestrator.")
            
            try:
                teacher_agent = create_interactive_teacher_agent(tutor_context.vector_store_id)
                # Teacher input can be generic; its instructions guide it based on context
                teacher_input = f"Explain segment {tutor_context.user_model_state.current_topic_segment_index} of topic '{tutor_context.current_teaching_topic}'."
                
                print(f"[Interact] Running Teacher Agent: {teacher_agent.name} for topic '{tutor_context.current_teaching_topic}' segment {tutor_context.user_model_state.current_topic_segment_index}")
                teacher_result = await Runner.run(
                    teacher_agent,
                    teacher_input,
                    context=tutor_context, # Pass the *same context object*
                    run_config=run_config 
                )
                # The teacher's output (ExplanationResponse or QuestionResponse) is the final response
                final_response_data = teacher_result.final_output # Type is TeacherInteractionOutput
                logger.log_teacher_output(final_response_data) # Log teacher's specific output
                print(f"[Interact] Teacher Output Type: {type(final_response_data)}, Content: {final_response_data}")

                # Update context based on teacher's action (Runner updates context in-place)
                # Example: Teacher might set pending_interaction_type in the context
                session_manager.update_session(session_id, tutor_context.model_dump(mode='json'))
                print(f"[Interact] Context after Teacher: pending_interaction={tutor_context.user_model_state.pending_interaction_type}, current_topic='{tutor_context.current_teaching_topic}', segment={tutor_context.user_model_state.current_topic_segment_index}")

                # Return the Teacher's output, not the Orchestrator's message
                response_data = final_response_data

            except Exception as teacher_error:
                detailed_error = traceback.format_exc()
                logger.log_error("TeacherAgentError", f"Error running Teacher Agent: {teacher_error}\n{detailed_error}")
                # Return an error response instead of raising HTTP exception immediately
                response_data = ErrorResponse(
                    response_type="error", 
                    message=f"Error during teaching phase: {teacher_error}", 
                    details={"traceback": detailed_error}
                )
        
        # B) Handle other orchestrator outputs (e.g., asking for clarification, quiz)
        elif isinstance(orchestrator_output, (ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse)):
            # If orchestrator returns a direct response, use it
            response_data = orchestrator_output
            # Update context if orchestrator modified it (Runner does this)
            session_manager.update_session(session_id, tutor_context.model_dump(mode='json'))
            print(f"[Interact] Context after Orchestrator (direct response): {tutor_context.model_dump(mode='json')}") # Log context state
        else:
            # Fallback for unexpected orchestrator output
            print(f"[Interact] Unexpected Orchestrator output type: {type(orchestrator_output)}")
            response_data = ErrorResponse(error=f"Unexpected orchestrator output type: {type(orchestrator_output)}")
            session_manager.update_session(session_id, tutor_context.model_dump(mode='json')) # Save context anyway

        # --- Construct and Return Response ---
        print(f"[Interact] Final Response Data Type: {type(response_data)}")
        if isinstance(response_data, ExplanationResponse):
            response_type = "explanation"
        elif isinstance(response_data, QuestionResponse):
            response_type = "question"
        elif isinstance(response_data, FeedbackResponse):
            response_type = "feedback"
        elif isinstance(response_data, MessageResponse):
            response_type = "message"
        elif isinstance(response_data, ErrorResponse):
            response_type = "error"
        else:
            print(f"[Interact] ERROR: Unknown final response data type: {type(response_data)}")
            response_type = "error"
            # Create a default error response if none exists
            if not isinstance(response_data, ErrorResponse):
                 response_data = ErrorResponse(error="Unknown or invalid response type from agent.")

        # Return the structured response
        return InteractionResponseData(
            content_type=response_type,
            data=response_data, # Send the response from the final agent run
            user_model_state=tutor_context.user_model_state # Send updated state
        )

    except HTTPException as http_exc:
        print(f"[Interact] HTTPException: {http_exc.detail}") # Log HTTP exceptions
        raise http_exc # Re-raise HTTP exceptions
    except Exception as e:
        logger.log_error("InteractEndpoint", e)
        print(f"[Interact] Internal Server Error: {e}") # Log general errors
        # Print traceback for detailed debugging
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Remove POST /quiz/submit (Legacy) ---
# Quiz answers are now handled via the /interact endpoint.
# An end-of-session quiz submission could potentially be added back later
# if needed, but the core loop uses /interact.

# TODO: Implement endpoint for session analysis if needed:
# POST /sessions/{session_id}/analyze-session (Full Session Analysis) 