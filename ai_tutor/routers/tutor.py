from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form
from typing import List, Optional, Union
import os
import shutil
import time

from ai_tutor.session_manager import SessionManager
from ai_tutor.tools.file_upload import FileUploadManager
from ai_tutor.agents import (
    analyze_documents,
    create_planner_agent,
    create_teacher_agent,
    create_quiz_creator_agent,
    create_quiz_teacher_agent,
    generate_quiz_feedback,
    analyze_teaching_session,
    process_handoff_data
)
from ai_tutor.agents.analyzer_agent import AnalysisResult
from ai_tutor.agents.models import (
    LessonPlan, LessonContent, Quiz, QuizUserAnswers, QuizFeedback, SessionAnalysis
)
from ai_tutor.api_models import DocumentUploadResponse, AnalysisResponse
from ai_tutor.context import TutorContext
from ai_tutor.output_logger import get_logger, TutorOutputLogger
from agents import Runner, RunConfig

router = APIRouter()
session_manager = SessionManager()

# Directory for temporary file uploads
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- Dependency to get session state ---
async def get_session_state(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

# --- Helper to get logger ---
def get_session_logger(session_id: str) -> TutorOutputLogger:
    # Customize logger per session if needed, e.g., different file path
    log_file = os.path.join("logs", f"session_{session_id}.log") # Example path
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    return get_logger(output_file=log_file)


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
    session: dict = Depends(get_session_state)
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
                "uploaded_files": session.get("uploaded_files", []) + uploaded_filenames
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
        # Provide context for tracing
        tutor_context = TutorContext(session_id=session_id, vector_store_id=vector_store_id)
        analysis_result: Optional[AnalysisResult] = await analyze_documents(vector_store_id, context=tutor_context)

        if analysis_result:
            analysis_status = "completed"
            # Store analysis result (as dict or object)
            analysis_data = analysis_result.model_dump() if analysis_result else None
            session_manager.update_session(session_id, {"document_analysis": analysis_data})
            message += "Analysis completed."

            # --- Crucial: Create Knowledge Base file for Planner ---
            # The planner agent relies on this file existing locally.
            kb_filename = f"knowledge_base_{session_id}.txt" # Session-specific KB file
            kb_path = os.path.join(TEMP_UPLOAD_DIR, kb_filename)
            try:
                with open(kb_path, "w", encoding="utf-8") as f:
                    f.write("KNOWLEDGE BASE\n=============\n\n")
                    f.write("DOCUMENT ANALYSIS:\n=================\n\n")
                    f.write(analysis_result.analysis_text) # Write the text part
                session_manager.update_session(session_id, {"knowledge_base_path": kb_path})
                print(f"Knowledge Base file created for session {session_id} at {kb_path}")
            except Exception as kb_error:
                 logger.log_error("KnowledgeBaseCreation", kb_error)
                 message += f"Warning: Could not create Knowledge Base file: {kb_error}"
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
async def get_session_analysis_results(session_id: str, session: dict = Depends(get_session_state)):
    """Retrieves the results of the document analysis for the session."""
    analysis_data = session.get("document_analysis")
    if analysis_data:
        try:
            # Re-create the AnalysisResult object from stored data
            analysis_obj = AnalysisResult(**analysis_data)
            return AnalysisResponse(status="completed", analysis=analysis_obj)
        except Exception as e:
             return AnalysisResponse(status="error", error=f"Failed to parse analysis data: {e}")
    else:
        # Check if analysis failed or just hasn't run
        # This logic depends on how status is tracked (not fully implemented here)
        return AnalysisResponse(status="not_started", analysis=None)


@router.post(
    "/sessions/{session_id}/plan",
    response_model=Union[LessonPlan, Quiz], # Handoff might return a Quiz
    summary="Generate Lesson Plan",
    tags=["Tutoring Workflow"]
)
async def generate_session_lesson_plan(session_id: str, session: dict = Depends(get_session_state)):
    """Generates a lesson plan based on the analyzed documents."""
    logger = get_session_logger(session_id)
    vector_store_id = session.get("vector_store_id")
    kb_path = session.get("knowledge_base_path") # Get path to the session's KB file

    if not vector_store_id:
        raise HTTPException(status_code=400, detail="Documents must be uploaded and analysis completed first.")

    # --- Crucial: Ensure Knowledge Base file exists ---
    # Modify the read_knowledge_base tool or planner agent temporarily to read from the session-specific path
    # OR copy the session KB to the expected "Knowledge Base" path before running
    default_kb_path = "Knowledge Base"
    if not kb_path or not os.path.exists(kb_path):
         raise HTTPException(status_code=400, detail="Knowledge Base file not found for this session. Ensure analysis completed.")
    try:
        # Temporarily link or copy the session KB to the default path
        if os.path.exists(default_kb_path): os.remove(default_kb_path) # Avoid conflicts if running multiple locally
        shutil.copyfile(kb_path, default_kb_path)
    except Exception as e:
        logger.log_error("KnowledgeBaseLink", e)
        raise HTTPException(status_code=500, detail=f"Failed to prepare Knowledge Base file: {e}")
    # --- End KB Check ---

    try:
        planner_agent = create_planner_agent(vector_store_id)
        tutor_context = TutorContext(session_id=session_id, vector_store_id=vector_store_id)
        run_config = RunConfig(workflow_name="AI Tutor API - Planning", group_id=session_id)

        # Run synchronously for now
        result = await Runner.run(planner_agent, "Create a lesson plan.", run_config=run_config, context=tutor_context)

        final_output = result.final_output
        output_to_store = None
        response_output = None

        if isinstance(final_output, LessonPlan):
            session_manager.update_session(session_id, {"lesson_plan": final_output.model_dump()})
            logger.log_planner_output(final_output)
            response_output = final_output
        elif isinstance(final_output, Quiz):
            # Handoff occurred, store the quiz
            session_manager.update_session(session_id, {"quiz": final_output.model_dump()})
            logger.log_quiz_creator_output(final_output) # Log as quiz creator output
            # Need to synthesize a LessonPlan for the response type if API expects only LessonPlan here
            # For now, let's return the Quiz directly since the response model allows Union
            response_output = final_output
        elif isinstance(final_output, LessonContent):
             # Handoff might have gone further
             session_manager.update_session(session_id, {"lesson_content": final_output.model_dump()})
             logger.log_teacher_output(final_output)
             # Need to synthesize LessonPlan
             response_output = LessonPlan(title="Synthesized Plan", description="Generated via handoff", target_audience="N/A", prerequisites=[], sections=[], total_estimated_duration_minutes=0) # Placeholder
        else:
            raise HTTPException(status_code=500, detail=f"Planner agent returned unexpected output type: {type(final_output)}")

        return response_output

    except Exception as e:
        logger.log_error("PlannerAgentRun", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson plan: {e}")
    finally:
         # Clean up the temporary default KB file
         if os.path.exists(default_kb_path):
             os.remove(default_kb_path)

# TODO: Implement endpoints for:
# POST /sessions/{session_id}/content (Lesson Content Generation)
# GET /sessions/{session_id}/lesson (Retrieve Lesson Content)
# GET /sessions/{session_id}/quiz (Retrieve Quiz)
# POST /sessions/{session_id}/quiz/submit (Submit Quiz Answers)
# POST /sessions/{session_id}/analyze-session (Full Session Analysis)
# GET /sessions/{session_id}/status/{task_id} (If using background tasks) 