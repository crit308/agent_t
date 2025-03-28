from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form, Body
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
from ai_tutor.manager import AITutorManager
from pydantic import BaseModel

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
        if os.path.exists(default_kb_path): os.remove(default_kb_path)
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
        response_output = None
        lesson_plan_to_store = None
        lesson_content_to_store = None
        quiz_to_store = None

        # Use a temporary manager instance for helper methods
        manager_instance = AITutorManager(output_logger=logger)
        manager_instance.context = tutor_context

        if isinstance(final_output, LessonPlan):
            lesson_plan_to_store = final_output
            logger.log_planner_output(final_output)
            # Lesson content will be generated/fetched separately
            response_output = final_output

        elif isinstance(final_output, LessonContent):
            # Handoff Plan -> Content occurred
            lesson_content_to_store = final_output
            logger.log_teacher_output(final_output)
            # Synthesize plan for storage
            try:
                # Create a basic lesson plan from content
                sections = [LessonSection(
                    title=content.title if hasattr(content, 'title') else "Generated Section",
                    content=content.content if hasattr(content, 'content') else "",
                    estimated_duration_minutes=30  # Default duration
                ) for content in final_output.sections]
                
                lesson_plan_to_store = LessonPlan(
                    title=final_output.title,
                    description=final_output.description if hasattr(final_output, 'description') else "Generated from content",
                    target_audience=final_output.target_audience if hasattr(final_output, 'target_audience') else "General",
                    prerequisites=final_output.prerequisites if hasattr(final_output, 'prerequisites') else [],
                    sections=sections,
                    total_estimated_duration_minutes=sum(section.estimated_duration_minutes for section in sections)
                )
                logger.log_planner_output(lesson_plan_to_store)
            except Exception as e:
                logger.log_error("SynthesizePlan", f"Could not synthesize plan from content: {e}")
            response_output = final_output

        elif isinstance(final_output, Quiz):
            # Handoff Plan -> Content -> Quiz occurred
            quiz_to_store = final_output
            logger.log_quiz_creator_output(final_output)
            # Synthesize content AND plan for storage
            try:
                lesson_content_to_store = manager_instance._create_lesson_content_from_quiz(final_output)
                logger.log_teacher_output(lesson_content_to_store)
                # Create lesson plan from the synthesized content
                sections = [LessonSection(
                    title=f"Quiz Section {i+1}",
                    content=question.question,
                    estimated_duration_minutes=15  # Default duration per question
                ) for i, question in enumerate(final_output.questions)]
                
                lesson_plan_to_store = LessonPlan(
                    title=final_output.title if hasattr(final_output, 'title') else "Quiz-based Lesson",
                    description="Lesson plan synthesized from quiz content",
                    target_audience=final_output.target_audience if hasattr(final_output, 'target_audience') else "General",
                    prerequisites=[],
                    sections=sections,
                    total_estimated_duration_minutes=sum(section.estimated_duration_minutes for section in sections)
                )
                logger.log_planner_output(lesson_plan_to_store)
            except Exception as e:
                logger.log_error("SynthesizePlanContent", f"Could not synthesize plan/content from quiz: {e}")
            response_output = final_output

        else:
            # Handle error or fallback
            lesson_plan_to_store = LessonPlan(
                title="Fallback Lesson Plan",
                description="Generated as fallback due to unexpected output",
                target_audience="General",
                prerequisites=[],
                sections=[],
                total_estimated_duration_minutes=0
            )
            logger.log_planner_output(lesson_plan_to_store)
            response_output = lesson_plan_to_store

        # --- Store generated artifacts in session state ---
        update_data = {}
        if lesson_plan_to_store:
            update_data["lesson_plan"] = lesson_plan_to_store.model_dump()
            print(f"Adding lesson_plan to update_data with title: {lesson_plan_to_store.title}")
            
        if lesson_content_to_store:
            update_data["lesson_content"] = lesson_content_to_store.model_dump()
            print(f"Adding lesson_content to update_data with title: {lesson_content_to_store.title}")
            
        if quiz_to_store:
            quiz_data = quiz_to_store.model_dump()
            update_data["quiz"] = quiz_data
            print(f"Adding quiz to update_data with title: {quiz_to_store.title}")
            print(f"Quiz data to be stored: {quiz_data.get('title')}, {len(quiz_data.get('questions', []))} questions")

        if update_data:
            try:
                print(f"\nAttempting to store the following in session {session_id}:")
                for key, value in update_data.items():
                    print(f"  -> {key}: {value.get('title', 'No title')} ({type(value)})")
                
                success = session_manager.update_session(session_id, update_data)
                if success:
                    print(f"\nSuccessfully stored in session {session_id}:")
                    print(f"  -> Stored items: {list(update_data.keys())}")
                    if "quiz" in update_data:
                        print(f"  -> Quiz title stored: {update_data['quiz'].get('title')}")
                        print(f"  -> Quiz questions stored: {len(update_data['quiz'].get('questions', []))}")
                else:
                    logger.log_error("SessionUpdate", f"Failed to update session {session_id}")
                    print(f"Failed to update session {session_id} - update_session returned False")
            except Exception as e:
                logger.log_error("SessionUpdateError", f"Error updating session {session_id}: {e}")
                print(f"Exception while updating session {session_id}: {str(e)}")

        return response_output

    except Exception as e:
        logger.log_error("PlannerAgentRun", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson plan: {e}")
    finally:
        # Clean up the temporary default KB file
        if os.path.exists(default_kb_path):
            os.remove(default_kb_path)

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

@router.post(
    "/sessions/{session_id}/quiz/submit",
    response_model=QuizFeedback,
    summary="Submit Quiz Answers for Evaluation",
    tags=["Tutoring Workflow"]
)
async def submit_session_quiz_answers(
    session_id: str,
    answers: QuizUserAnswers = Body(...),
    session: dict = Depends(get_session_state)
):
    logger = get_session_logger(session_id)
    print(f"Received quiz submission for session {session_id}")

    # 1. Retrieve and Parse Original Quiz
    original_quiz: Optional[Quiz] = None
    try:
        quiz_data = session.get("quiz")
        if not quiz_data:
            logger.log_error("SubmitQuiz", f"Original quiz not found in session state for {session_id}")
            raise HTTPException(status_code=404, detail="Quiz not found for this session. Cannot submit answers.")
        print("Found original quiz data in session state.")
        original_quiz = Quiz(**quiz_data)
        print(f"Successfully parsed original quiz: '{original_quiz.title}'")
    except HTTPException:
        raise
    except Exception as e:
        logger.log_error("SubmitQuizParseQuiz", f"Failed to parse stored quiz data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load original quiz for evaluation: {e}")

    # 2. Prepare context
    tutor_context = TutorContext(
        session_id=session_id,
        vector_store_id=session.get("vector_store_id")
    )
    print("Prepared tutor context.")

    # 3. Call generate_quiz_feedback
    quiz_feedback: Optional[QuizFeedback] = None
    try:
        print("Calling generate_quiz_feedback...")
        quiz_feedback = await generate_quiz_feedback(
            quiz=original_quiz,
            user_answers=answers,
            context=tutor_context
        )
        if not quiz_feedback:
            logger.log_error("SubmitQuizFeedbackGen", "generate_quiz_feedback returned None")
            raise HTTPException(status_code=500, detail="Quiz feedback generation failed unexpectedly.")
        print(f"Quiz feedback generated successfully (Score: {quiz_feedback.correct_answers}/{quiz_feedback.total_questions}).")

    except Exception as e:
        logger.log_error("SubmitQuizFeedbackGen", f"Exception during generate_quiz_feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to evaluate quiz answers: {e}")

    # 4. Store Feedback
    try:
        print("Storing quiz feedback in session state...")
        success = session_manager.update_session(session_id, {"quiz_feedback": quiz_feedback.model_dump()})
        if not success:
            print(f"WARNING: Failed to update session state with quiz feedback for {session_id}, but proceeding.")
        else:
            print("Quiz feedback stored successfully.")
        logger.log_quiz_teacher_output(quiz_feedback)

    except Exception as e:
        logger.log_error("SubmitQuizStoreFeedback", f"Exception during storing quiz feedback: {e}", exc_info=True)
        # Don't fail here if feedback was already generated, just log the storage error

    # 5. Return Feedback
    print("Returning quiz feedback to client.")
    return quiz_feedback

# TODO: Implement endpoints for:
# POST /sessions/{session_id}/content (Lesson Content Generation)
# GET /sessions/{session_id}/lesson (Retrieve Lesson Content)
# GET /sessions/{session_id}/quiz (Retrieve Quiz)
# POST /sessions/{session_id}/quiz/submit (Submit Quiz Answers)
# POST /sessions/{session_id}/analyze-session (Full Session Analysis)
# GET /sessions/{session_id}/status/{task_id} (If using background tasks) 