from __future__ import annotations
from google.adk.tools import FunctionTool, ToolContext
from google.adk.runners import Runner, RunConfig
from typing import Any, Optional, Literal, Union, cast, Dict, List
import os
from datetime import datetime
import traceback # Import traceback for error logging
import logging # Add logging import

# --- Import TutorContext directly ---
# We need the actual class definition available for get_type_hints called by the decorator.
# This relies on context.py being fully loaded *before* this file attempts to define the tools.
from ai_tutor.context import TutorContext, UserConceptMastery, UserModelState

# --- Import necessary models ---
#from ai_tutor.agents.models import LessonPlan, QuizQuestion, QuizFeedbackItem, LessonContent, Quiz, LessonSection, LearningObjective
# Models needed by the tools themselves or for type hints
from ai_tutor.agents.models import FocusObjective, QuizQuestion, QuizFeedbackItem, ExplanationResult, QuizCreationResult, TeacherTurnResult # Import new models
# from ai_tutor.agents.models import LessonPlan, LessonContent, Quiz, LessonSection, LearningObjective # Remove unused models

# Import API response models for potentially formatting tool output
from ai_tutor.api_models import (
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse
)

# --- Get Supabase client dependency (needed for the tool) ---
from ai_tutor.dependencies import get_supabase_client

# Setup module logger
logger = logging.getLogger(__name__)

# --- File Handling Tools ---

@FunctionTool
async def read_knowledge_base(tool_context: ToolContext) -> str:
    """Reads the Knowledge Base content stored in the Supabase 'folders' table."""
    try:
        # --- Access state via ToolContext ---
        # State contains the TutorContext dict
        state_dict = tool_context.state
        tutor_context_data = state_dict # Assuming state IS the TutorContext dict

        folder_id_str = tutor_context_data.get("folder_id")
        user_id_str = tutor_context_data.get("user_id")
        analysis_result_data = tutor_context_data.get("analysis_result")

        if not folder_id_str or not user_id_str:
            logger.error("Tool read_knowledge_base: Missing folder_id or user_id in context state.")
            return "Error: Folder ID or User ID not found in session context."

        folder_id = UUID(folder_id_str)
        user_id = UUID(user_id_str)

        # Check context first
        if analysis_result_data and analysis_result_data.get("analysis_text"):
            logger.info("Tool read_knowledge_base: Found analysis text in context state.")
            return analysis_result_data["analysis_text"]

        # --- Query Supabase if not in context ---
        logger.info(f"Tool read_knowledge_base: Querying Supabase for folder {folder_id}.")
        supabase = await get_supabase_client() # Or get client via other means
        response = supabase.table("folders").select("knowledge_base").eq("id", str(folder_id)).eq("user_id", str(user_id)).maybe_single().execute()

        if response.data and response.data.get("knowledge_base"):
            kb_content = response.data["knowledge_base"]
            logger.info(f"Tool read_knowledge_base: Successfully read KB from Supabase for folder {folder_id}.")
            # **Important**: This tool likely CANNOT update the context directly.
            # The *calling agent* (Planner) needs to receive this content and potentially
            # use another tool (like update_user_model) or rely on the framework
            # to store relevant parts back into the state if needed later.
            return kb_content
        else:
            logger.warning(f"Tool read_knowledge_base: KB not found for folder {folder_id}.")
            return f"Error: Knowledge Base not found for folder {folder_id}."
    except Exception as e:
        logger.exception(f"Tool read_knowledge_base: Error accessing context or Supabase: {e}")
        return f"Error reading Knowledge Base: {e}"

@FunctionTool
async def get_document_content(tool_context: ToolContext, file_path_in_storage: str) -> str:
    """
    Retrieves the text content of a document stored in Supabase Storage.
    The 'file_path_in_storage' should be the full path used when uploading
    (e.g., 'user_uuid/folder_uuid/filename.pdf').
    """
    logger.info(f"Tool get_document_content: Attempting to fetch '{file_path_in_storage}'")
    try:
        # Get Supabase client - This is tricky. Ideally, the client is passed
        # via context or a singleton/dependency injection pattern compatible with ADK.
        # Using the dependency function directly might work if called within FastAPI scope,
        # but not guaranteed within agent execution.
        # For now, assume get_supabase_client() works or adapt as needed.
        supabase = await get_supabase_client() # Dependency might need context
        bucket_name = "document_uploads" # TODO: Make configurable

        response = supabase.storage.from_(bucket_name).download(file_path_in_storage)

        # Check for errors (depends on Supabase client library specifics)
        # Assuming response is bytes content on success, or raises error.
        # Need robust error handling based on actual Supabase client behavior.

        # Basic Text Extraction (Improve based on file type if needed)
        # This example assumes simple text or attempts decoding.
        # For complex types like PDF/DOCX, you'd need libraries like pypdf, python-docx.
        try:
            content = response.decode('utf-8')
            logger.info(f"Tool get_document_content: Successfully fetched and decoded text for '{file_path_in_storage}'")
            return content
        except UnicodeDecodeError:
            logger.warning(f"Tool get_document_content: Could not decode '{file_path_in_storage}' as UTF-8. Returning raw representation.")
            return f"[Binary Content: {file_path_in_storage}]" # Placeholder for non-text
        except Exception as decode_err:
            logger.error(f"Tool get_document_content: Error decoding content for '{file_path_in_storage}': {decode_err}")
            return f"[Error decoding content for {file_path_in_storage}]"

    except Exception as e:
        # Log Supabase client errors (check exception type for specifics)
        logger.exception(f"Tool get_document_content: Failed to download '{file_path_in_storage}' from Supabase: {e}")
        return f"[Error retrieving document content for {file_path_in_storage}]"

# --- Orchestrator Tool Implementations ---

@FunctionTool
async def call_quiz_creator_mini(
    tool_context: google.adk.tools.ToolContext,
    topic: str
) -> Union[QuizQuestion, str]: # Return Question object or error string
    """DEPRECATED: Use call_quiz_creator_agent instead. Generates a single multiple-choice question."""
    return "Error: This tool is deprecated. Use call_quiz_creator_agent to invoke the quiz creator."

@FunctionTool
async def call_quiz_teacher_evaluate(ctx: google.adk.runners.RunContextWrapper[TutorContext], user_answer_index: int) -> Union[QuizFeedbackItem, str]:
    """Evaluates the user's answer to the current question using the Quiz Teacher logic (via helper function)."""
    print(f"[Tool call_quiz_teacher_evaluate] Evaluating user answer index '{user_answer_index}'.")
    
    try:
        # --- Import evaluation function *Inside* ---
        from ai_tutor.agents.quiz_teacher_agent import evaluate_single_answer # Import helper here
        # -------------------------------------------
        question_to_evaluate = ctx.context.current_quiz_question
        if not question_to_evaluate:
            return "Error: No current question found in context to evaluate."
        if not isinstance(question_to_evaluate, QuizQuestion):
            # Add type check for safety
            return f"Error: Expected QuizQuestion in context, found {type(question_to_evaluate).__name__}."

        print(f"[Tool call_quiz_teacher_evaluate] Evaluating answer for question: {question_to_evaluate.question[:50]}...")
        feedback_item = await evaluate_single_answer(
            question=question_to_evaluate,
            user_answer_index=user_answer_index,
            context=ctx.context # Pass context
        )

        if feedback_item:
            print(f"[Tool call_quiz_teacher_evaluate] Evaluation complete. Feedback: Correct={feedback_item.is_correct}, Explanation: {feedback_item.explanation[:50]}...")
            # Clear pending interaction *after* successful evaluation & getting feedback
            ctx.context.user_model_state.pending_interaction_type = None
            ctx.context.user_model_state.pending_interaction_details = None
            # Clear the question itself from context now that it's evaluated
            ctx.context.current_quiz_question = None
            return feedback_item
        else:
            # The evaluate_single_answer helper should ideally return feedback or raise
            error_msg = f"Error: Evaluation failed for question on topic '{getattr(question_to_evaluate, 'related_section', 'N/A')}'."
            print(f"[Tool call_quiz_teacher_evaluate] {error_msg}")
            return error_msg # Return error string
            
    except Exception as e:
        error_msg = f"Exception in call_quiz_teacher_evaluate: {str(e)}"
        print(f"[Tool] {error_msg}")
        # Optionally clear pending state even on error? Depends on desired flow.
        # ctx.context.user_model_state.pending_interaction_type = None 
        return error_msg # Return error string

@FunctionTool
def determine_next_learning_step(ctx: google.adk.runners.RunContextWrapper[TutorContext]) -> Dict[str, Any]:
    """DEPRECATED: The Planner agent now determines the next focus. Use call_planner_agent."""
    return {"error": "This tool is deprecated. Use call_planner_agent to get the next focus."}

@FunctionTool
async def update_user_model(
    ctx: google.adk.runners.RunContextWrapper[TutorContext],
    topic: str,
    outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained'],
    confusion_point: Optional[str] = None,
    last_accessed: Optional[str] = None,
    mastered_objective_title: Optional[str] = None, # Optional: Mark an objective as mastered
) -> str:
    """Updates the user model state with interaction outcomes and temporal data."""
    print(f"[Tool update_user_model] Updating '{topic}' with outcome '{outcome}'")

    # Ensure context and user model state exist
    if not ctx.context or not ctx.context.user_model_state:
        return "Error: TutorContext or UserModelState not found."

    if not topic or not isinstance(topic, str):
        return "Error: Invalid topic provided for user model update."

    # Initialize concept if needed
    if topic not in ctx.context.user_model_state.concepts:
        ctx.context.user_model_state.concepts[topic] = UserConceptMastery()

    concept_state = ctx.context.user_model_state.concepts[topic]
    concept_state.last_interaction_outcome = outcome

    # Update last_accessed with ISO 8601 timestamp
    concept_state.last_accessed = last_accessed or datetime.now().isoformat()

    # Add confusion point if provided
    if confusion_point and confusion_point not in concept_state.confusion_points:
        concept_state.confusion_points.append(confusion_point)

    # Update attempts and mastery for evaluative outcomes
    if outcome in ['correct', 'incorrect', 'mastered', 'struggled']:
        concept_state.attempts += 1

        # Adjust mastery level based on outcome
        if outcome in ['correct', 'mastered']:
            concept_state.mastery_level = min(1.0, concept_state.mastery_level + 0.2)
        elif outcome in ['incorrect', 'struggled']:
            concept_state.mastery_level = max(0.0, concept_state.mastery_level - 0.1)
            
            # Adjust learning pace if struggling
            if len(concept_state.confusion_points) > 2:
                ctx.context.user_model_state.learning_pace_factor = max(0.5, 
                    ctx.context.user_model_state.learning_pace_factor - 0.1)

    # Update mastered objectives if provided
    if mastered_objective_title and mastered_objective_title not in ctx.context.user_model_state.mastered_objectives_current_section:
         ctx.context.user_model_state.mastered_objectives_current_section.append(mastered_objective_title)
         print(f"[Tool] Marked objective '{mastered_objective_title}' as mastered for current section.")

    print(f"[Tool] Updated '{topic}' - Mastery: {concept_state.mastery_level:.2f}, "
          f"Pace: {ctx.context.user_model_state.learning_pace_factor:.2f}")
    return f"User model updated for {topic}."

@FunctionTool
def update_explanation_progress(ctx: google.adk.runners.RunContextWrapper[TutorContext], segment_index: int) -> str:
    """DEPRECATED: The Orchestrator manages micro-steps directly."""
    return "Error: This tool is deprecated. Orchestrator manages micro-steps."

@FunctionTool
async def get_user_model_status(ctx: google.adk.runners.RunContextWrapper[TutorContext], topic: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves detailed user model state, optionally for a specific topic."""
    print(f"[Tool get_user_model_status] Retrieving status for topic '{topic}'")

    if not ctx.context.user_model_state:
        return {"error": "No user model state found in context."}

    state = ctx.context.user_model_state

    if topic:
        if topic not in state.concepts:
            return {
                "topic": topic,
                "exists": False,
                "message": "Topic not found in user model."
            }
            
        concept = state.concepts[topic]
        return {
            "topic": topic,
            "exists": True,
            "mastery_level": concept.mastery_level,
            "attempts": concept.attempts,
            "last_outcome": concept.last_interaction_outcome,
            "confusion_points": concept.confusion_points,
            "last_accessed": concept.last_accessed
        }
    
    # Return full state summary if no specific topic requested
    # Use model_dump for serializable output
    return state.model_dump(mode='json')

@FunctionTool
async def reflect_on_interaction(
    ctx: google.adk.runners.RunContextWrapper[TutorContext],
    topic: str,
    interaction_summary: str, # e.g., "User answered checking question incorrectly."
    user_response: Optional[str] = None, # The actual user answer/input
    feedback_provided: Optional[QuizFeedbackItem] = None # Feedback from teacher tool if available
) -> Dict[str, Any]:
    """
    Analyzes the last interaction for a given topic, identifies potential reasons for user difficulty,
    and suggests adaptive next steps for the Orchestrator.
    """
    print(f"[Tool reflect_on_interaction] Called for topic '{topic}'. Summary: {interaction_summary}")

    # Basic reflection logic (can be enhanced, e.g., calling another LLM for deeper analysis)
    suggestions = []
    analysis = f"Reflection on interaction regarding '{topic}': {interaction_summary}. "

    if feedback_provided and not feedback_provided.is_correct:
        analysis += f"User incorrectly selected '{feedback_provided.user_selected_option}' when the correct answer was '{feedback_provided.correct_option}'. "
        analysis += f"Explanation: {feedback_provided.explanation}. "
        suggestions.append(f"Re-explain the core concept using the provided explanation: '{feedback_provided.explanation}'.")
        if feedback_provided.improvement_suggestion:
            suggestions.append(f"Focus on the improvement suggestion: '{feedback_provided.improvement_suggestion}'.")
        suggestions.append(f"Try asking a slightly different checking question on the same concept.")
    elif "incorrect" in interaction_summary.lower() or "struggled" in interaction_summary.lower():
        suggestions.append(f"Consider re-explaining the last segment of '{topic}' using a different approach or analogy.")
        suggestions.append(f"Ask a simpler checking question focused on the specific confusion points for '{topic}'.")
    else: # If interaction was positive or just an explanation
        analysis += "Interaction seems positive or neutral."
        suggestions.append("Proceed with the next logical step in the micro-plan (e.g., next segment, checking question).")

    print(f"[Tool reflect_on_interaction] Analysis: {analysis}. Suggestions: {suggestions}")
    return {"analysis": analysis, "suggested_next_steps": suggestions}

# --- NEW Tools to Call Other Agents ---

@FunctionTool
async def call_planner_agent(
    ctx: google.adk.runners.RunContextWrapper[TutorContext],
    user_state_summary: Optional[str] = None # Optional summary of user state
) -> Union[FocusObjective, str]:
    """Calls the Planner Agent to determine the next learning focus objective. Returns FocusObjective on success, or an error string on failure."""
    print("[Tool call_planner_agent] Calling Planner Agent...")
    try:
        # --- Import and Create Agent *Inside* ---
        from ai_tutor.agents.planner_agent import create_planner_agent # Import here
        # ----------------------------------------
        if not ctx.context.vector_store_id:
            return "Error: Vector store ID not found in context for Planner."

        planner_agent = create_planner_agent(ctx.context.vector_store_id)
        run_config = RunConfig(workflow_name="Orchestrator_PlannerCall", group_id=ctx.context.session_id)

        # Construct prompt for the planner
        planner_prompt = f"""
        Determine the next learning focus for the user.
        First, call `read_knowledge_base` to understand the material's structure and concepts.
        Analyze the knowledge base.
        {f'Consider the user state: {user_state_summary}' if user_state_summary else 'Assume the user is starting or has just completed the previous focus.'}
        Identify the single most important topic or concept for the user to focus on next.
        Output your decision ONLY as a FocusObjective object.
        """

        result = await Runner.run(
            planner_agent,
            planner_prompt,
            context=ctx.context,
            run_config=run_config
        )

        # Check the type of the final output BEFORE trying to access attributes
        if isinstance(result.final_output, FocusObjective):
            focus_objective = result.final_output
            print(f"[Tool call_planner_agent] Planner returned focus: {focus_objective.topic}")
            # Store the new focus in context
            ctx.context.current_focus_objective = focus_objective
            return focus_objective
        else:
            error_msg = f"Error: Planner agent did not return a valid FocusObjective object. Got type: {type(result.final_output).__name__}. Raw output: {result.final_output}"
            print(f"[Tool call_planner_agent] {error_msg}")
            # Return a descriptive error string that the orchestrator might understand
            return "PLANNER_OUTPUT_ERROR: Planner failed to generate valid focus objective."

    except Exception as e:
        # Catch any exception during the agent run or creation
        error_msg = f"EXCEPTION calling Planner Agent: {str(e)}\n{traceback.format_exc()}"
        print(f"[Tool] {error_msg}")
        # Return a different error string for exceptions vs. wrong output type
        return "PLANNER_EXECUTION_ERROR: An exception occurred while running the planner."

@FunctionTool
async def call_teacher_agent(
    ctx: google.adk.runners.RunContextWrapper[TutorContext],
    topic: str,
    explanation_details: str # e.g., "Explain the concept generally", "Provide an example"
) -> Union[ExplanationResult, str]:
    """Calls the Teacher Agent to provide an explanation for a specific topic/detail."""
    print(f"[Tool call_teacher_agent] Requesting explanation for '{topic}': {explanation_details}")
    try:
        # --- Import and Create Agent *Inside* ---
        from ai_tutor.agents.teacher_agent import create_interactive_teacher_agent # Naming needs update based on actual refactor
        # ----------------------------------------
        if not ctx.context.vector_store_id:
            return "Error: Vector store ID not found in context for Teacher."

        teacher_agent = create_interactive_teacher_agent(ctx.context.vector_store_id) # TODO: Update function name if changed
        run_config = RunConfig(workflow_name="Orchestrator_TeacherCall", group_id=ctx.context.session_id)

        teacher_prompt = f"""
        Explain the topic: '{topic}'.
        Specific instructions for this explanation: {explanation_details}.
        Use the file_search tool if needed to find specific information or examples from the documents.
        Format your response ONLY as an ExplanationResult object containing the explanation text in the 'details' field.
        """

        result = await Runner.run(
            teacher_agent,
            teacher_prompt,
            context=ctx.context,
            run_config=run_config
        )

        # Log the raw output for debugging
        print(f"[Tool call_teacher_agent] Teacher Agent Raw Output Type: {type(result.final_output)}")
        print(f"[Tool call_teacher_agent] Teacher Agent Raw Output Content: {result.final_output}")

        # Check the type BEFORE trying to access attributes or using final_output_as
        if isinstance(result.final_output, ExplanationResult):
            explanation_result = result.final_output
            if explanation_result.status == "delivered":
                print(f"[Tool call_teacher_agent] Teacher delivered structured explanation for '{topic}'.")
                ctx.context.last_interaction_summary = f"Teacher explained {topic}." # Update summary
                return explanation_result # Return the structured result
            else:
                # Handle structured failure/skipped cases
                error_msg = f"TEACHER_RESULT_ERROR: Teacher agent returned status '{explanation_result.status}'. Details: {explanation_result.details}"
                print(f"[Tool call_teacher_agent] {error_msg}")
                return error_msg # Return error string
        elif isinstance(result.final_output, str):
            # If it's a string, wrap it in a successful ExplanationResult for consistency.
            print(f"[Tool call_teacher_agent] Teacher delivered explanation for '{topic}'.")

            wrapped_result = ExplanationResult(status="delivered", details=result.final_output)
            ctx.context.last_interaction_summary = f"Teacher explained {topic} (raw string)."
            return wrapped_result # Return the wrapped result
        else:
            # Handle other unexpected output types
            error_msg = f"TEACHER_OUTPUT_ERROR: Teacher agent returned unexpected output type: {type(result.final_output).__name__}. Raw output: {result.final_output}"
            print(f"[Tool call_teacher_agent] {error_msg}")
            return error_msg # Return error string

    except Exception as e:
        error_msg = f"Error calling Teacher Agent: {str(e)}\n{traceback.format_exc()}"
        print(f"[Tool] {error_msg}")
        return error_msg

@FunctionTool
async def call_quiz_creator_agent(
    ctx: google.adk.runners.RunContextWrapper[TutorContext],
    topic: str,
    instructions: str # e.g., "Create one medium difficulty question", "Create a 3-question quiz covering key concepts"
) -> Union[QuizCreationResult, str]:
    """Calls the Quiz Creator Agent to generate one or more quiz questions."""
    print(f"[Tool call_quiz_creator_agent] Requesting quiz creation for '{topic}': {instructions}")
    try:
        # --- Import and Create Agent *Inside* ---
        from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent # Or the new tool function name
        # ----------------------------------------

        # Assuming create_quiz_creator_agent is the function that returns the agent instance
        quiz_creator_agent = create_quiz_creator_agent() # TODO: Update function name if changed. Pass API key if needed.
        run_config = RunConfig(workflow_name="Orchestrator_QuizCreatorCall", group_id=ctx.context.session_id)

        quiz_creator_prompt = f"""
        Create quiz questions based on the following instructions:
        Topic: '{topic}'
        Instructions: {instructions}
        Format your response ONLY as a QuizCreationResult object. Include the created question(s) in the appropriate field ('question' or 'quiz').
        """

        result = await Runner.run(
            quiz_creator_agent,
            quiz_creator_prompt,
            context=ctx.context,
            run_config=run_config
        )

        quiz_creation_result = result.final_output_as(QuizCreationResult)
        if quiz_creation_result and quiz_creation_result.status == "created":
            question_count = 1 if quiz_creation_result.question else len(quiz_creation_result.quiz.questions) if quiz_creation_result.quiz else 0
            print(f"[Tool call_quiz_creator_agent] Quiz Creator created {question_count} question(s) for '{topic}'.")
            # Store the created question if it's a single one for evaluation
            if quiz_creation_result.question:
                 ctx.context.current_quiz_question = quiz_creation_result.question
            return quiz_creation_result
        else:
            details = getattr(quiz_creation_result, 'details', 'No details provided.')
            return f"Error: Quiz Creator agent failed for '{topic}'. Status: {getattr(quiz_creation_result, 'status', 'unknown')}. Details: {details}"

    except Exception as e:
        error_msg = f"Error calling Quiz Creator Agent: {str(e)}\n{traceback.format_exc()}"
        print(f"[Tool] {error_msg}")
        return error_msg

# Ensure all tools intended for the orchestrator are exported or available
__all__ = [
    # --- NEW ---
    'call_planner_agent',
    'call_teacher_agent',
    'call_quiz_creator_agent',
    # --- Kept ---
    'call_quiz_teacher_evaluate',
    'reflect_on_interaction',
    'update_user_model',
    'get_user_model_status',
    # --- Removed ---
    # 'call_quiz_creator_mini',
    # 'determine_next_learning_step',
    # 'update_explanation_progress',
] 