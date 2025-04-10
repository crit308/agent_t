from __future__ import annotations
from google.adk.tools import FunctionTool, ToolContext
from google.adk.runners import Runner, RunConfig
from typing import Any, Optional, Literal, Union, cast, Dict, List
import os
from datetime import datetime
import traceback # Import traceback for error logging
import logging # Add logging import
import json

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
async def call_quiz_teacher_evaluate(tool_context: ToolContext, user_answer_index: int) -> Union[QuizFeedbackItem, str]:
    """Evaluates the user's answer to the current question using the Quiz Teacher logic (via helper function)."""
    print(f"[Tool call_quiz_teacher_evaluate] Evaluating user answer index '{user_answer_index}'.")
    
    try:
        # --- Import evaluation function *Inside* ---
        from ai_tutor.agents.quiz_teacher_agent import evaluate_single_answer # Import helper here
        # -------------------------------------------
        # Access state via tool_context.state dictionary
        state_dict = tool_context.state
        current_quiz_question_data = state_dict.get("current_quiz_question")

        if not current_quiz_question_data:
            return "Error: No current question found in context to evaluate."

        # Validate and parse the question data from the state dictionary
        try:
            question_to_evaluate = QuizQuestion.model_validate(current_quiz_question_data)
        except Exception as parse_err:
            return f"Error: Failed to parse QuizQuestion from context state: {parse_err}"

        print(f"[Tool call_quiz_teacher_evaluate] Evaluating answer for question: {question_to_evaluate.question[:50]}...")
        
        # Re-parse TutorContext to pass to the helper if needed
        # Ideally, evaluate_single_answer would only take necessary data, not the whole context object
        tutor_context_for_helper = TutorContext.model_validate(state_dict) 
        
        feedback_item = await evaluate_single_answer(
            question=question_to_evaluate,
            user_answer_index=user_answer_index,
            context=tutor_context_for_helper # Pass re-parsed context if needed
        )

        if feedback_item:
            print(f"[Tool call_quiz_teacher_evaluate] Evaluation complete. Feedback: Correct={feedback_item.is_correct}, Explanation: {feedback_item.explanation[:50]}...")
            # Modify the state dictionary directly
            user_model_state = state_dict.setdefault("user_model_state", {})
            user_model_state["pending_interaction_type"] = None
            user_model_state["pending_interaction_details"] = None
            state_dict["current_quiz_question"] = None
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
        # tool_context.state.get("user_model_state", {})["pending_interaction_type"] = None 
        return error_msg # Return error string

@FunctionTool
def determine_next_learning_step(tool_context: ToolContext) -> Dict[str, Any]:
    """DEPRECATED: The Planner agent now determines the next focus. Use call_planner_agent."""
    return {"error": "This tool is deprecated. Use call_planner_agent to get the next focus."}

@FunctionTool
async def update_user_model(
    tool_context: ToolContext,
    topic: str,
    outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained'],
    confusion_point: Optional[str] = None,
    last_accessed: Optional[str] = None,
    mastered_objective_title: Optional[str] = None, # Optional: Mark an objective as mastered
) -> str:
    """Updates the user model state (within tool_context.state) with interaction outcomes and temporal data."""
    print(f"[Tool update_user_model] Updating '{topic}' with outcome '{outcome}'")
    state_dict = tool_context.state # Access state dict

    # Ensure context and user model state exist
    if not state_dict or "user_model_state" not in state_dict:
        return "Error: TutorContext state or UserModelState not found in tool_context.state."

    if not topic or not isinstance(topic, str):
        return "Error: Invalid topic provided for user model update."

    # Initialize concept if needed within the state dictionary
    user_model_state_dict = state_dict.setdefault("user_model_state", UserModelState().model_dump())
    concepts_dict = user_model_state_dict.setdefault("concepts", {})
    # Ensure concept_state is a dictionary for modification
    concept_state = concepts_dict.setdefault(topic, UserConceptMastery().model_dump())
    if not isinstance(concept_state, dict):
        # If it exists but is not a dict (unlikely), reset it
        concept_state = UserConceptMastery().model_dump()
        concepts_dict[topic] = concept_state 

    concept_state['last_interaction_outcome'] = outcome

    # Update last_accessed with ISO 8601 timestamp
    concept_state['last_accessed'] = last_accessed or datetime.now().isoformat()

    # Add confusion point if provided
    if confusion_point:
        confusion_points_list = concept_state.setdefault("confusion_points", [])
        if confusion_point not in confusion_points_list:
            confusion_points_list.append(confusion_point)

    # Update attempts and mastery for evaluative outcomes
    if outcome in ['correct', 'incorrect', 'mastered', 'struggled']:
        concept_state['attempts'] = concept_state.get('attempts', 0) + 1

        # Adjust mastery level based on outcome
        mastery_level = concept_state.get('mastery_level', 0.0)
        if outcome in ['correct', 'mastered']:
            mastery_level = min(1.0, mastery_level + 0.2)
        elif outcome in ['incorrect', 'struggled']:
            mastery_level = max(0.0, mastery_level - 0.1)
        concept_state['mastery_level'] = mastery_level
            
        # Adjust learning pace if struggling
        if len(concept_state.get("confusion_points", [])) > 2:
            current_pace = user_model_state_dict.get("learning_pace_factor", 1.0)
            user_model_state_dict["learning_pace_factor"] = max(0.5, current_pace - 0.1)

    # Update mastered objectives if provided
    if mastered_objective_title:
        mastered_list = user_model_state_dict.setdefault("mastered_objectives_current_section", [])
        if mastered_objective_title not in mastered_list:
            mastered_list.append(mastered_objective_title)
            print(f"[Tool] Marked objective '{mastered_objective_title}' as mastered for current section.")

    print(f"[Tool] Updated '{topic}' - Mastery: {concept_state.get('mastery_level', 0.0):.2f}, "
          f"Pace: {user_model_state_dict.get('learning_pace_factor', 1.0):.2f}")
    return f"User model updated for {topic}." # Note: state is modified in-place

@FunctionTool
def update_explanation_progress(tool_context: ToolContext, segment_index: int) -> str:
    """DEPRECATED: The Orchestrator manages micro-steps directly."""
    return "Error: This tool is deprecated. Orchestrator manages micro-steps."

@FunctionTool
async def get_user_model_status(tool_context: ToolContext, topic: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves detailed user model state, optionally for a specific topic."""
    print(f"[Tool get_user_model_status] Retrieving status for topic '{topic}'")

    state_dict = tool_context.state
    user_model_state_dict = state_dict.get("user_model_state")

    if not user_model_state_dict:
        return {"error": "No user model state found in context."}

    if topic:
        concepts_dict = user_model_state_dict.get("concepts", {})
        if topic not in concepts_dict:
            return {
                "topic": topic,
                "exists": False,
                "message": "Topic not found in user model."
            }
            
        concept = concepts_dict[topic] # Should be a dictionary now
        return {
            "topic": topic,
            "exists": True,
            "mastery_level": concept.get("mastery_level"),
            "attempts": concept.get("attempts"),
            "last_outcome": concept.get("last_interaction_outcome"),
            "confusion_points": concept.get("confusion_points", []),
            "last_accessed": concept.get("last_accessed")
        }
    
    # Return full state summary if no specific topic requested
    # Already a dictionary, just return it
    return user_model_state_dict 

@FunctionTool
async def reflect_on_interaction(
    tool_context: ToolContext,
    topic: str,
    interaction_summary: str, # e.g., "User answered checking question incorrectly."
    user_response: Optional[str] = None, # The actual user answer/input
    feedback_provided_data: Optional[Dict] = None # Pass feedback as dict from state
) -> Dict[str, Any]:
    """
    Analyzes the last interaction for a given topic, identifies potential reasons for user difficulty,
    and suggests adaptive next steps for the Orchestrator.
    """
    print(f"[Tool reflect_on_interaction] Called for topic '{topic}'. Summary: {interaction_summary}")

    feedback_provided: Optional[QuizFeedbackItem] = None
    if feedback_provided_data:
        try:
            feedback_provided = QuizFeedbackItem.model_validate(feedback_provided_data)
        except Exception as parse_err:
            print(f"[Tool reflect_on_interaction] Warning: Could not parse feedback_provided data: {parse_err}")

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
    ctx: ToolContext
) -> Union[FocusObjective, str]:
    """Calls the Planner Agent to determine the next learning focus objective."""
    print("[Tool call_planner_agent] Calling Planner Agent...")
    # --- Prepare Input for Planner ---
    kb_content = await read_knowledge_base(ctx) # Fetch KB content
    if isinstance(kb_content, str) and kb_content.startswith("Error:"):
        return kb_content # Propagate error

    # Optionally get user state summary (assuming get_user_model_status returns a suitable dict/string)
    user_state_data = await get_user_model_status(ctx)
    # Ensure user_state_data is serializable to JSON
    if isinstance(user_state_data, dict):
        try:
            user_state_summary = json.dumps(user_state_data)
        except TypeError as e:
            user_state_summary = f"Error serializing user state: {e}"
            print(f"[Tool call_planner_agent] {user_state_summary}")
    else:
        user_state_summary = str(user_state_data)

    planner_prompt = f"""
    Knowledge Base Summary:
    {kb_content}

    User Model State Summary:
    {user_state_summary}

    Based on the above information, determine the next single learning focus objective for the user. Output ONLY the FocusObjective JSON object.
    """
    try:
        # --- Import and Create Agent *Inside* ---
        from ai_tutor.agents.planner_agent import create_planner_agent # Import here
        # ----------------------------------------

        planner_agent = create_planner_agent() # Create planner (no longer needs vs_id)
        run_config = RunConfig(
            # workflow_name="Orchestrator_PlannerCall", # Remove invalid args
            # group_id=str(ctx.state.session_id)       # Remove invalid args
        )

        # Run the planner agent with the prepared prompt
        result = await Runner.run(
            planner_agent,
            planner_prompt,
            session_id=str(ctx.state.session_id), # Pass session_id for context
            run_config=run_config # Pass the cleaned run_config
        )

        # Check the type of the final output BEFORE trying to access attributes
        if isinstance(result.final_output, FocusObjective):
            focus_objective = result.final_output
            print(f"[Tool call_planner_agent] Planner returned focus: {focus_objective.topic}")
            # Store the new focus in context state dictionary
            ctx.state["current_focus_objective"] = focus_objective.model_dump() # Modify state dict
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
    tool_context: ToolContext,
    topic: str,
    explanation_details: str # e.g., "Explain the concept generally", "Provide an example"
) -> Union[ExplanationResult, str]:
    """Calls the Teacher Agent to provide an explanation for a specific topic/detail."""
    print(f"[Tool call_teacher_agent] Requesting explanation for '{topic}': {explanation_details}")
    state_dict = tool_context.state # Access state dict
    try:
        # --- Import and Create Agent *Inside* ---
        from ai_tutor.agents.teacher_agent import create_interactive_teacher_agent # Naming needs update based on actual refactor
        # ----------------------------------------
        # Vector store ID should be part of the state dictionary
        vector_store_id = state_dict.get("vector_store_id")
        if not vector_store_id:
            return "Error: Vector store ID not found in context state for Teacher."
        
        session_id = state_dict.get("session_id") # Get session_id from state if available

        teacher_agent = create_interactive_teacher_agent(vector_store_id) # TODO: Update function name if changed
        run_config = RunConfig(
            # Remove invalid args
        )

        teacher_prompt = f"""
        Explain the topic: '{topic}'.
        Specific instructions for this explanation: {explanation_details}.
        Use the file_search tool if needed to find specific information or examples from the documents.
        Format your response ONLY as an ExplanationResult object containing the explanation text in the 'details' field.
        """

        result = await Runner.run(
            teacher_agent,
            teacher_prompt,
            session_id=session_id, # Pass session_id if available
            # state=state_dict, # Pass the state dictionary if needed by the runner/agent directly (unlikely for standard ADK run)
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
                state_dict["last_interaction_summary"] = f"Teacher explained {topic}." # Update summary in state
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
            state_dict["last_interaction_summary"] = f"Teacher explained {topic} (raw string)."
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
    tool_context: ToolContext,
    topic: str,
    instructions: str # e.g., "Create one medium difficulty question", "Create a 3-question quiz covering key concepts"
) -> Union[QuizCreationResult, str]:
    """Calls the Quiz Creator Agent to generate one or more quiz questions."""
    print(f"[Tool call_quiz_creator_agent] Requesting quiz creation for '{topic}': {instructions}")
    state_dict = tool_context.state # Access state dict
    try:
        # --- Import and Create Agent *Inside* ---
        from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent # Or the new tool function name
        # ----------------------------------------
        
        session_id = state_dict.get("session_id") # Get session_id from state

        # Assuming create_quiz_creator_agent is the function that returns the agent instance
        quiz_creator_agent = create_quiz_creator_agent() # TODO: Update function name if changed. Pass API key if needed.
        run_config = RunConfig(
            # Remove invalid args
        )

        quiz_creator_prompt = f"""
        Create quiz questions based on the following instructions:
        Topic: '{topic}'
        Instructions: {instructions}
        Format your response ONLY as a QuizCreationResult object. Include the created question(s) in the appropriate field ('question' or 'quiz').
        """

        result = await Runner.run(
            quiz_creator_agent,
            quiz_creator_prompt,
            session_id=session_id, # Pass session_id if available
            run_config=run_config
        )

        quiz_creation_result = result.final_output_as(QuizCreationResult)
        if quiz_creation_result and quiz_creation_result.status == "created":
            question_count = 1 if quiz_creation_result.question else len(quiz_creation_result.quiz.questions) if quiz_creation_result.quiz else 0
            print(f"[Tool call_quiz_creator_agent] Quiz Creator created {question_count} question(s) for '{topic}'.")
            # Store the created question if it's a single one for evaluation
            if quiz_creation_result.question:
                 state_dict["current_quiz_question"] = quiz_creation_result.question
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