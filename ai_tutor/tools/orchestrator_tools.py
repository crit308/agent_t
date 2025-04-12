from __future__ import annotations
from google.adk.tools import FunctionTool, ToolContext, BaseTool
from google.adk.runners import Runner, RunConfig
# Import FunctionDeclaration and Schema from the low-level library for manual definition
# from google.genai.types import FunctionDeclaration, Schema # Remove this
from google.ai.generativelanguage import FunctionDeclaration, Schema, Tool
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

# Remove decorator from the function definition
# @FunctionTool
async def update_user_model(
    tool_context: ToolContext, # Keep in function signature for ADK injection
    topic: str,
    outcome: str,
    confusion_point: Optional[str],
    last_accessed: Optional[str],
    mastered_objective_title: Optional[str]
) -> str:
    """Updates the user model state (within tool_context.state) with interaction outcomes and temporal data."""
    print(f"[Tool update_user_model] Updating '{topic}' with outcome '{outcome}'")
    
    # --- Add validation for outcome string ---
    valid_outcomes = {'correct', 'incorrect', 'mastered', 'struggled', 'explained'}
    if outcome not in valid_outcomes:
        error_msg = f"Error: Invalid outcome '{outcome}' provided. Must be one of: {valid_outcomes}"
        print(f"[Tool update_user_model] {error_msg}")
        return error_msg
    # --------------------------------------
        
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

    # Update last_accessed with ISO 8601 timestamp, handle None or str
    if last_accessed is None:
        concept_state['last_accessed'] = datetime.now().isoformat()
    elif isinstance(last_accessed, str):
        concept_state['last_accessed'] = last_accessed
    else:
        # Handle unexpected type for last_accessed if necessary, or default
        concept_state['last_accessed'] = datetime.now().isoformat() 

    # Add confusion point if provided and is a string
    if confusion_point and isinstance(confusion_point, str):
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

    # Update mastered objectives if provided and is a string
    if mastered_objective_title and isinstance(mastered_objective_title, str):
        mastered_list = user_model_state_dict.setdefault("mastered_objectives_current_section", [])
        if mastered_objective_title not in mastered_list:
            mastered_list.append(mastered_objective_title)
            print(f"[Tool] Marked objective '{mastered_objective_title}' as mastered for current section.")

    print(f"[Tool] Updated '{topic}' - Mastery: {concept_state.get('mastery_level', 0.0):.2f}, "
          f"Pace: {user_model_state_dict.get('learning_pace_factor', 1.0):.2f}")
    return f"User model updated for {topic}." # Note: state is modified in-place

# Manually define the FunctionDeclaration using types from google.ai.generativelanguage
update_user_model_declaration = FunctionDeclaration(
    name="update_user_model",
    description="Updates the user model state (within tool_context.state) with interaction outcomes and temporal data.",
    parameters=Schema(
        type="OBJECT",
        properties={
            "topic": Schema(type="STRING", description="The specific topic or concept being updated."),
            "outcome": Schema(type="STRING", description="The outcome of the interaction (e.g., 'correct', 'incorrect', 'explained')."),
            "confusion_point": Schema(type="STRING", description="Specific point of confusion noted by the user or system.", nullable=True),
            "last_accessed": Schema(type="STRING", description="ISO 8601 timestamp of the last access (optional).", nullable=True),
            "mastered_objective_title": Schema(type="STRING", description="Title of a learning objective now considered mastered (optional).", nullable=True),
        },
        required=["topic", "outcome"],
    )
)

# --- Define custom BaseTool class --- 
class UpdateUserModelTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="update_user_model", 
            description="Updates the user model state (within tool_context.state) with interaction outcomes and temporal data."
        )
        # Store the manual declaration
        self._declaration = update_user_model_declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Calls the underlying update_user_model function."""
        topic = args.get("topic")
        outcome = args.get("outcome")
        confusion_point = args.get("confusion_point")
        last_accessed = args.get("last_accessed")
        mastered_objective_title = args.get("mastered_objective_title")
        return await update_user_model(
            tool_context=tool_context, 
            topic=topic, 
            outcome=outcome, 
            confusion_point=confusion_point,
            last_accessed=last_accessed,
            mastered_objective_title=mastered_objective_title
        )

# Create an instance of the custom tool
update_user_model_tool = UpdateUserModelTool()

@FunctionTool
def update_explanation_progress(tool_context: ToolContext, segment_index: int) -> str:
    """DEPRECATED: The Orchestrator manages micro-steps directly."""
    return "Error: This tool is deprecated. Orchestrator manages micro-steps."

# Remove decorator from get_user_model_status
# @FunctionTool
async def get_user_model_status(tool_context: ToolContext, topic: Optional[str]) -> Dict[str, Any]:
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
    return user_model_state_dict 

# Manually define the FunctionDeclaration for get_user_model_status
get_user_model_status_declaration = FunctionDeclaration(
    name="get_user_model_status",
    description="Retrieves detailed user model state, optionally for a specific topic.",
    parameters=Schema(
        type="OBJECT",
        properties={
            "topic": Schema(type="STRING", description="The specific topic to get the status for (optional).", nullable=True),
        },
        # No required parameters as topic is optional
    )
)

# Define custom BaseTool class for get_user_model_status
class GetUserModelStatusTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="get_user_model_status",
            description="Retrieves detailed user model state, optionally for a specific topic."
        )
        self._declaration = get_user_model_status_declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Calls the underlying get_user_model_status function."""
        topic = args.get("topic") # Will be None if not provided by LLM
        return await get_user_model_status(tool_context=tool_context, topic=topic)

# Create an instance of the custom tool
get_user_model_status_tool = GetUserModelStatusTool()

# Remove decorator from reflect_on_interaction
# @FunctionTool
async def reflect_on_interaction(
    tool_context: ToolContext,
    topic: str,
    interaction_summary: str, # e.g., "User answered checking question incorrectly."
    user_response: Optional[str],
    feedback_provided_data: Optional[Dict]
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

# Manually define the FunctionDeclaration for reflect_on_interaction
reflect_on_interaction_declaration = FunctionDeclaration(
    name="reflect_on_interaction",
    description="Analyzes the last interaction for a given topic, identifies potential reasons for user difficulty, and suggests adaptive next steps.",
    parameters=Schema(
        type="OBJECT",
        properties={
            "topic": Schema(type="STRING", description="The topic the interaction was about."),
            "interaction_summary": Schema(type="STRING", description="A summary of what happened in the interaction (e.g., 'User answered correctly')."),
            "user_response": Schema(type="STRING", description="The user's actual response or input (optional).", nullable=True),
            "feedback_provided_data": Schema(type="OBJECT", description="The feedback item provided to the user (optional dictionary).", nullable=True),
        },
        required=["topic", "interaction_summary"],
    )
)

# Define custom BaseTool class for reflect_on_interaction
class ReflectOnInteractionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="reflect_on_interaction",
            description="Analyzes the last interaction for a given topic, identifies potential reasons for user difficulty, and suggests adaptive next steps."
        )
        self._declaration = reflect_on_interaction_declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Calls the underlying reflect_on_interaction function."""
        topic = args.get("topic")
        interaction_summary = args.get("interaction_summary")
        user_response = args.get("user_response")
        feedback_provided_data = args.get("feedback_provided_data")
        return await reflect_on_interaction(
            tool_context=tool_context,
            topic=topic,
            interaction_summary=interaction_summary,
            user_response=user_response,
            feedback_provided_data=feedback_provided_data
        )

# Create an instance of the custom tool
reflect_on_interaction_tool = ReflectOnInteractionTool()

# --- NEW Tools to Call Other Agents ---

@FunctionTool
async def call_planner_agent(
    ctx: ToolContext
) -> Union[FocusObjective, str]:
    """Calls the Planner Agent to determine the next learning focus objective. Expects JSON output."""
    print("[Tool call_planner_agent] Calling Planner Agent...")
    # --- Prepare Input for Planner ---
    kb_content = await read_knowledge_base(ctx)
    if isinstance(kb_content, str) and kb_content.startswith("Error:"):
        return kb_content

    user_state_data = await get_user_model_status_tool.run_async(args={}, tool_context=ctx) # Use the tool instance
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

    Based on the above information, determine the next single learning focus objective for the user. Output ONLY the JSON object string matching the FocusObjective schema.
    """
    try:
        from ai_tutor.agents.planner_agent import create_planner_agent
        planner_agent = create_planner_agent()
        
        # Check if ctx.state is available and has session_id
        session_id = None
        if ctx.state and isinstance(ctx.state, dict):
             session_id = ctx.state.get("session_id")
        elif hasattr(ctx.state, 'session_id'): # Fallback if state is an object
             session_id = getattr(ctx.state, 'session_id')
        
        run_config_kwargs = {}
        if session_id:
             run_config_kwargs['group_id'] = str(session_id)
             run_config_kwargs['workflow_name'] = "Orchestrator_PlannerCall"
        
        run_config = RunConfig(**run_config_kwargs)

        # Run the planner agent, expect text output
        # --- Modified Run Call ---
        last_event: Optional[Event] = None
        async for event in Runner.run_async(
            agent=planner_agent, # Correct keyword
            new_message=planner_prompt, # Correct keyword
            session_id=str(session_id) if session_id else None, # Correct keyword
            user_id=str(ctx.user_id) if hasattr(ctx, 'user_id') else None, # Correct keyword
            run_config=run_config
        ):
            last_event = event
            # We only care about the final text event
        
        if not last_event or not last_event.content or not last_event.content.parts or not last_event.content.parts[0].text:
            error_msg = "Error: Planner agent did not return any text output."
            print(f"[Tool call_planner_agent] {error_msg}")
            return "PLANNER_OUTPUT_ERROR: Planner returned no text."

        planner_output_text = last_event.content.parts[0].text
        print(f"[Tool call_planner_agent] Planner Raw Output:\n{planner_output_text}")

        # --- Parse and Validate Output ---
        try:
            # Clean potential markdown ```json fences
            if planner_output_text.strip().startswith('```json'):
                 planner_output_text = planner_output_text.strip()[7:-3].strip()
            elif planner_output_text.strip().startswith('```'):
                 planner_output_text = planner_output_text.strip()[3:-3].strip()
                 
            planner_output_data = json.loads(planner_output_text)
            focus_objective = FocusObjective.model_validate(planner_output_data)
            print(f"[Tool call_planner_agent] Planner parsed focus: {focus_objective.topic}")
            
            # Store the new focus in context state dictionary if state is available
            if ctx.state and isinstance(ctx.state, dict):
                ctx.state["current_focus_objective"] = focus_objective.model_dump()
            elif ctx.state and hasattr(ctx.state, 'current_focus_objective'):
                 setattr(ctx.state, 'current_focus_objective', focus_objective.model_dump())
                 
            return focus_objective # Return the validated Pydantic object
        except json.JSONDecodeError as json_err:
            error_msg = f"Error: Planner agent output was not valid JSON. Error: {json_err}. Output: {planner_output_text[:200]}..."
            print(f"[Tool call_planner_agent] {error_msg}")
            return "PLANNER_OUTPUT_ERROR: Planner output was not valid JSON."
        except ValidationError as pydantic_err:
            error_msg = f"Error: Planner agent output did not match FocusObjective schema. Error: {pydantic_err}. Output: {planner_output_text[:200]}..."
            print(f"[Tool call_planner_agent] {error_msg}")
            return "PLANNER_OUTPUT_ERROR: Planner output did not match schema."
        # -----------------------------

    except Exception as e:
        error_msg = f"EXCEPTION calling Planner Agent: {str(e)}\n{traceback.format_exc()}"
        print(f"[Tool] {error_msg}")
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
    'reflect_on_interaction_tool', # Export the custom tool instance
    'update_user_model_tool', # Export the tool instance, not the function
    'get_user_model_status_tool', # Export the custom tool instance
    # --- Removed ---
    # 'call_quiz_creator_mini',
    # 'determine_next_learning_step',
    # 'update_explanation_progress',
] 