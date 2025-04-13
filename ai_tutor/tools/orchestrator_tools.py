from __future__ import annotations
from google.adk.tools import FunctionTool, ToolContext, BaseTool
from google.adk.runners import Runner, RunConfig
# Import FunctionDeclaration and Schema from the low-level library for manual definition
# from google.genai.types import FunctionDeclaration, Schema # Remove this
from google.ai.generativelanguage import FunctionDeclaration, Schema, Tool, Type
from typing import Any, Optional, Literal, Union, cast, Dict, List
import os
from datetime import datetime
import traceback # Import traceback for error logging
import logging # Add logging import
import json
from uuid import UUID # Import UUID
from pydantic import ValidationError # Import for tool validation

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

# @FunctionTool # REMOVED Decorator
async def read_knowledge_base(tool_context: ToolContext) -> str:
    """Reads the Knowledge Base content stored in the Supabase 'folders' table."""
    log_prefix = f"[Tool read_knowledge_base Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.info(f"{log_prefix} Starting execution.")
    try:
        # --- Access state via ToolContext ---
        # State contains the TutorContext dict
        state_dict = tool_context.state
        tutor_context_data = state_dict # Assuming state IS the TutorContext dict

        folder_id_str = tutor_context_data.get("folder_id")
        user_id_str = tutor_context_data.get("user_id")
        analysis_result_data = tutor_context_data.get("analysis_result")

        if not folder_id_str or not user_id_str:
            logger.error(f"{log_prefix} Tool read_knowledge_base: Missing folder_id or user_id in context state.")
            return "Error: Folder ID or User ID not found in session context."

        folder_id = UUID(folder_id_str)
        user_id = UUID(user_id_str)

        # Check context first
        if analysis_result_data and analysis_result_data.get("analysis_text"):
            logger.info(f"{log_prefix} Tool read_knowledge_base: Found analysis text in context state.")
            return analysis_result_data["analysis_text"]

        # --- Query Supabase if not in context ---
        logger.info(f"{log_prefix} Tool read_knowledge_base: Querying Supabase for folder {folder_id}.")
        supabase = await get_supabase_client() # Or get client via other means
        response = supabase.table("folders").select("knowledge_base").eq("id", str(folder_id)).eq("user_id", str(user_id)).maybe_single().execute()

        if response.data and response.data.get("knowledge_base"):
            kb_content = response.data["knowledge_base"]
            logger.info(f"{log_prefix} Tool read_knowledge_base: Successfully read KB from Supabase for folder {folder_id}. Length: {len(kb_content)}")
            # **Important**: This tool likely CANNOT update the context directly.
            # The *calling agent* (Planner) needs to receive this content and potentially
            # use another tool (like update_user_model) or rely on the framework
            # to store relevant parts back into the state if needed later.
            return kb_content
        else:
            logger.warning(f"{log_prefix} Tool read_knowledge_base: KB not found for folder {folder_id}.")
            return f"Error: Knowledge Base not found for folder {folder_id}."
    except Exception as e:
        logger.exception(f"{log_prefix} Tool read_knowledge_base: Error accessing context or Supabase: {e}")
        return f"Error reading Knowledge Base: {e}"
    finally:
        logger.info(f"{log_prefix} Finished execution.")

# --- Manual Declaration for read_knowledge_base ---
read_knowledge_base_declaration = FunctionDeclaration(
    name="read_knowledge_base",
    description="Reads the Knowledge Base content (extracted text) stored in the Supabase 'folders' table for the current session.",
    # No parameters needed as info comes from ToolContext
    parameters=Schema(type=Type.OBJECT, properties={})
)

# --- Custom Tool Class using Manual Declaration ---
class ManualReadKBTool(BaseTool):
    def __init__(self):
        self._declaration = read_knowledge_base_declaration
        self._func = read_knowledge_base # Store the actual async function
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        """Returns the manually defined FunctionDeclaration."""
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Executes the underlying function logic. Args are ignored."""
        return await self._func(tool_context=tool_context)

# Instantiate the custom tool
read_knowledge_base_tool = ManualReadKBTool()

# --- Manual Declaration for get_document_content ---
get_document_content_declaration = FunctionDeclaration(
    name="get_document_content",
    description="Retrieves the text content of a document stored in Supabase Storage. The 'file_path_in_storage' should be the full path used when uploading (e.g., 'user_uuid/folder_uuid/filename.pdf').",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "file_path_in_storage": Schema(type=Type.STRING, description="The full path to the file in Supabase storage (e.g., 'user_uuid/folder_uuid/filename.pdf').")
        },
        required=["file_path_in_storage"]
    )
)

# --- Custom Tool Class using Manual Declaration ---
class ManualDocContentTool(BaseTool):
    def __init__(self):
        self._declaration = get_document_content_declaration
        self._func = get_document_content # Store the actual async function
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        """Returns the manually defined FunctionDeclaration."""
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Executes the underlying function logic."""
        # Extract args based on the declaration
        file_path = args.get("file_path_in_storage")
        if file_path is None:
             # This shouldn't happen if the LLM respects the schema, but handle defensively
             return "[Error: Missing required argument 'file_path_in_storage']"
        return await self._func(tool_context=tool_context, file_path_in_storage=file_path)

# Remove the decorator from the function definition
# @FunctionTool(declaration=get_document_content_declaration) # REMOVED
async def get_document_content(tool_context: ToolContext, file_path_in_storage: str) -> str:
    """
    Retrieves the text content of a document stored in Supabase Storage.
    The 'file_path_in_storage' should be the full path used when uploading
    (e.g., 'user_uuid/folder_uuid/filename.pdf').
    (Now called by ManualDocContentTool.run_async)
    """
    log_prefix = f"[Tool get_document_content Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.info(f"{log_prefix} Starting execution for path '{file_path_in_storage}'")
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
            logger.info(f"{log_prefix} Successfully fetched and decoded text. Length: {len(content)}")
            return content
        except UnicodeDecodeError:
            logger.warning(f"{log_prefix} Tool get_document_content: Could not decode '{file_path_in_storage}' as UTF-8. Returning raw representation.")
            return f"[Binary Content: {file_path_in_storage}]" # Placeholder for non-text
        except Exception as decode_err:
            logger.error(f"{log_prefix} Tool get_document_content: Error decoding content for '{file_path_in_storage}': {decode_err}")
            return f"[Error decoding content for {file_path_in_storage}]"

    except Exception as e:
        # Log Supabase client errors (check exception type for specifics)
        logger.exception(f"{log_prefix} Tool get_document_content: Failed to download '{file_path_in_storage}' from Supabase: {e}")
        return f"[Error retrieving document content for {file_path_in_storage}]"
    finally:
        logger.info(f"{log_prefix} Finished execution for path '{file_path_in_storage}'")

# Instantiate the custom tool
get_document_content_tool = ManualDocContentTool()

# --- Orchestrator Tool Implementations ---

@FunctionTool
async def call_quiz_creator_mini(
    tool_context: google.adk.tools.ToolContext,
    topic: str
) -> Union[QuizQuestion, str]: # Return Question object or error string
    """DEPRECATED: Use call_quiz_creator_agent instead. Generates a single multiple-choice question."""
    log_prefix = f"[Tool call_quiz_creator_mini Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.warning(f"{log_prefix} Deprecated tool called for topic '{topic}'")
    return "Error: This tool is deprecated. Use call_quiz_creator_agent to invoke the quiz creator."

# @FunctionTool # REMOVED
async def call_quiz_teacher_evaluate(tool_context: ToolContext, user_answer_index: int) -> Union[QuizFeedbackItem, str]:
    """Evaluates the user's answer to the current question using the Quiz Teacher logic (via helper function)."""
    log_prefix = f"[Tool call_quiz_teacher_evaluate Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.info(f"{log_prefix} Starting evaluation for answer index {user_answer_index}.")
    result = "Error: Evaluation failed unexpectedly."
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

        print(f"{log_prefix} Evaluating answer for question: {question_to_evaluate.question[:50]}...")
        
        # Re-parse TutorContext to pass to the helper if needed
        # Ideally, evaluate_single_answer would only take necessary data, not the whole context object
        tutor_context_for_helper = TutorContext.model_validate(state_dict) 
        
        feedback_item = await evaluate_single_answer(
            question=question_to_evaluate,
            user_answer_index=user_answer_index,
            context=tutor_context_for_helper # Pass re-parsed context if needed
        )

        if feedback_item:
            print(f"{log_prefix} Evaluation complete. Correct: {feedback_item.is_correct}")
            result = feedback_item
            return result
        else:
            # The evaluate_single_answer helper should ideally return feedback or raise
            error_msg = f"Error: Evaluation failed for question on topic '{getattr(question_to_evaluate, 'related_section', 'N/A')}'."
            print(f"{log_prefix} {error_msg}")
            result = error_msg # Return error string
            return result
            
    except Exception as e:
        error_msg = f"Exception in evaluation: {str(e)}"
        print(f"{log_prefix} {error_msg}")
        result = error_msg # Return error string
        return result
    finally:
        logger.info(f"{log_prefix} Finished evaluation. Result Type: {type(result).__name__}")

# --- Manual Declaration for call_quiz_teacher_evaluate ---
call_quiz_teacher_evaluate_declaration = FunctionDeclaration(
    name="call_quiz_teacher_evaluate",
    description="Evaluates the user's answer (index) to the current question.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "user_answer_index": Schema(type=Type.INTEGER, description="The 0-based index of the user's selected answer.")
        },
        required=["user_answer_index"]
    )
)

# --- Custom Tool Class for call_quiz_teacher_evaluate ---
class ManualEvaluateTool(BaseTool):
    def __init__(self):
        self._declaration = call_quiz_teacher_evaluate_declaration
        self._func = call_quiz_teacher_evaluate
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        answer_index = args.get("user_answer_index")
        if answer_index is None or not isinstance(answer_index, int):
            return "[Error: Missing or invalid required argument 'user_answer_index']"
        return await self._func(tool_context=tool_context, user_answer_index=answer_index)

# Instantiate the custom tool
call_quiz_teacher_evaluate_tool = ManualEvaluateTool()

# @FunctionTool # REMOVED
def determine_next_learning_step(tool_context: ToolContext) -> Dict[str, Any]:
    """DEPRECATED: The Planner agent now determines the next focus. Use call_planner_agent."""
    log_prefix = f"[Tool determine_next_learning_step Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.warning(f"{log_prefix} Deprecated tool called.")
    return {"error": "This tool is deprecated. Use call_planner_agent to get the next focus."}

# Add decorator back to the function definition
@FunctionTool
async def update_user_model(
    tool_context: ToolContext, # Keep in function signature for ADK injection
    topic: str,
    outcome: str,
    confusion_point: str, # Ensure this is required
    last_accessed: str, # Ensure this is required
    mastered_objective_title: str # Ensure this is required
) -> str:
    """Updates the user model state (within tool_context.state) with interaction outcomes and temporal data.

    Note: For parameters that are optional in practice, pass an empty string ("") when no value is available.
    - confusion_point: Pass "" if no confusion point identified
    - last_accessed: Pass "" to use current timestamp
    - mastered_objective_title: Pass "" if no objective was mastered
    """
    log_prefix = f"[Tool update_user_model Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.info(f"{log_prefix} Starting update for topic '{topic}' with outcome '{outcome}'")
    result = f"User model update failed for {topic}."
    try:
        # --- Add validation for outcome string ---
        valid_outcomes = {'correct', 'incorrect', 'mastered', 'struggled', 'explained'}
        if outcome not in valid_outcomes:
            error_msg = f"Error: Invalid outcome '{outcome}' provided. Must be one of: {valid_outcomes}"
            print(f"{log_prefix} {error_msg}")
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

        # Update last_accessed with ISO 8601 timestamp
        # Use current time if last_accessed is empty
        if last_accessed and last_accessed.strip():
            concept_state['last_accessed'] = last_accessed
        else:
            concept_state['last_accessed'] = datetime.now().isoformat()

        # Add confusion point if it's not empty
        if confusion_point and confusion_point.strip():
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

        # Update mastered objectives if title is not empty
        if mastered_objective_title and mastered_objective_title.strip():
            mastered_list = user_model_state_dict.setdefault("mastered_objectives_current_section", [])
            if mastered_objective_title not in mastered_list:
                mastered_list.append(mastered_objective_title)
                print(f"{log_prefix} Marked objective '{mastered_objective_title}' as mastered for current section.")

        print(f"{log_prefix} Updated '{topic}' - Mastery: {concept_state.get('mastery_level', 0.0):.2f}, "
              f"Pace: {user_model_state_dict.get('learning_pace_factor', 1.0):.2f}")
        result = f"User model updated for {topic}." # Note: state is modified in-place
        logger.info(f"{log_prefix} Update successful.")
        return result
    except Exception as e:
         logger.exception(f"{log_prefix} Exception during update: {e}")
         # Return error message based on existing logic inside the function
         if "Invalid outcome" in str(e):
              result = f"Error: Invalid outcome '{outcome}' provided."
         elif "Invalid topic" in str(e):
              result = "Error: Invalid topic provided."
         elif "state or UserModelState not found" in str(e):
              result = "Error: TutorContext state or UserModelState not found."
         else:
              result = f"Error updating user model: {e}"
         return result
    finally:
         logger.info(f"{log_prefix} Finished update. Result: {result}")

# --- Manual Declaration for update_user_model ---
update_user_model_declaration = FunctionDeclaration(
    name="update_user_model",
    description="Updates the user model state (within tool_context.state) with interaction outcomes and temporal data.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "topic": Schema(type=Type.STRING, description="The topic the interaction was about."),
            "outcome": Schema(type=Type.STRING, description="Outcome ('correct', 'incorrect', 'mastered', 'struggled', 'explained')."),
            "confusion_point": Schema(type=Type.STRING, description="Specific point of confusion, or empty string."),
            "last_accessed": Schema(type=Type.STRING, description="ISO 8601 timestamp or empty string to use current time."),
            "mastered_objective_title": Schema(type=Type.STRING, description="Title of objective mastered, or empty string.")
        },
        # Explicitly list all parameters as required
        required=["topic", "outcome", "confusion_point", "last_accessed", "mastered_objective_title"]
    )
)

# Add decorator back to get_user_model_status
# @FunctionTool # REMOVED Decorator
async def get_user_model_status(tool_context: ToolContext, topic: str) -> Dict[str, Any]:
    """Retrieves detailed user model state for a specific topic."""
    log_prefix = f"[Tool get_user_model_status Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.info(f"{log_prefix} Retrieving status for topic '{topic}'")
    result = {"error": "Failed to retrieve user model status."}
    try:
        state_dict = tool_context.state
        user_model_state_dict = state_dict.get("user_model_state")

        if not user_model_state_dict:
            return {"error": "No user model state found in context."}

        concepts_dict = user_model_state_dict.get("concepts", {})
        if topic not in concepts_dict:
            result = {
                "topic": topic,
                "exists": False,
                "message": "Topic not found in user model."
            }
            logger.info(f"{log_prefix} Topic not found.")
            return result

        concept = concepts_dict[topic] # Should be a dictionary now
        result = {
            "topic": topic,
            "exists": True,
            "mastery_level": concept.get("mastery_level"),
            "attempts": concept.get("attempts"),
            "last_outcome": concept.get("last_interaction_outcome"),
            "confusion_points": concept.get("confusion_points", []),
            "last_accessed": concept.get("last_accessed")
        }
        logger.info(f"{log_prefix} Status retrieved successfully.")
        return result
    except Exception as e:
        logger.exception(f"{log_prefix} Exception retrieving status: {e}")
        result = {"error": f"Exception retrieving user model status: {e}"}
        return result
    finally:
        logger.info(f"{log_prefix} Finished retrieval. Found: {result.get('exists', False)}")

# --- Manual Declaration for get_user_model_status ---
get_user_model_status_declaration = FunctionDeclaration(
    name="get_user_model_status",
    description="Retrieves detailed user model state for a specific topic.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "topic": Schema(type=Type.STRING, description="The specific topic to retrieve the status for.")
        },
        required=["topic"] # Topic is required
    )
)

# --- Custom Tool Class using Manual Declaration ---
class ManualGetUserModelStatusTool(BaseTool):
    def __init__(self):
        self._declaration = get_user_model_status_declaration
        self._func = get_user_model_status
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        topic = args.get("topic")
        if topic is None or not isinstance(topic, str):
            return "[Error: Missing or invalid required argument 'topic']"
        return await self._func(tool_context=tool_context, topic=topic)

# Instantiate the custom tool
get_user_model_status_tool = ManualGetUserModelStatusTool()

# Add decorator back to reflect_on_interaction
# @FunctionTool # REMOVED Decorator
async def reflect_on_interaction(
    tool_context: ToolContext,
    topic: str,
    interaction_summary: str, # e.g., "User answered checking question incorrectly."
    user_response: str, # Ensure required
    feedback_provided_data: Dict[str, Any] # Ensure required
) -> Dict[str, Any]:
    """
    Analyzes the last interaction for a given topic, identifies potential reasons for user difficulty,
    and suggests adaptive next steps for the Orchestrator.
    (Now potentially needs manual declaration too if issues persist)
    """
    log_prefix = f"[Tool reflect_on_interaction Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.info(f"{log_prefix} Starting reflection for topic '{topic}'. Summary: {interaction_summary}")
    result = {"error": "Reflection failed."}
    try:
        feedback_provided: Optional[QuizFeedbackItem] = None
        if feedback_provided_data and isinstance(feedback_provided_data, dict) and feedback_provided_data:
            try:
                feedback_provided = QuizFeedbackItem.model_validate(feedback_provided_data)
            except Exception as parse_err:
                print(f"{log_prefix} Warning: Could not parse feedback_provided data: {parse_err}")
        else:
            print(f"{log_prefix} No valid feedback_provided_data received.")

        # Basic reflection logic (can be enhanced, e.g., calling another LLM for deeper analysis)
        suggestions = []
        analysis = f"Reflection on interaction regarding '{topic}': {interaction_summary}. "

        # Only include user response in analysis if it's not empty
        if user_response and user_response.strip():
            analysis += f"User response: \\\"{user_response}\\\". "

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

        print(f"{log_prefix} Analysis: {analysis}. Suggestions: {suggestions}")
        result = {"analysis": analysis, "suggested_next_steps": suggestions}
        logger.info(f"{log_prefix} Reflection complete. Suggestions generated: {len(suggestions)}")
        return result
    except Exception as e:
        logger.exception(f"{log_prefix} Exception during reflection: {e}")
        result = {"error": f"Exception during reflection: {e}"}
        return result
    finally:
        logger.info(f"{log_prefix} Finished reflection.")

# --- Manual Declaration for reflect_on_interaction ---
reflect_on_interaction_declaration = FunctionDeclaration(
    name="reflect_on_interaction",
    description="Analyzes the last interaction for a given topic, identifies potential reasons for user difficulty, and suggests adaptive next steps.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "topic": Schema(type=Type.STRING, description="The topic the interaction was about."),
            "interaction_summary": Schema(type=Type.STRING, description="Summary of the interaction (e.g., 'User answered checking question incorrectly')."),
            "user_response": Schema(type=Type.STRING, description="The user's actual response text, or empty string."),
            # feedback_provided_data expects a dictionary, even if empty {}
            "feedback_provided_data": Schema(type=Type.OBJECT, description="Dictionary containing feedback details, or empty dict.")
        },
        # Explicitly require all parameters
        required=["topic", "interaction_summary", "user_response", "feedback_provided_data"]
    )
)

# --- Custom Tool Class for reflect_on_interaction ---
class ManualReflectTool(BaseTool):
    def __init__(self):
        self._declaration = reflect_on_interaction_declaration
        self._func = reflect_on_interaction
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        # Retrieve args, providing defaults only if absolutely necessary and handled in func
        return await self._func(
            tool_context=tool_context,
            topic=args.get("topic", ""), # Provide empty string if missing
            interaction_summary=args.get("interaction_summary", ""), # Provide empty string if missing
            user_response=args.get("user_response", ""), # Provide empty string if missing
            feedback_provided_data=args.get("feedback_provided_data", {}) # Provide empty dict if missing
        )

# Instantiate the custom tool
reflect_on_interaction_tool = ManualReflectTool()

@FunctionTool
async def call_planner_agent(
    ctx: ToolContext
) -> Union[FocusObjective, str]:
    """Calls the Planner Agent to determine the next learning focus objective. Expects JSON output."""
    log_prefix = f"[Tool call_planner_agent Session: {ctx.session_id if ctx else 'N/A'}]"
    logger.info(f"{log_prefix} Starting execution.")
    result: Union[FocusObjective, str] = "PLANNER_ERROR: Planner failed unexpectedly."
    try:
        # --- Prepare Input for Planner ---
        kb_content = await read_knowledge_base(ctx)
        if isinstance(kb_content, str) and kb_content.startswith("Error:"):
            return kb_content

        # Get user state for the current topic if available, otherwise get general state
        current_topic = ""
        if ctx.state and isinstance(ctx.state, dict):
            current_focus = ctx.state.get("current_focus_objective", {})
            if isinstance(current_focus, dict):
                current_topic = current_focus.get("topic", "general")
            else:
                current_topic = "general"
        else:
            current_topic = "general"

        user_state_data = await get_user_model_status(ctx, current_topic)
        if isinstance(user_state_data, dict):
            try:
                user_state_summary = json.dumps(user_state_data)
            except TypeError as e:
                user_state_summary = f"Error serializing user state: {e}"
                print(f"{log_prefix} {user_state_summary}")
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
                print(f"{log_prefix} {error_msg}")
                return "PLANNER_OUTPUT_ERROR: Planner returned no text."

            planner_output_text = last_event.content.parts[0].text
            print(f"{log_prefix} Planner Raw Output:\n{planner_output_text}")

            # --- Parse and Validate Output ---
            try:
                # Clean potential markdown ```json fences
                if planner_output_text.strip().startswith('```json'):
                     planner_output_text = planner_output_text.strip()[7:-3].strip()
                elif planner_output_text.strip().startswith('```'):
                     planner_output_text = planner_output_text.strip()[3:-3].strip()
                     
                planner_output_data = json.loads(planner_output_text)
                focus_objective = FocusObjective.model_validate(planner_output_data)
                print(f"{log_prefix} Planner parsed focus: {focus_objective.topic}")
                
                # Store the new focus in context state dictionary if state is available
                if ctx.state and isinstance(ctx.state, dict):
                    ctx.state["current_focus_objective"] = focus_objective.model_dump()
                elif ctx.state and hasattr(ctx.state, 'current_focus_objective'):
                     setattr(ctx.state, 'current_focus_objective', focus_objective.model_dump())
                     
                result = focus_objective # Return the validated Pydantic object
                return result
            except json.JSONDecodeError as json_err:
                error_msg = f"Error: Planner agent output was not valid JSON. Error: {json_err}. Output: {planner_output_text[:200]}..."
                print(f"{log_prefix} {error_msg}")
                result = "PLANNER_OUTPUT_ERROR: Planner output was not valid JSON."
                return result
            except ValidationError as pydantic_err:
                error_msg = f"Error: Planner agent output did not match FocusObjective schema. Error: {pydantic_err}. Output: {planner_output_text[:200]}..."
                print(f"{log_prefix} {error_msg}")
                result = "PLANNER_OUTPUT_ERROR: Planner output did not match schema."
                return result
            # -----------------------------

        except Exception as e:
            error_msg = f"EXCEPTION calling Planner Agent: {str(e)}\n{traceback.format_exc()}"
            print(f"{log_prefix} {error_msg}")
            result = "PLANNER_EXECUTION_ERROR: An exception occurred while running the planner."
            return result
        finally:
             log_result = result.topic if isinstance(result, FocusObjective) else result
             logger.info(f"{log_prefix} Finished execution. Result: {log_result}")

    except Exception as e:
        error_msg = f"EXCEPTION calling Planner Agent: {str(e)}\n{traceback.format_exc()}"
        print(f"{log_prefix} {error_msg}")
        result = "PLANNER_EXECUTION_ERROR: An exception occurred while running the planner."
        return result
    finally:
         log_result = result.topic if isinstance(result, FocusObjective) else result
         logger.info(f"{log_prefix} Finished execution. Result: {log_result}")

@FunctionTool
async def call_teacher_agent(
    tool_context: ToolContext,
    topic: str,
    explanation_details: str # e.g., "Explain the concept generally", "Provide an example"
) -> Union[ExplanationResult, str]:
    """Calls the Teacher Agent to provide an explanation for a specific topic/detail."""
    log_prefix = f"[Tool call_teacher_agent Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.info(f"{log_prefix} Starting execution for topic '{topic}': {explanation_details}")
    result: Union[ExplanationResult, str] = "TEACHER_ERROR: Teacher failed unexpectedly."
    state_dict = tool_context.state # Access state dict
    try:
        # --- Import and Create Agent *Inside* ---
        from ai_tutor.agents.teacher_agent import create_interactive_teacher_agent # Naming needs update based on actual refactor
        # ----------------------------------------
        # Vector store ID should be part of the state dictionary
        vector_store_id = state_dict.get("vector_store_id")
        # !! IMPORTANT: The vector store ID requirement was likely removed from the Teacher agent
        # !! Remove this check if the teacher no longer needs it.
        # if not vector_store_id:
        #     logger.error(f"{log_prefix} Vector store ID not found in context state for Teacher.")
        #     return "Error: Vector store ID not found in context state for Teacher."

        session_id = state_dict.get("session_id") # Get session_id from state if available
        user_id = state_dict.get("user_id") # Get user_id from state

        # Pass vector_store_id only if the agent function still requires it
        teacher_agent = create_interactive_teacher_agent() # Assuming no vector_store_id needed now
        run_config = RunConfig(
             group_id=str(session_id) if session_id else None,
             workflow_name="Orchestrator_TeacherCall"
        )

        teacher_prompt = f"""
        Explain the topic: '{topic}'.
        Specific instructions for this explanation: {explanation_details}.
        Use the file_search tool if needed to find specific information or examples from the documents.
        Format your response ONLY as an ExplanationResult object containing the explanation text in the 'details' field.
        """

        # --- Correctly handle run_async generator --- #
        adk_runner = Runner("ai_tutor", teacher_agent, tool_context.session_service) # Use session service from context
        final_agent_event: Optional[Event] = None
        logger.info(f"{log_prefix} Calling Teacher Agent run_async...")
        async for event in adk_runner.run_async(
            user_id=str(user_id) if user_id else None,
            session_id=str(session_id) if session_id else None,
            new_message=teacher_prompt,
            run_config=run_config
            # state=state_dict, # Pass state if runner needs it directly (unlikely)
        ):
            logger.info(f"{log_prefix} Teacher Event: Author={event.author}, Type={type(event.content)}, Actions={event.actions}")
            # Keep track of the last event from the agent itself
            if event.author == teacher_agent.name:
                final_agent_event = event

        # --- Process the final event --- #
        if not final_agent_event or not final_agent_event.content:
            error_msg = "TEACHER_OUTPUT_ERROR: Teacher agent finished without generating content."
            logger.error(f"{log_prefix} {error_msg}")
            result = error_msg
            return result

        # Attempt to parse the final event's content as ExplanationResult
        # Assuming the agent's last message follows the requested format
        final_output_text = None
        if final_agent_event.content.parts and final_agent_event.content.parts[0].text:
            final_output_text = final_agent_event.content.parts[0].text

        if final_output_text:
            try:
                 # Clean potential markdown fences
                 if final_output_text.strip().startswith('```json'):
                      final_output_text = final_output_text.strip()[7:-3].strip()
                 elif final_output_text.strip().startswith('```'):
                      final_output_text = final_output_text.strip()[3:-3].strip()

                 output_data = json.loads(final_output_text)
                 explanation_result = ExplanationResult.model_validate(output_data)

                 if explanation_result.status == "delivered":
                     logger.info(f"{log_prefix} Teacher delivered structured explanation for '{topic}'.")
                     if state_dict: # Update state only if available
                         state_dict["last_interaction_summary"] = f"Teacher explained {topic}."
                     result = explanation_result
                     return result
                 else:
                     # Handle structured failure/skipped cases
                     error_msg = f"TEACHER_RESULT_ERROR: Teacher agent returned status '{explanation_result.status}'. Details: {explanation_result.details}"
                     logger.error(f"{log_prefix} {error_msg}")
                     result = error_msg
                     return result
            except json.JSONDecodeError as json_err:
                 error_msg = f"TEACHER_OUTPUT_ERROR: Failed to decode Teacher agent output as JSON. Error: {json_err}. Output: {final_output_text[:200]}..."
                 logger.error(f"{log_prefix} {error_msg}")
                 result = error_msg
                 return result
            except ValidationError as pydantic_err:
                 error_msg = f"TEACHER_OUTPUT_ERROR: Teacher agent output did not match ExplanationResult schema. Error: {pydantic_err}. Output: {final_output_text[:200]}..."
                 logger.error(f"{log_prefix} {error_msg}")
                 result = error_msg
                 return result
        else:
             # Handle case where final event had no text
             error_msg = f"TEACHER_OUTPUT_ERROR: Teacher agent's final event had no text content."
             logger.error(f"{log_prefix} {error_msg}")
             result = error_msg
             return result

    except Exception as e:
        error_msg = f"Error calling Teacher Agent: {str(e)}\n{traceback.format_exc()}"
        logger.exception(f"{log_prefix} {error_msg}")
        result = error_msg
        return result
    finally:
         log_status = result.status if isinstance(result, ExplanationResult) else ("Error" if isinstance(result, str) else "Unknown")
         logger.info(f"{log_prefix} Finished execution. Result Status: {log_status}")

@FunctionTool
async def call_quiz_creator_agent(
    tool_context: ToolContext,
    topic: str,
    instructions: str # e.g., "Create one medium difficulty question", "Create a 3-question quiz covering key concepts"
) -> Union[QuizCreationResult, str]:
    """Calls the Quiz Creator Agent to generate one or more quiz questions."""
    log_prefix = f"[Tool call_quiz_creator_agent Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.info(f"{log_prefix} Starting execution for topic '{topic}': {instructions}")
    result: Union[QuizCreationResult, str] = "QUIZ_CREATOR_ERROR: Quiz Creator failed unexpectedly."
    state_dict = tool_context.state # Access state dict
    try:
        # --- Import and Create Agent *Inside* ---
        from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent # Or the new tool function name
        # ----------------------------------------

        session_id = state_dict.get("session_id") # Get session_id from state
        user_id = state_dict.get("user_id") # Get user_id from state

        # Assuming create_quiz_creator_agent is the function that returns the agent instance
        quiz_creator_agent = create_quiz_creator_agent() # TODO: Update function name if changed. Pass API key if needed.
        run_config = RunConfig(
            group_id=str(session_id) if session_id else None,
            workflow_name="Orchestrator_QuizCreatorCall"
        )

        quiz_creator_prompt = f"""
        Create quiz questions based on the following instructions:
        Topic: '{topic}'
        Instructions: {instructions}
        Format your response ONLY as a QuizCreationResult object. Include the created question(s) in the appropriate field ('question' or 'quiz').
        """

        # --- Correctly handle run_async generator --- #
        adk_runner = Runner("ai_tutor", quiz_creator_agent, tool_context.session_service)
        final_agent_event: Optional[Event] = None
        logger.info(f"{log_prefix} Calling Quiz Creator Agent run_async...")
        async for event in adk_runner.run_async(
            user_id=str(user_id) if user_id else None,
            session_id=str(session_id) if session_id else None,
            new_message=quiz_creator_prompt,
            run_config=run_config
        ):
            logger.info(f"{log_prefix} Quiz Creator Event: Author={event.author}, Type={type(event.content)}, Actions={event.actions}")
            # Keep track of the last event from the agent itself
            if event.author == quiz_creator_agent.name:
                final_agent_event = event

        # --- Process the final event --- #
        if not final_agent_event or not final_agent_event.content:
            error_msg = "QUIZ_CREATOR_OUTPUT_ERROR: Quiz Creator agent finished without generating content."
            logger.error(f"{log_prefix} {error_msg}")
            result = error_msg
            return result

        # Attempt to parse the final event's content as QuizCreationResult
        final_output_text = None
        if final_agent_event.content.parts and final_agent_event.content.parts[0].text:
            final_output_text = final_agent_event.content.parts[0].text

        if final_output_text:
            try:
                # Clean potential markdown fences
                if final_output_text.strip().startswith('```json'):
                     final_output_text = final_output_text.strip()[7:-3].strip()
                elif final_output_text.strip().startswith('```'):
                     final_output_text = final_output_text.strip()[3:-3].strip()

                output_data = json.loads(final_output_text)
                quiz_creation_result = QuizCreationResult.model_validate(output_data)

                if quiz_creation_result.status == "created":
                    question_count = 1 if quiz_creation_result.question else len(quiz_creation_result.quiz.questions) if quiz_creation_result.quiz else 0
                    logger.info(f"{log_prefix} Quiz Creator created {question_count} question(s) for '{topic}'.")
                    # Store the created question if it's a single one for evaluation
                    if quiz_creation_result.question and state_dict:
                         # Ensure current_quiz_question is stored as a dict
                         state_dict["current_quiz_question"] = quiz_creation_result.question.model_dump()
                         logger.info(f"{log_prefix} Stored single created question in context state.")
                    result = quiz_creation_result
                    return result
                else:
                    details = getattr(quiz_creation_result, 'details', 'No details provided.')
                    error_msg = f"QUIZ_CREATOR_RESULT_ERROR: Quiz Creator agent failed for '{topic}'. Status: {getattr(quiz_creation_result, 'status', 'unknown')}. Details: {details}"
                    logger.error(f"{log_prefix} {error_msg}")
                    result = error_msg
                    return result
            except json.JSONDecodeError as json_err:
                 error_msg = f"QUIZ_CREATOR_OUTPUT_ERROR: Failed to decode Quiz Creator agent output as JSON. Error: {json_err}. Output: {final_output_text[:200]}..."
                 logger.error(f"{log_prefix} {error_msg}")
                 result = error_msg
                 return result
            except ValidationError as pydantic_err:
                 error_msg = f"QUIZ_CREATOR_OUTPUT_ERROR: Quiz Creator output did not match QuizCreationResult schema. Error: {pydantic_err}. Output: {final_output_text[:200]}..."
                 logger.error(f"{log_prefix} {error_msg}")
                 result = error_msg
                 return result
        else:
             error_msg = f"QUIZ_CREATOR_OUTPUT_ERROR: Quiz Creator agent's final event had no text content."
             logger.error(f"{log_prefix} {error_msg}")
             result = error_msg
             return result

    except Exception as e:
        error_msg = f"Error calling Quiz Creator Agent: {str(e)}\n{traceback.format_exc()}"
        logger.exception(f"{log_prefix} {error_msg}")
        result = error_msg
        return result
    finally:
         log_status = result.status if isinstance(result, QuizCreationResult) else ("Error" if isinstance(result, str) else "Unknown")
         logger.info(f"{log_prefix} Finished execution. Result Status: {log_status}")

# Ensure all tools intended for the orchestrator are exported or available
__all__ = [
    # --- Agent Calling Helpers (Not ADK Tools) ---
    'call_planner_agent', # Keep if called directly from orchestrator logic
    'call_teacher_agent',
    'call_quiz_creator_agent',
    # --- Manual Tool Instances ---
    'get_user_model_status_tool',
    'update_user_model_tool',
    'reflect_on_interaction_tool',
    'call_quiz_teacher_evaluate_tool',
    # --- Deprecated/Removed --- 
    # ...
] 