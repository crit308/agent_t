from __future__ import annotations
from google.adk.tools import FunctionTool, ToolContext, BaseTool
from google.adk.runners import Runner, RunConfig
from google.genai.types import Content, Part  # Correct types here
from google.adk.events import Event          # Correct import for Event
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

# --- Import agent functions used by manual tools --- REMOVED (these don't exist as funcs)
# from ai_tutor.agents.teacher_agent import call_teacher_agent 
# from ai_tutor.agents.quiz_creator_agent import call_quiz_creator_agent

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
        self._func = read_knowledge_base # REINSTATED
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        """Returns the manually defined FunctionDeclaration."""
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Executes the underlying function logic. Args are ignored."""
        # Correctly call the stored function
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
        self._func = get_document_content # REINSTATED
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
             return "[Error: Missing required argument 'file_path_in_storage']"
        # Correctly call the stored function 
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

# Remove FunctionTool decorator from deprecated function
# @FunctionTool 
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
            print(f"{log_prefix} Evaluation helper returned no feedback item.")
            return "Error: Failed to get evaluation feedback."
    except ImportError as imp_err:
        print(f"{log_prefix} Import error for evaluation function: {imp_err}")
        return f"Error: Evaluation logic unavailable ({imp_err})."
    except Exception as e:
        print(f"{log_prefix} Exception during evaluation: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return f"Error evaluating answer: {e}"
    finally:
        print(f"{log_prefix} Finished evaluation. Result: {result}")

# Manual Declaration for call_quiz_teacher_evaluate
call_quiz_teacher_evaluate_declaration = FunctionDeclaration(
    name="call_quiz_teacher_evaluate",
    description="Evaluates the user's answer to the current multiple-choice question and returns feedback.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "user_answer_index": Schema(type=Type.INTEGER, description="The index (0-based) of the option selected by the user.")
        },
        required=["user_answer_index"]
    )
)

# Custom Tool Class using Manual Declaration
class ManualEvaluateTool(BaseTool):
    def __init__(self):
        self._declaration = call_quiz_teacher_evaluate_declaration
        self._func = call_quiz_teacher_evaluate # REINSTATED
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        """Returns the manually defined FunctionDeclaration."""
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Executes the underlying function logic."""
        answer_index = args.get("user_answer_index")
        if answer_index is None:
             return "[Error: Missing required argument 'user_answer_index']"
        # Correctly call the stored function
        return await self._func(tool_context=tool_context, user_answer_index=answer_index)

# Instantiate the custom tool
call_quiz_teacher_evaluate_tool = ManualEvaluateTool()

# @FunctionTool # REMOVED
def determine_next_learning_step(tool_context: ToolContext) -> Dict[str, Any]:
    """DEPRECATED: The Planner agent now determines the next focus. Use call_planner_agent."""
    log_prefix = f"[Tool determine_next_learning_step Session: {tool_context.session_id if tool_context else 'N/A'}]"
    logger.warning(f"{log_prefix} Deprecated tool called.")
    return {"error": "This tool is deprecated. Use call_planner_agent to get the next focus."}

# REMOVE the FunctionTool decorator - NOW HANDLED BY ManualUpdateUserModelTool
async def update_user_model(
    tool_context: ToolContext,
    topic: str,
    outcome: str,
    confusion_point: str,
    last_accessed: str,
    mastered_objective_title: str
) -> str:
    """Updates the user model state (within tool_context.state) with interaction outcomes and temporal data.

    Note: For parameters that are optional in practice, pass empty string or None.
    - confusion_point: Pass None or "" if no confusion point identified
    - last_accessed: Pass None or "" to use current timestamp
    - mastered_objective_title: Pass None or "" if no objective was mastered
    """
    log_prefix = f"[Tool update_user_model Session: {tool_context.session_id if tool_context else 'N/A'}]"
    # Handle optional parameters explicitly if they are None
    confusion_point = confusion_point or "" 
    last_accessed = last_accessed or ""
    mastered_objective_title = mastered_objective_title or ""
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
        # Use model_dump() for initial default if needed, then access as dict
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
    description="Updates the user model state with interaction outcomes and temporal data. Pass empty strings ('') for optional fields if no value is available.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "topic": Schema(type=Type.STRING, description="The topic the interaction related to."),
            "outcome": Schema(type=Type.STRING, description="Result of the interaction (correct, incorrect, mastered, struggled, explained)."),
            "confusion_point": Schema(type=Type.STRING, description="Specific point of confusion identified, or empty string."),
            "last_accessed": Schema(type=Type.STRING, description="ISO 8601 timestamp of interaction, or empty string to use current time."),
            "mastered_objective_title": Schema(type=Type.STRING, description="Title of the learning objective mastered, or empty string.")
        },
        # All parameters are required by the function logic (handle empty strings inside)
        required=["topic", "outcome", "confusion_point", "last_accessed", "mastered_objective_title"]
    )
)

# --- Custom Tool Class for update_user_model ---
class ManualUpdateUserModelTool(BaseTool):
    def __init__(self):
        self._declaration = update_user_model_declaration
        self._func = update_user_model # REINSTATED
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        """Returns the manually defined FunctionDeclaration."""
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Executes the underlying function logic, extracting args."""
        # Extract args safely, providing empty strings as fallback if missing (though schema requires them)
        topic = args.get("topic", "")
        outcome = args.get("outcome", "")
        # Pass None for optional args if missing, let the function handle defaults/None
        confusion_point = args.get("confusion_point")
        last_accessed = args.get("last_accessed")
        mastered_objective_title = args.get("mastered_objective_title")

        # Validate outcome here before calling the function might be slightly cleaner
        valid_outcomes = {'correct', 'incorrect', 'mastered', 'struggled', 'explained'}
        if outcome not in valid_outcomes:
            error_msg = f"[Tool Error] Invalid outcome '{outcome}' provided. Must be one of: {valid_outcomes}"
            logger.error(f"[Tool {self.name}] {error_msg}")
            return error_msg # Return error directly from tool run

        # Correctly call the stored function
        return await self._func(
            tool_context=tool_context,
            topic=topic,
            outcome=outcome,
            confusion_point=confusion_point,
            last_accessed=last_accessed,
            mastered_objective_title=mastered_objective_title
        )

# Instantiate the custom tool
update_user_model_tool = ManualUpdateUserModelTool()

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
        self._func = get_user_model_status # REINSTATED
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        topic = args.get("topic")
        if topic is None or not isinstance(topic, str):
            return "[Error: Missing or invalid required argument 'topic']"
        # Correctly call the stored function
        return await self._func(tool_context=tool_context, topic=topic)

# Instantiate the custom tool
get_user_model_status_tool = ManualGetUserModelStatusTool()

# Add decorator back to reflect_on_interaction
# @FunctionTool # REMOVED Decorator
async def reflect_on_interaction(
    tool_context: ToolContext,
    topic: str,
    interaction_summary: str, 
    user_response: str,
    feedback_provided_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyzes the last interaction for a given topic, identifies potential reasons for user difficulty,
    and suggests adaptive next steps for the Orchestrator.
    (Now potentially needs manual declaration too if issues persist)
    """
    log_prefix = f"[Tool reflect_on_interaction Session: {tool_context.session_id if tool_context else 'N/A'}]"
    # Handle optional parameters explicitly
    user_response = user_response or ""
    feedback_provided_data = feedback_provided_data or {}
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
        self._func = reflect_on_interaction # REINSTATED
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        # Retrieve args, pass None if missing for optional parameters
        topic = args.get("topic")
        interaction_summary = args.get("interaction_summary")
        user_response = args.get("user_response")
        feedback_provided_data = args.get("feedback_provided_data")

        # Basic validation (can be enhanced)
        if not topic or not interaction_summary:
             return "[Error: Missing required arguments 'topic' or 'interaction_summary']"
        
        # Correctly call the stored function
        return await self._func( 
            tool_context=tool_context,
            topic=topic,
            interaction_summary=interaction_summary,
            user_response=user_response, # Pass potential None
            feedback_provided_data=feedback_provided_data # Pass potential None
        )

# Instantiate the custom tool
reflect_on_interaction_tool = ManualReflectTool()

# REMOVE the FunctionTool decorator - NOW HANDLED BY ManualPlannerAgentTool
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
        Determine the next learning focus based on the available Knowledge Base and user model status.
        Use the `read_knowledge_base` tool first, then `get_user_model_status`.
        Return ONLY the FocusObjective JSON object.
        """

        try:
            from ai_tutor.agents.planner_agent import create_planner_agent
            planner_agent = create_planner_agent() # TODO: Update function name if changed. Pass API key if needed.
            
            # Check if ctx.state is available and has session_id
            session_id = None
            if ctx.state and isinstance(ctx.state, dict):
                 session_id = ctx.state.get("session_id")
            elif hasattr(ctx.state, 'session_id'): # Fallback if state is an object
                 session_id = getattr(ctx.state, 'session_id')
            
            # --- Correctly handle run_async generator --- #
            # Ensure session_service is available from the calling context
            if not ctx.session_service:
                 return "TOOL_CONFIG_ERROR: Session service not available in ToolContext."
                 
            adk_runner = Runner(
                app_name="ai_tutor", # Use consistent app name 
                agent=planner_agent, 
                session_service=ctx.session_service
            )
            final_agent_event: Optional[Event] = None
            logger.info(f"{log_prefix} Calling Planner Agent run_async...")
            async for event in adk_runner.run_async(
                user_id=str(ctx.user_id) if hasattr(ctx, 'user_id') else None,
                session_id=str(session_id) if session_id else None,
                new_message=Content(parts=[Part(text=planner_prompt)]) # Wrap prompt
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

# --- Manual Declaration for call_planner_agent ---
call_planner_agent_declaration = FunctionDeclaration(
    name="call_planner_agent",
    description="Calls the Planner Agent to determine the next learning focus objective. Expects JSON output.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={} # No parameters needed, uses context
    )
)

# --- Custom Tool Class for call_planner_agent ---
class ManualPlannerAgentTool(BaseTool):
    def __init__(self):
        self._declaration = call_planner_agent_declaration
        # self._func = call_planner_agent # REMOVED (confirming)
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        """Returns the manually defined FunctionDeclaration."""
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Executes the underlying function logic."""
        return await self._func(ctx=tool_context)

# Instantiate the custom tool
call_planner_agent_tool = ManualPlannerAgentTool()

# --- Manual Declaration for call_teacher_agent ---
call_teacher_agent_declaration = FunctionDeclaration(
    name="call_teacher_agent",
    description="Calls the teacher agent to generate explanations or handle student interactions.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "objective": Schema(type=Type.STRING, description="The learning objective or topic to focus on"),
            "student_input": Schema(type=Type.STRING, description="The student's question or input"),
            "interaction_type": Schema(type=Type.STRING, description="The type of interaction (explanation, question, feedback)")
        },
        required=["objective", "student_input", "interaction_type"]
    )
)

# --- Custom Tool Class for Teacher Agent ---
class ManualTeacherAgentTool(BaseTool):
    def __init__(self):
        self._declaration = call_teacher_agent_declaration
        # self._func = call_teacher_agent # REMOVED - Agent is run via Runner
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Instantiates and runs the Teacher agent via ADK Runner."""
        log_prefix = f"[Tool {self.name} Session: {tool_context.session_id if tool_context else 'N/A'}]"
        logger.info(f"{log_prefix} Starting teacher agent execution.")

        objective = args.get("objective")
        student_input = args.get("student_input")
        interaction_type = args.get("interaction_type")
        
        if not all([objective, student_input, interaction_type]):
            return "[Error: Missing required arguments for teacher agent tool call]"

        # --- Run Teacher Agent via Runner --- #
        try:
            from ai_tutor.agents.teacher_agent import create_interactive_teacher_agent
            teacher_agent = create_interactive_teacher_agent()
            
            # Get required context info
            session_id = str(getattr(tool_context, 'session_id', None))
            user_id = str(getattr(tool_context, 'user_id', None))
            session_service = getattr(tool_context, 'session_service', None)

            if not session_service:
                 logger.error(f"{log_prefix} Session service not available in ToolContext.")
                 return "TOOL_CONFIG_ERROR: Session service not available."
            if not user_id:
                 logger.error(f"{log_prefix} User ID not available in ToolContext.")
                 return "TOOL_CONFIG_ERROR: User ID not available."
            if not session_id:
                 logger.error(f"{log_prefix} Session ID not available in ToolContext.")
                 return "TOOL_CONFIG_ERROR: Session ID not available."

            # Construct a prompt for the teacher agent based on tool args
            # NOTE: This is a simplified prompt. The teacher agent expects FocusObjective.
            # This might need refinement based on how teacher_agent processes input.
            teacher_prompt = f"Objective: {objective}\nInteraction Type: {interaction_type}\nStudent Input: {student_input}"

            adk_runner = Runner(
                app_name="ai_tutor", 
                agent=teacher_agent, 
                session_service=session_service
            )
            
            last_event = None
            logger.info(f"{log_prefix} Calling Teacher Agent run_async...")
            async for event in adk_runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=Content(parts=[Part(text=teacher_prompt)])
            ):
                last_event = event
                # Log intermediate events if needed
                # logger.debug(f"{log_prefix} Teacher Agent Event: {event.author} - {event.content.parts[0].text[:50] if event.content and event.content.parts else 'No Text'}")

            if not last_event or not last_event.content or not last_event.content.parts:
                error_msg = "Error: Teacher agent did not return any content output."
                logger.error(f"{log_prefix} {error_msg}")
                return "TEACHER_OUTPUT_ERROR: Teacher returned no content."

            # Extract final result (assuming text or structured output)
            # TODO: Adapt based on Teacher Agent's actual output (TeacherTurnResult?)
            final_output = last_event.content.parts[0].text
            logger.info(f"{log_prefix} Teacher agent finished. Output: {final_output[:100]}...")
            return final_output # Or parse/validate if expecting structured JSON

        except Exception as e:
            error_msg = f"EXCEPTION calling Teacher Agent via tool: {str(e)}\n{traceback.format_exc()}"
            logger.exception(f"{log_prefix} {error_msg}")
            return "TEACHER_EXECUTION_ERROR: An exception occurred while running the teacher agent."
        # --- End Runner Logic --- #

# --- Manual Declaration for call_quiz_creator_agent ---
call_quiz_creator_declaration = FunctionDeclaration(
    name="call_quiz_creator_agent",
    description="Calls the quiz creator agent to generate quiz questions or provide feedback.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "objective": Schema(type=Type.STRING, description="The learning objective to test"),
            "request_type": Schema(type=Type.STRING, description="Type of request (create_quiz or provide_feedback)"),
            "student_response": Schema(type=Type.STRING, description="Student's answer for feedback (optional)")
        },
        required=["objective", "request_type"]
    )
)

# --- Custom Tool Class for Quiz Creator Agent ---
class ManualQuizCreatorAgentTool(BaseTool):
    def __init__(self):
        self._declaration = call_quiz_creator_declaration
        # self._func = call_quiz_creator_agent # REMOVED - Agent is run via Runner
        super().__init__(name=self._declaration.name, description=self._declaration.description)

    @property
    def function_declaration(self) -> FunctionDeclaration:
        return self._declaration

    async def run_async(self, *, args: Dict[str, Any], tool_context: ToolContext) -> Any:
        """Instantiates and runs the Quiz Creator agent via ADK Runner."""
        log_prefix = f"[Tool {self.name} Session: {tool_context.session_id if tool_context else 'N/A'}]"
        logger.info(f"{log_prefix} Starting quiz creator agent execution.")
        
        objective = args.get("objective")
        request_type = args.get("request_type")
        student_response = args.get("student_response", "") # Optional
        
        if not all([objective, request_type]):
            return "[Error: Missing required arguments for quiz creator agent tool call]"
            
        # --- Run Quiz Creator Agent via Runner --- #
        try:
            from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
            quiz_agent = create_quiz_creator_agent()
            
            # Get required context info
            session_id = str(getattr(tool_context, 'session_id', None))
            user_id = str(getattr(tool_context, 'user_id', None))
            session_service = getattr(tool_context, 'session_service', None)

            if not session_service:
                 logger.error(f"{log_prefix} Session service not available in ToolContext.")
                 return "TOOL_CONFIG_ERROR: Session service not available."
            if not user_id:
                 logger.error(f"{log_prefix} User ID not available in ToolContext.")
                 return "TOOL_CONFIG_ERROR: User ID not available."
            if not session_id:
                 logger.error(f"{log_prefix} Session ID not available in ToolContext.")
                 return "TOOL_CONFIG_ERROR: Session ID not available."

            # Construct prompt for the quiz agent
            quiz_prompt = f"Objective: {objective}\nRequest Type: {request_type}"
            if student_response:
                 quiz_prompt += f"\nStudent Response: {student_response}"
            quiz_prompt += "\nPlease generate the quiz or feedback as requested."

            adk_runner = Runner(
                app_name="ai_tutor", 
                agent=quiz_agent, 
                session_service=session_service
            )
            
            last_event = None
            logger.info(f"{log_prefix} Calling Quiz Creator Agent run_async...")
            async for event in adk_runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=Content(parts=[Part(text=quiz_prompt)])
            ):
                last_event = event
                # logger.debug(f"{log_prefix} Quiz Creator Event: {event.author} - {event.content.parts[0].text[:50] if event.content and event.content.parts else 'No Text'}")
            
            if not last_event or not last_event.content or not last_event.content.parts:
                error_msg = "Error: Quiz Creator agent did not return any content output."
                logger.error(f"{log_prefix} {error_msg}")
                return "QUIZ_OUTPUT_ERROR: Quiz Creator returned no content."

            # Extract final result (Quiz Creator outputs QuizCreationResult schema)
            quiz_output_text = last_event.content.parts[0].text
            logger.info(f"{log_prefix} Quiz Creator agent finished. Raw Output: {quiz_output_text[:100]}...")
            
            # Attempt to parse the JSON output according to QuizCreationResult schema
            try:
                # Clean potential markdown ```json fences
                if quiz_output_text.strip().startswith('```json'):
                    quiz_output_text = quiz_output_text.strip()[7:-3].strip()
                elif quiz_output_text.strip().startswith('```'):
                    quiz_output_text = quiz_output_text.strip()[3:-3].strip()
                    
                quiz_result_data = json.loads(quiz_output_text)
                # Validate against the expected Pydantic model
                quiz_creation_result = QuizCreationResult.model_validate(quiz_result_data)
                logger.info(f"{log_prefix} Successfully parsed QuizCreationResult. Status: {quiz_creation_result.status}")
                return quiz_creation_result.model_dump() # Return as dict for tool compatibility
            except json.JSONDecodeError as json_err:
                error_msg = f"Error: Quiz Creator output was not valid JSON. Error: {json_err}. Output: {quiz_output_text[:200]}..."
                logger.error(f"{log_prefix} {error_msg}")
                return {"error": "QUIZ_OUTPUT_ERROR: Quiz Creator output was not valid JSON."}
            except ValidationError as pydantic_err:
                error_msg = f"Error: Quiz Creator output did not match QuizCreationResult schema. Error: {pydantic_err}. Output: {quiz_output_text[:200]}..."
                logger.error(f"{log_prefix} {error_msg}")
                return {"error": "QUIZ_OUTPUT_ERROR: Quiz Creator output did not match schema."}

        except Exception as e:
            error_msg = f"EXCEPTION calling Quiz Creator Agent via tool: {str(e)}\n{traceback.format_exc()}"
            logger.exception(f"{log_prefix} {error_msg}")
            return "QUIZ_EXECUTION_ERROR: An exception occurred while running the quiz creator agent."
        # --- End Runner Logic --- #

# Instantiate the custom tools
quiz_creator_tool = ManualQuizCreatorAgentTool()

# Update __all__ to include the new tool instances
__all__ = [
    'read_knowledge_base_tool',
    'get_document_content_tool',
    'teacher_agent_tool',
    'quiz_creator_tool'
] 