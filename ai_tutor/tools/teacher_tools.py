from __future__ import annotations
import logging
from typing import List, Dict, Any, Literal, cast
import uuid
from uuid import UUID

# Import necessary response models from api_models.py
from ai_tutor.api_models import ExplanationResponse, QuestionResponse, MessageResponse # Add others if needed
# from ai_tutor.agents.models import QuizQuestion # If asking structured questions
from ..agents.models import QuizQuestion # Use relative import
from ai_tutor.context import TutorContext # Import TutorContext for type hint

# Use ADK imports
from google.adk.tools import LongRunningFunctionTool, ToolContext, FunctionTool
# Use types from the base google-generativeai library as used by ADK internally
# Import Content and Part from the content_types submodule
# from google.generativeai.types import FunctionResponse # Old incorrect import
from google.genai.types import Content, Part, FunctionResponse # Corrected import including FunctionResponse
from google.adk.events import Event, EventActions # Import Event classes

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Tool implementations now return the data structure the API needs

@FunctionTool
async def present_explanation(
    ctx: ToolContext, # Use ADK ToolContext directly if RunContextWrapper is removed
    explanation_text: str,
    segment_index: int,
    is_last_segment: bool
) -> ExplanationResponse:
    """Presents a segment of explanation text to the user."""
    # Access state via ctx.state
    state_dict = ctx.state
    topic = state_dict.get("current_teaching_topic", "Unknown Topic")
    print(f"[TeacherTool] Presenting explanation for '{topic}', segment {segment_index}")

    # Update state dictionary directly
    user_model_state = state_dict.setdefault("user_model_state", UserModelState().model_dump())
    user_model_state['current_topic_segment_index'] = segment_index + 1 # Progress index
    user_model_state['pending_interaction_type'] = None # Clear pending interaction
    user_model_state['pending_interaction_details'] = None
    state_dict["last_interaction_summary"] = f"Presented explanation segment {segment_index} for {topic}."
    # Note: Tool needs to signal state changes back via EventActions or rely on framework

    # Return the structured data the API will send
    return ExplanationResponse(
        response_type="explanation",
        text=explanation_text,
        topic=topic,
        segment_index=segment_index,
        is_last_segment=is_last_segment,
    )

@FunctionTool
async def ask_checking_question(
    ctx: ToolContext, # Use ADK ToolContext
    question: QuizQuestion # Use the QuizQuestion model for structure
) -> QuestionResponse:
    """Asks the user a question to check their understanding of the current topic."""
    state_dict = ctx.state
    topic = state_dict.get("current_teaching_topic", "Unknown Topic")
    interaction_id = f"chk_{uuid.uuid4().hex[:6]}"
    print(f"[TeacherTool] Asking checking question for '{topic}': {question.question[:50]}...")

    # Update state dictionary directly
    user_model_state = state_dict.setdefault("user_model_state", UserModelState().model_dump())
    user_model_state['pending_interaction_type'] = 'checking_question'
    user_model_state['pending_interaction_details'] = {
        "interaction_id": interaction_id,
        "question": question.model_dump() # Store the question details
    }
    state_dict["current_quiz_question"] = question.model_dump() # Store as dict
    state_dict["last_interaction_summary"] = f"Asked checking question on {topic}."
    # Note: Tool needs to signal state changes back

    # Add interaction_id to the question object sent back if needed
    # (or frontend can handle correlation based on the response wrapper)

    # Return the structured data the API will send
    return QuestionResponse(
        response_type="question",
        question=question, # Send the full question object
        topic=topic,
        # context = interaction_id # Optionally add context/ID here if needed by frontend
    )

# Add a tool for prompting summary if desired:
# @function_tool
# async def prompt_for_summary(ctx: RunContextWrapper[TutorContext], topic: str) -> MessageResponse: ... 

class AskUserQuestionTool(LongRunningFunctionTool):
    """
    A long-running tool used by the Teacher Agent to ask the user a question
    and wait for their answer.
    """
    def __init__(self):
        # The actual function logic is within 'run'
        super().__init__(func=self.run_tool_logic)
        self.name = "ask_user_question_and_get_answer"
        self.description = "Asks the user a multiple-choice question and waits for their answer index."
        
        # Define the input schema explicitly using basic types
        # This mirrors the required fields of QuizQuestion
        self.input_schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
                "correct_answer_index": {"type": "integer"},
                "explanation": {"type": "string"},
                "difficulty": {"type": "string"},
                "related_section": {"type": "string"}
            },
            "required": [
                "question", 
                "options", 
                "correct_answer_index", 
                "explanation", 
                "difficulty", 
                "related_section"
            ]
        }

    async def run_tool_logic(self, question_data: Dict[str, Any], tool_context: ToolContext):
        """
        The core logic: signal pause, wait for resume, return answer.
        """
        try:
            # Validate input using Pydantic (optional but recommended)
            question_obj = QuizQuestion.model_validate(question_data)
            logger.info(f"AskUserQuestionTool: Asking question - '{question_obj.question[:50]}...'")

            # Create and yield the pause event with question data
            pause_event = Event(
                author=tool_context.agent_name, # Associate with the agent calling the tool
                content=Content(role="tool", parts=[Part.from_text(f"Waiting for user answer to question: {question_obj.question[:30]}...")]), # Use imported Content/Part factory
                actions=EventActions(
                    # Use a custom action field to signal the pause and carry data
                    custom_action={
                        "type": "ask_question",
                        "data": question_obj.model_dump()
                    }
                )
            )

            # Yield the pause event
            yield pause_event
            logger.info(f"{self.name}: Yielded pause signal event for tool_call_id {tool_context.function_call_id}.")
            # --- Execution Pauses Here ---
            # The generator finishes here after yielding the pause event.
            # The agent's execution resumes when a FunctionResponse event is processed.

        except Exception as e:
            error_msg = f"Error in {self.name}: {str(e)}"
            logger.error(error_msg)

            # Yield an error event
            error_event = Event(
                author=tool_context.agent_name,
                content=Content( # Use imported Content
                    role="tool",
                    parts=[Part(text=error_msg)]
                ),
                actions=EventActions(
                    custom_action={
                        "type": "error",
                        "data": {"message": error_msg}
                    }
                )
            )
            yield error_event
            # Generator finishes after yielding error

# Instantiate the tool for export
ask_user_question_and_get_answer_tool = AskUserQuestionTool() 