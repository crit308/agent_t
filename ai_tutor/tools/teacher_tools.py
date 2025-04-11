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
from google.adk.tools import LongRunningFunctionTool, ToolContext, FunctionTool # Import FunctionTool
# Use types from the base google-generativeai library as used by ADK internally
# Import Content and Part directly
from google.generativeai.types import Content, Part
from google.adk.events import Event, EventActions # Import Event classes

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Tool implementations now return the data structure the API needs

@FunctionTool
async def present_explanation(
    ctx: RunContextWrapper[TutorContext],
    explanation_text: str,
    segment_index: int,
    is_last_segment: bool
) -> ExplanationResponse:
    """Presents a segment of explanation text to the user."""
    topic = ctx.context.current_teaching_topic or "Unknown Topic"
    print(f"[TeacherTool] Presenting explanation for '{topic}', segment {segment_index}")

    # Update context state (this tool *is* the action)
    ctx.context.user_model_state.current_topic_segment_index = segment_index + 1 # Progress index
    ctx.context.user_model_state.pending_interaction_type = None # Clear pending interaction
    ctx.context.user_model_state.pending_interaction_details = None
    ctx.context.last_interaction_summary = f"Presented explanation segment {segment_index} for {topic}."

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
    ctx: RunContextWrapper[TutorContext],
    question: QuizQuestion # Use the QuizQuestion model for structure
) -> QuestionResponse:
    """Asks the user a question to check their understanding of the current topic."""
    topic = ctx.context.current_teaching_topic or "Unknown Topic"
    interaction_id = f"chk_{uuid.uuid4().hex[:6]}"
    print(f"[TeacherTool] Asking checking question for '{topic}': {question.question[:50]}...")

    # Update context state to indicate we're waiting for an answer
    ctx.context.user_model_state.pending_interaction_type = 'checking_question'
    ctx.context.user_model_state.pending_interaction_details = {
        "interaction_id": interaction_id,
        "question": question.model_dump() # Store the question details
    }
    # Also store the question where the evaluator expects it
    ctx.context.current_quiz_question = question
    ctx.context.last_interaction_summary = f"Asked checking question on {topic}."

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
        # Define the input schema based on QuizQuestion (simplified for tool input)
        self.input_schema = QuizQuestion # Or a subset Pydantic model if preferred

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
                content=Content(role="tool", parts=[Part.from_text(f"Waiting for user answer to question: {question_obj.question[:30]}...")]), # Use imported Content/Part
                actions=EventActions(
                    # Use a custom action field to signal the pause and carry data
                    custom_action={
                        "type": "ask_question",
                        "data": question_obj.dict()
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
                content=Content( # Use imported Content/Part
                    role="tool",
                    parts=[Part.from_text(error_msg)]
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