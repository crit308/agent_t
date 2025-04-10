from agents import function_tool
from agents.run_context import RunContextWrapper
from typing import List, Dict, Any, Literal, cast
import uuid
from __future__ import annotations
import logging
from uuid import UUID

# Import necessary response models from api_models.py
from ai_tutor.api_models import ExplanationResponse, QuestionResponse, MessageResponse # Add others if needed
from ai_tutor.agents.models import QuizQuestion # If asking structured questions
from ai_tutor.context import TutorContext # Import TutorContext for type hint

from google.adk.tools import LongRunningFunctionTool, ToolContext
from google.generativeai import types as adk_types
from google.adk.events import Event, EventActions # Import Event classes

logger = logging.getLogger(__name__)

# Tool implementations now return the data structure the API needs

@function_tool
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

@function_tool
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
    def __init__(self, name: str = "ask_user_question_and_get_answer"):
        # The actual function logic is within run_async_stream
        super().__init__(func=self._placeholder_func_for_schema)
        self.name = name
        self.description = "Asks the user a multiple-choice question and waits for their answer index."
        self.input_schema = QuizQuestion # Use QuizQuestion model for input validation

    def _placeholder_func_for_schema(self, question: QuizQuestion) -> Dict[str, Any]:
        """Placeholder for schema generation. Actual logic is in run_async_stream."""
        pass

    async def run_async_stream(
        self, *, args: Dict[str, Any], tool_context: ToolContext
    ) -> AsyncGenerator[Event, None]:
        """
        The core logic: signal pause, wait for resume, return answer.
        This implementation yields a specific Event to signal the pause.
        """
        try:
            # Validate input using Pydantic
            question_obj = QuizQuestion.model_validate(args)
            logger.info(f"AskUserQuestionTool: Asking question - '{question_obj.question[:50]}...'")

            # Create and yield the pause event with question data
            pause_event = Event(
                author=tool_context.agent_name,
                content=adk_types.Content(
                    role="tool",
                    parts=[adk_types.Part(text=f"Waiting for user answer to question about: {question_obj.related_section}")]
                ),
                actions=EventActions(
                    custom_action={
                        "signal": "request_user_input",
                        "tool_call_id": tool_context.function_call_id,
                        "question_data": question_obj.model_dump()
                    }
                ),
                invocation_id=tool_context.invocation_id
            )
            yield pause_event
            logger.info(f"AskUserQuestionTool: Yielded pause signal event for tool_call_id {tool_context.function_call_id}.")

            # Execution pauses here until the Runner receives a FunctionResponse event
            # with matching tool_call_id and feeds it back into this session's execution

        except Exception as e:
            error_msg = f"Error in AskUserQuestionTool: {e}"
            logger.exception(error_msg)
            # Yield an error event
            error_event = Event(
                author=tool_context.agent_name,
                content=adk_types.Content(
                    role="tool",
                    parts=[adk_types.Part(text=error_msg)]
                ),
                actions=EventActions(
                    custom_action={"error": error_msg}
                ),
                invocation_id=tool_context.invocation_id
            )
            yield error_event

# Instantiate the tool for export
ask_user_question_and_get_answer_tool = AskUserQuestionTool() 