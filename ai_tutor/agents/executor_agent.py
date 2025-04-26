from __future__ import annotations
import logging
from typing import Optional, Union, Dict

from ai_tutor.exceptions import ExecutorError # Import from new file

from ai_tutor.context import TutorContext
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, FocusObjective # Added FocusObjective
from ai_tutor.api_models import (
    InteractionResponseData,
    ExplanationResponse,
    QuestionResponse,
    # QuizFeedbackItem, # Defined in agents.models
    MessageResponse,
    ErrorResponse,
)
from ai_tutor.context import UserModelState
from ai_tutor.core.llm import LLMClient
# Import necessary skill decorators and potentially the Runner if needed
# from ai_tutor.skills import skill # Example
# from agents.run_context import RunContextWrapper # Example if using ADK Runner
from ai_tutor.utils.tool_helpers import invoke 
# Corrected skill imports:
from ai_tutor.skills.explain_concept import explain_concept
from ai_tutor.skills.create_quiz import create_quiz

logger = logging.getLogger(__name__)

# Define the possible structured data payloads for InteractionResponseData
ResponseType = Union[ExplanationResponse, QuestionResponse, QuizFeedbackItem, MessageResponse, ErrorResponse]

# Corrected SYSTEM_PROMPT_TEMPLATE (now properly closed and clear about response_type)
SYSTEM_PROMPT_TEMPLATE = (
    "You are an AI Tutor. Your primary goal is to teach the user about the current Focus Objective.\n\n"
    "**Current Focus Objective:**\n"
    "Topic: {objective_topic}\n"
    "Learning Goal: {objective_goal}\n"
    "Target Mastery: {objective_mastery}\n\n"
    "**Current User State:**\n"
    "{user_model_state_summary}\n\n"
    "**Your Task:**\n"
    "1.  Read the User's Last Message (if any): \"{user_message}\"\n"
    "2.  Analyze the current objective, user state, and user message.\n"
    "3.  Decide the best next pedagogical step. Choose ONE appropriate skill to call:\n"
    "    *   `explain_concept(topic: str, details: str)`: To explain a new part of the topic or clarify something. Break explanations into manageable segments. Use `current_topic_segment_index` from context if continuing an explanation.\n"
    "    *   `create_quiz(topic: str, instructions: str)`: To create a single multiple-choice question to check understanding.\n"
    "    *   `evaluate_quiz(user_answer_index: int)`: To evaluate the user's answer to the *most recently asked question* (stored in context).\n"
    "    *   `remediate_concept(topic: str, remediation_details: str)`: To provide targeted help if the user is struggling or answered incorrectly.\n"
    "    *   `update_user_model(topic: str, outcome: str, details: Optional[str] = None)`: Call this *after* evaluating an answer or determining understanding/struggle from a user message. Use outcomes like 'correct', 'incorrect', 'unsure', 'clarification_needed'.\n"
    "    *   (If the user asks a direct question, prioritize answering it, potentially using `explain_concept` or generating a simple text response.)\n"
    "4.  Execute the chosen skill (internally, you don't show the call itself).\n"
    "5.  Format your final response as a single JSON object conforming EXACTLY to the `InteractionResponseData` schema. Do NOT add any text before or after the JSON.\n\n"
    "**InteractionResponseData Schema:**\n"
    "```json\n"
    "{{\n"
    "  \"content_type\": \"<type_string>\", // explanation, question, feedback, message, error\n"
    "  \"data\": {{ \"response_type\": \"<type_string>\", ... }}, // The specific Pydantic model matching content_type. MUST include 'response_type' field inside 'data'.\n"
    "  \"user_model_state\": {{ ... }} // The FULL, LATEST UserModelState object AFTER any updates.\n"
    "}}\n"
    "```\n\n"
    "**Workflow Logic:**\n"
    "*   If explaining, use `explain_concept`. Generate `ExplanationResponse` in `data` field (with `response_type: \'explanation\'`). Increment `current_topic_segment_index` in the returned `user_model_state`.\n"
    "*   **Handling 'Next': If the user input indicates they want to proceed (e.g., 'next', 'continue', or the special input '[NEXT]'), check the `current_topic_segment_index` in the `user_model_state`. If it's less than a threshold (e.g., 2), call `explain_concept` for the next segment. Otherwise, call `create_quiz` to check understanding.**\n"
    "*   If asking a question, use `create_quiz`. Generate `QuestionResponse` in `data` field (with `response_type: \'question\'`). **Crucially, BEFORE sending the response, store the generated `QuizQuestion` object in the `TutorContext.current_quiz_question` field.** Reset `current_topic_segment_index` in the returned `user_model_state`.\n"
    "*   If evaluating an answer, use `evaluate_quiz`. Then, use `update_user_model`. Generate `QuizFeedbackItem` in `data` field (with `response_type: \'feedback\'`).\n"
    "*   If user asks a question/clarification, use `explain_concept` or generate `MessageResponse` (with `response_type: \'message\'`). Use `update_user_model` if appropriate.\n"
    "*   Check for `objective_complete` condition after `update_user_model`. If met, you might inform the user with a `MessageResponse` before the planner takes over.\n"
    "\n"
    "**Important:** Think step-by-step internally to decide the best skill and parameters. Ensure the final output is ONLY the valid `InteractionResponseData` JSON, including the `response_type` inside the `data` field. **Always include the complete, updated `user_model_state` in your response.**"
)

def _get_user_model_state_summary(user_model_state: Optional[UserModelState]) -> str:
    """Generates a concise summary of the user model state for the prompt."""
    if not user_model_state or not user_model_state.concepts:
        return "User has no tracked concepts yet."
    
    state_items = []
    for topic, state in user_model_state.concepts.items():
        state_items.append(f"- {topic}: Mastery={state.mastery:.2f}, Confidence={state.confidence}, Attempts={state.attempts}")
    
    if not state_items:
         return "User has no tracked concepts yet."
         
    return "Current user concept understanding:\n" + "\n".join(state_items)

async def run_executor(ctx: TutorContext, user_input: Optional[str] = None) -> InteractionResponseData:
    """
    Runs the Executor Agent logic for one turn.

    Args:
        ctx: The current TutorContext, potentially modified by this function (e.g., current_quiz_question).
        user_input: The user's message from the frontend.

    Returns:
        An InteractionResponseData object containing the AI's response and updated state.
    """
    logger.info(f"Executor running for session {ctx.session_id}. User input: '{user_input}'")

    # Check if the objective exists and has a topic
    if not ctx.current_focus_objective or not ctx.current_focus_objective.topic:
        logger.error(f"Executor run failed: Missing or invalid focus objective in context for session {ctx.session_id}.")
        raise ExecutorError("Cannot proceed without a valid focus objective.") # Raise error to be handled by tutor_ws.py

    topic = ctx.current_focus_objective.topic # Safely access topic now

    if user_input == "[NEXT]":
        # --- Logic to decide next step based on segment_index ---
        current_segment_index = ctx.user_model_state.current_topic_segment_index
        max_segments = 3 # Or get from config/objective if available

        logger.info(f"Executor handling '[NEXT]': Current Segment Index = {current_segment_index}, Max Segments = {max_segments}, Topic = {topic}") # Log state

        # Condition to check if more explanation is needed vs. moving to quiz
        if current_segment_index < max_segments - 1: # e.g., 0, 1 for max_segments=3
            # Explain next segment
            next_segment_index = current_segment_index + 1
            logger.info(f"Executor: Calling explain_concept for segment {next_segment_index}")
            explanation_string = await invoke(
                explain_concept, # Function object itself
                ctx,             # RunContextWrapper
                topic=topic,     # Keyword arg for topic
                details=f"Provide explanation segment {next_segment_index} for {topic} based on learning goal: {ctx.current_focus_objective.learning_goal}."
            )
            # Update context segment index AFTER successful explanation generation
            ctx.user_model_state.current_topic_segment_index = next_segment_index

            # Determine if this NEW segment is the last one
            is_last = (next_segment_index >= max_segments - 1)

            explanation_payload = ExplanationResponse(
                response_type="explanation",
                text=f"Segment {next_segment_index}: {explanation_string}",
                topic=topic,
                segment_index=next_segment_index,
                is_last_segment=is_last
            )
            interaction_response = InteractionResponseData(
                content_type="explanation",
                data=explanation_payload,
                user_model_state=ctx.user_model_state
            )
            logger.info(f"Executor successfully generated explanation response for session {ctx.session_id}. Type: {interaction_response.content_type}")
            return interaction_response

        else:
            # Move to quiz
            logger.info("Executor: Entering quiz generation path.") # Log Entry
            logger.info(f"Executor: About to call create_quiz for topic '{topic}'.") # Log Before Skill Call
            quiz_question = await invoke(
                create_quiz, # Function object
                ctx,         # RunContextWrapper
                topic=topic, # Keyword arg for topic
                instructions=f"Generate a multiple-choice question about the main concepts of {topic}."
            )
            # Log After Skill Call
            logger.info(f"Executor: create_quiz returned type: {type(quiz_question)}")
            if quiz_question:
                 # Use model_dump_json for logging Pydantic models
                 logger.info(f"Executor: create_quiz returned question: {quiz_question.model_dump_json(indent=2) if isinstance(quiz_question, QuizQuestion) else quiz_question}")
            
            # Add an explicit check
            if not isinstance(quiz_question, QuizQuestion):
                logger.error("Executor: create_quiz did NOT return a valid QuizQuestion object!")
                raise ExecutorError("Invalid data received from create_quiz skill.")

            # Log Before Storing
            logger.info(f"Executor: Attempting to store question in context.")
            ctx.current_quiz_question = quiz_question
            logger.info(f"Executor: Stored quiz question in context: {quiz_question.question[:50]}...")

            # Log Before Payload Creation
            logger.info("Executor: Creating QuestionResponse payload.")
            question_payload = QuestionResponse(
                response_type="question",
                question=quiz_question,
                topic=topic
            )
            # Log Before Wrapper Creation
            logger.info("Executor: Wrapping in InteractionResponseData.")
            interaction_response = InteractionResponseData(
                content_type="question",
                data=question_payload,
                user_model_state=ctx.user_model_state
            )
            # Log Before Return
            logger.info(f"Executor: Returning InteractionResponseData with content_type='question'.")
            return interaction_response

    else: # Handle actual user text messages
        logger.warning(f"Executor: Received user text '{user_input}' - basic explanation logic executing.")
        # Fallback to first explanation for now if direct message received after planning
        explanation_string = await invoke(explain_concept, ctx, topic=topic, details=f"Explain the basics of {topic}.")
        ctx.user_model_state.current_topic_segment_index = 0 # Reset/set segment index
        explanation_payload = ExplanationResponse(
            response_type="explanation",
            text=explanation_string,
            topic=topic,
            segment_index=0,
            is_last_segment=False # Assume more follows
        )
        interaction_response = InteractionResponseData(
            content_type="explanation",
            data=explanation_payload,
            user_model_state=ctx.user_model_state
        )
        logger.info(f"Executor generated initial explanation response for session {ctx.session_id}. Type: {interaction_response.content_type}")
        return interaction_response

# Note: The actual skill implementations (explain_concept, create_quiz, etc.)
# need to be created in the skills/ directory as per the requirements document.
# The Executor LLM will conceptually "call" these, but in this direct LLM approach,
# the LLM generates the output *as if* it called the skill.
# If using an Agent framework (like ADK Runner), the LLM would output a tool call,
# the framework would execute the skill function, and return the result to the LLM.
# This implementation uses the direct LLM approach for simplicity based on the prompt design. 