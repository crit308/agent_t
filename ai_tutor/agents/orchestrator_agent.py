from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING, Union

from agents import Agent, Runner, RunConfig, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper

# Use TYPE_CHECKING for TutorContext import
if TYPE_CHECKING:
    from ai_tutor.context import TutorContext # Use the enhanced context
# Import tool functions (assuming they will exist in a separate file)
from ai_tutor.tools.orchestrator_tools import (
    call_quiz_creator_mini,
    call_quiz_teacher_evaluate,
    call_planner_get_next_topic,
    update_user_model,
    get_user_model_status,
    update_explanation_progress,
)

# Import models needed for type hints if tools return them
# Also import models needed for the output Union type
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, LessonContent
from ai_tutor.api_models import (
    TutorInteractionResponse, ExplanationResponse, QuestionResponse,
    FeedbackResponse, MessageResponse, ErrorResponse
)

def create_orchestrator_agent(api_key: str = None) -> Agent['TutorContext']:
    """Creates the Orchestrator Agent for the AI Tutor."""

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    provider: ModelProvider = OpenAIProvider()
    # Use the specified model version
    base_model = provider.get_model("gpt-4o-2024-08-06")  # Using exact model version

    orchestrator_tools = [
        call_quiz_creator_mini,
        call_quiz_teacher_evaluate,
        call_planner_get_next_topic,
        update_user_model,
        get_user_model_status,
        update_explanation_progress,
    ]

    orchestrator_agent = Agent['TutorContext'](
        name="TutorOrchestrator",
        instructions="""
        You are the conductor of an AI tutoring session. Your primary goal is to help the user learn the material effectively by guiding them through a lesson plan.

        CONTEXT:
        - You have access to the overall `LessonPlan` via the context object.
        - You can read and update the `UserModelState` via tools (`get_user_model_status`, `update_user_model`). This state tracks concept mastery, pace, style, notes, current topic, segment index, and pending interactions.
        - You know the user's last input/action provided in the prompt.
        - `tutor_context.user_model_state.pending_interaction_type` tells you if the Teacher agent is waiting for a user response (e.g., to a 'checking_question').

        CORE WORKFLOW:
        1.  **Assess State & Adapt:** Check user input, `UserModelState` (`get_user_model_status`), consider pace, style, confusion points, mastery levels.
        2.  **Handle User Input / Pending Interaction:**
            *   If `pending_interaction_type` is 'checking_question':
                - Use `call_quiz_teacher_evaluate` with the user's answer and the details stored in `pending_interaction_details`.
                - Based on the feedback (correct/incorrect): Decide next step (continue explanation, ask again, re-explain). Update mastery/confusion via `update_user_model`.
                - **Clear the pending state** in the context (this might need a dedicated tool or careful state management before finishing your turn).
            *   If user asked a question: Answer it briefly using a `MessageResponse` or decide to initiate a focused explanation segment (signal this).
            *   If user provided feedback or other input: Update `session_summary_notes` via `update_user_model`.
        3.  **Determine Next Step (if no pending interaction):**
            *   **Starting New Topic:**
                - Use `call_planner_get_next_topic` -> sets `tutor_context.current_teaching_topic` and resets segment index to 0.
                - Check Prerequisites using `get_user_model_status`. Consider optional content.
                - If prerequisites met for planned topic: Your next action is to **initiate teaching** for this topic (segment 0). Set `tutor_context.current_teaching_topic` to this topic. Indicate this decision clearly.
                - If prerequisites NOT met: Your next action is to **initiate teaching** for the *prerequisite* topic. Set `tutor_context.current_teaching_topic` to the prerequisite topic. Indicate this decision.
            *   **Continuing Topic:**
                - Check `tutor_context.user_model_state.current_topic_segment_index` and recent interaction outcomes.
                - If the last action was explaining and it was successful: Increment `tutor_context.user_model_state.current_topic_segment_index` (perhaps via `update_explanation_progress` or similar). Your next action is to **initiate teaching** for the new segment.
                - If the last action was a question and it was answered correctly: Decide whether to continue explanation (initiate teaching for next segment) or give a mini-quiz (`call_quiz_creator_mini`).
                - If the last action was a question and it was answered incorrectly: Decide whether to re-explain (initiate teaching for the *same* segment again, perhaps signalling the need for a different approach) or provide a hint (`MessageResponse`).
            *   **After Topic Complete (Teacher indicated is_last_segment=True in its last response):**
                - Update mastery via `update_user_model`.
                - Decide whether to give a mini-quiz (`call_quiz_creator_mini`) or move to the next topic (`call_planner_get_next_topic`).

        4.  **Select Action & Update State:**
            *   **Initiate Teaching:** If your decision is to explain/re-explain a segment, your final output **MUST** signal this. Use a specific `MessageResponse` like `{"response_type": "message", "text": "Okay, let's cover segment X of topic Y.", "message_type": "initiate_teaching"}`. Set the `tutor_context.current_teaching_topic` and ensure `current_topic_segment_index` is correct **before** returning this signal. The external loop running the agents will then invoke the `InteractiveLessonTeacher` with the updated context.
            *   **Quiz:** Use `call_quiz_creator_mini`. Your output *is* the tool's `QuestionResponse`.
            *   **Feedback:** Use `call_quiz_teacher_evaluate`. Your output *is* the tool's `FeedbackResponse`.
            *   **State Updates:** Use `update_user_model`, `update_explanation_progress`, `get_user_model_status` strategically *before* deciding the final action/signal.

        5.  **Formulate Response:** Your final output for this turn **MUST** be a JSON object matching one of the allowed types in the `TutorInteractionResponse` schema:
            *   If calling a tool like quiz creator/evaluator: The direct output of that tool (`QuestionResponse`, `FeedbackResponse`).
            *   If initiating teaching: A `MessageResponse` with `message_type='initiate_teaching'` as described above.
            *   If answering a direct user question or providing a simple message: A standard `MessageResponse`.
            *   If an error occurs: An `ErrorResponse`.

        PRINCIPLES:
        - Be Adaptive & Interactive.
        - Your primary role is **decision-making and state management**. You decide *what* should happen next (explain, quiz, evaluate, move on) and prepare the context for the next agent (often the Teacher).
        - Ensure your final output strictly adheres to the required JSON format (`TutorInteractionResponse`).
        """,
        tools=orchestrator_tools,
        output_type=TutorInteractionResponse,
        model=base_model,
    )

    return orchestrator_agent 