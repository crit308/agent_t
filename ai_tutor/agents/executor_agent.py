from __future__ import annotations
import logging
from typing import Optional, Union, Dict, Tuple, List

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
    FeedbackResponse,
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
# Need imports for evaluate_quiz and update_user_model if they exist
from ai_tutor.skills.evaluate_quiz import evaluate_quiz
from ai_tutor.skills.update_user_model import update_user_model
from ai_tutor.skills.draw_mcq import draw_mcq_actions

logger = logging.getLogger(__name__)

# Define the possible structured data payloads for InteractionResponseData
ResponseType = Union[ExplanationResponse, QuestionResponse, QuizFeedbackItem, MessageResponse, ErrorResponse]

# Define JSON examples separately to avoid escaping issues
EXAMPLE_RESPONSE_WITH_WHITEBOARD = '''{
  "content_type": "explanation",
  "data": {
    "response_type": "explanation",
    "text": "Evaporation is when water turns to vapor and rises.",
    "topic": "Evaporation",
    "segment_index": 0,
    "is_last_segment": false
  },
  "user_model_state": { ... },
  "whiteboard_actions": [
    { "id": "evap-label-1", "kind": "text", "x": 600, "y": 100, "text": "Evaporation", "fontSize": 18, "metadata": {"source": "assistant"} },
    { "id": "arrow-up-1", "kind": "line", "points": [650, 150, 650, 130], "stroke": "#000000", "strokeWidth": 2, "metadata": {"source": "assistant"} }
  ]
}'''

EXAMPLE_RESPONSE_WITHOUT_WHITEBOARD = '''{
  "content_type": "question",
  "data": {
    "response_type": "question",
    "question": { "question": "What is the first step...?", "options": [...], "correct_index": 0 }, 
    "topic": "Some Topic"
  },
  "user_model_state": { ... }
}'''

SYSTEM_PROMPT_TEMPLATE = """
You are an AI Tutor. Your primary goal is to teach the user about the current Focus Objective.

**Current Focus Objective:**
Topic: {objective_topic}
Learning Goal: {objective_goal}
Target Mastery: {objective_mastery}

**Current User State:**
{user_model_state_summary}

**Your Task:**
1.  Read the User's Last Message (if any): "{user_message}"
2.  Analyze the current objective, user state, and user message.
3.  Decide the best next pedagogical step. Choose ONE appropriate skill to call (or respond directly if needed):
    *   `explain_concept(topic: str, details: str)`: To explain a new part of the topic or clarify something. Break explanations into manageable segments. Use `current_topic_segment_index` from context if continuing an explanation.
    *   `create_quiz(topic: str, instructions: str)`: To create a single multiple-choice question to check understanding.
    *   `evaluate_quiz(user_answer_index: int)`: To evaluate the user's answer to the *most recently asked question*.
    *   `remediate_concept(topic: str, remediation_details: str)`: To provide targeted help if the user is struggling.
    *   `update_user_model(topic: str, outcome: str, details: Optional[str] = None)`: Call this *after* evaluating an answer or determining understanding/struggle. Use outcomes like 'correct', 'incorrect', 'unsure', 'clarification_needed'.
    *   (If the user asks a direct question, prioritize answering it.)
4.  Execute the chosen skill (internally). YOU DO NOT SHOW THE SKILL CALL ITSELF.
5.  Format your final response as a single JSON object conforming EXACTLY to the `InteractionResponseData` schema below. Do NOT add any text before or after the JSON.

**InteractionResponseData Schema:**
```json
{{
  "content_type": "<type_string>", // E.g., explanation, question, feedback, message, error
  "data": {{ "response_type": "<type_string>", ... }}, // The specific Pydantic model matching content_type. MUST include 'response_type' field inside 'data'.
  "user_model_state": {{ ... }}, // The FULL, LATEST UserModelState object AFTER any updates.
  "whiteboard_actions": Optional[List[CanvasObjectSpec]] // Optional: Only include if you need to draw on the whiteboard.
}}
```

**Whiteboard Actions (Optional, but Strongly Encouraged When Helpful):**
*   If a visual explanation, diagram, or multiple-choice question (MCQ) presentation would help the user, you MUST include the `whiteboard_actions` key in your JSON response. Otherwise, omit this key entirely.
*   The `whiteboard_actions` value MUST be a list of `CanvasObjectSpec` objects, each describing a shape, text, or group to be drawn.
*   Use whiteboard actions for:
    *   Visual explanations (e.g., diagrams of cycles, labeled parts, arrows showing processes)
    *   MCQ presentations (draw the question, each option as a group with a radio button and label)
    *   Any time a drawing would clarify or reinforce the concept
*   Omit `whiteboard_actions` if a drawing would not add value.
*   **Constraints:**
    *   Keep drawings simple and clear. Avoid clutter.
    *   Use unique, descriptive IDs for each object (e.g., "mcq-q1-opt-0-radio").
    *   Always include correct metadata: `{ "source": "assistant" }` and, for MCQs, add `role`, `question_id`, and `option_id` as appropriate.
    *   Strictly follow the CanvasObjectSpec format below.

*   `CanvasObjectSpec` Schema (Example):
    ```json
    {{
      "id": "string", // Required. Unique, simple ID (e.g., "text-1", "rect-0", "mcq-q1-opt-0-radio")
      "kind": "string", // Required. Type of object (text, rect, circle, line, path, image)
      "x": number, // Required. X coordinate (top-left for rect/text, center for circle)
      "y": number, // Required. Y coordinate
      "text": Optional[string], // For kind='text'
      "fill": Optional[string], // Optional fill color (e.g., '#FFFFFF')
      "stroke": Optional[string], // Optional stroke color (e.g., '#000000')
      "strokeWidth": Optional[number], // Optional stroke width
      "width": Optional[number], // For kind='rect' or 'image'
      "height": Optional[number], // For kind='rect' or 'image'
      "radius": Optional[number], // For kind='circle'
      "points": Optional[List[number]], // For kind='line' or 'path' (e.g., [x1, y1, x2, y2, ...])
      "fontSize": Optional[number], // For kind='text'
      "metadata": { "source": "assistant", ... } // Required metadata. For MCQs, add role, question_id, option_id.
    }}
    ```

**Detailed Example: Visual Diagram (Water Cycle)**
```json
{
  "content_type": "explanation",
  "data": {
    "response_type": "explanation",
    "text": "Evaporation is when water turns to vapor and rises.",
    "topic": "Evaporation",
    "segment_index": 0,
    "is_last_segment": false
  },
  "user_model_state": { ... },
  "whiteboard_actions": [
    { "id": "evap-label-1", "kind": "text", "x": 600, "y": 100, "text": "Evaporation", "fontSize": 18, "metadata": {"source": "assistant", "role": "label"} },
    { "id": "arrow-up-1", "kind": "line", "points": [650, 150, 650, 130], "stroke": "#000000", "strokeWidth": 2, "metadata": {"source": "assistant", "role": "arrow"} }
  ]
}
```

**Detailed Example: MCQ Presentation**
```json
{
  "content_type": "question",
  "data": {
    "response_type": "question",
    "question": {
      "question": "What is the primary mechanism by which water returns to the atmosphere in the water cycle?",
      "options": ["Condensation", "Evaporation", "Precipitation", "Transpiration"],
      "correct_index": 1
    },
    "topic": "Water Cycle"
  },
  "user_model_state": { ... },
  "whiteboard_actions": [
    { "id": "mcq-q1-text", "kind": "text", "x": 50, "y": 50, "text": "What is the primary mechanism by which water returns to the atmosphere in the water cycle?", "fontSize": 18, "width": 700, "metadata": {"source": "assistant", "role": "question", "question_id": "q1"} },
    { "id": "mcq-q1-opt-0-radio", "kind": "circle", "x": 70, "y": 108, "radius": 8, "stroke": "#555555", "strokeWidth": 1, "fill": "#FFFFFF", "metadata": {"source": "assistant", "role": "option_selector", "question_id": "q1", "option_id": 0} },
    { "id": "mcq-q1-opt-0-text", "kind": "text", "x": 95, "y": 108, "text": "A. Condensation", "fontSize": 16, "fill": "#333333", "metadata": {"source": "assistant", "role": "option_label", "question_id": "q1", "option_id": 0} },
    { "id": "mcq-q1-opt-1-radio", "kind": "circle", "x": 70, "y": 148, "radius": 8, "stroke": "#555555", "strokeWidth": 1, "fill": "#FFFFFF", "metadata": {"source": "assistant", "role": "option_selector", "question_id": "q1", "option_id": 1} },
    { "id": "mcq-q1-opt-1-text", "kind": "text", "x": 95, "y": 148, "text": "B. Evaporation", "fontSize": 16, "fill": "#333333", "metadata": {"source": "assistant", "role": "option_label", "question_id": "q1", "option_id": 1} },
    { "id": "mcq-q1-opt-2-radio", "kind": "circle", "x": 70, "y": 188, "radius": 8, "stroke": "#555555", "strokeWidth": 1, "fill": "#FFFFFF", "metadata": {"source": "assistant", "role": "option_selector", "question_id": "q1", "option_id": 2} },
    { "id": "mcq-q1-opt-2-text", "kind": "text", "x": 95, "y": 188, "text": "C. Precipitation", "fontSize": 16, "fill": "#333333", "metadata": {"source": "assistant", "role": "option_label", "question_id": "q1", "option_id": 2} },
    { "id": "mcq-q1-opt-3-radio", "kind": "circle", "x": 70, "y": 228, "radius": 8, "stroke": "#555555", "strokeWidth": 1, "fill": "#FFFFFF", "metadata": {"source": "assistant", "role": "option_selector", "question_id": "q1", "option_id": 3} },
    { "id": "mcq-q1-opt-3-text", "kind": "text", "x": 95, "y": 228, "text": "D. Transpiration", "fontSize": 16, "fill": "#333333", "metadata": {"source": "assistant", "role": "option_label", "question_id": "q1", "option_id": 3} }
  ]
}
```

**Workflow Logic:**
*   If explaining, use `explain_concept`. Generate `ExplanationResponse` in `data`. You *might* include `whiteboard_actions` if a visual helps. Increment `current_topic_segment_index` in `user_model_state`.
*   **Handling 'Next':** If user wants to proceed, check `current_topic_segment_index`. If more segments exist, call `explain_concept`. Otherwise, call `create_quiz`.
*   If asking a question, use `create_quiz`. Generate `QuestionResponse` in `data`. Store the question in `TutorContext.current_quiz_question`. Reset `current_topic_segment_index` in `user_model_state`. You *might* include `whiteboard_actions` if it helps illustrate the question.
*   If evaluating an answer, use `evaluate_quiz`, then `update_user_model`. Generate `FeedbackResponse` (containing `QuizFeedbackItem`) in `data`. Generally, no whiteboard actions needed here.
*   If user asks a question/clarification, use `explain_concept` or generate `MessageResponse` in `data`. Use `update_user_model` if appropriate. You *might* include `whiteboard_actions` for the explanation.
*   Check for `objective_complete` after `update_user_model`. If met, you might inform the user with a `MessageResponse`.

**Example Response with Whiteboard Actions:**
```json
{EXAMPLE_RESPONSE_WITH_WHITEBOARD}
```

**Example Response without Whiteboard Actions:**
```json
{EXAMPLE_RESPONSE_WITHOUT_WHITEBOARD}
```

**Important:** Think step-by-step. Ensure the final output is ONLY the valid `InteractionResponseData` JSON. **Always include the complete, updated `user_model_state`.** Only include `whiteboard_actions` when you intend to draw.
"""

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

# Status constants
STATUS_AWAITING_INPUT = "awaiting_user_input"
STATUS_OBJECTIVE_COMPLETE = "objective_complete"

async def run_executor(ctx: TutorContext, user_input: Optional[str], event_type: str, event_data: Optional[Dict] = None) -> Tuple[InteractionResponseData, str]:
    """
    Runs the Executor Agent logic for one turn based on the event type.

    Args:
        ctx: The current TutorContext, potentially modified by this function.
        user_input: The user's message from the frontend (for 'user_message').
        event_type: The type of event ('user_message', 'next', 'answer').
        event_data: Additional data associated with the event (e.g., {'answer_index': X} for 'answer').

    Returns:
        A tuple containing:
            - An InteractionResponseData object with the AI's response and updated state.
            - A status string ('awaiting_user_input' or 'objective_complete').
    """
    logger.info(f"Executor running for session {ctx.session_id}. Event Type: '{event_type}', User input: '{user_input}', Event Data: {event_data}")

    if not ctx.current_focus_objective or not ctx.current_focus_objective.topic:
        logger.error(f"Executor run failed: Missing or invalid focus objective in context for session {ctx.session_id}.")
        # Return an ErrorResponse and keep awaiting input
        error_payload = ErrorResponse(response_type="error", message="Cannot proceed without a valid focus objective.")
        interaction_response = InteractionResponseData(content_type="error", data=error_payload, user_model_state=ctx.user_model_state)
        logger.info(f"Executor returning early due to missing objective. Status: {STATUS_AWAITING_INPUT}")
        return interaction_response, STATUS_AWAITING_INPUT # Return status

    topic = ctx.current_focus_objective.topic
    response_to_send: InteractionResponseData
    status: str = STATUS_AWAITING_INPUT # Default status

    # --- Event Type Handling ---
    if event_type == 'next':
        # Logic moved from the old "[NEXT]" handling
        current_segment_index = ctx.user_model_state.current_topic_segment_index
        # TODO: Make max_segments configurable, perhaps from focus objective or KB analysis?
        max_segments = 3 # Hardcoded segment limit as per plan

        logger.info(f"Executor handling 'next': Current Segment Index = {current_segment_index}, Max Segments = {max_segments}, Topic = {topic}")

        if current_segment_index < max_segments - 1: # More segments to explain
            next_segment_index = current_segment_index + 1
            logger.info(f"Executor: Calling explain_concept for segment {next_segment_index}")
            explanation_string = await invoke(
                explain_concept,
                ctx,
                topic=topic,
                details=f"Provide explanation segment {next_segment_index} for {topic} based on learning goal: {ctx.current_focus_objective.learning_goal}."
            )
            ctx.user_model_state.current_topic_segment_index = next_segment_index
            is_last = (next_segment_index >= max_segments - 1)

            explanation_payload = ExplanationResponse(
                response_type="explanation",
                text=explanation_string,
                topic=topic,
                segment_index=next_segment_index,
                is_last_segment=is_last
            )
            response_to_send = InteractionResponseData(
                content_type="explanation",
                data=explanation_payload,
                user_model_state=ctx.user_model_state
            )
            logger.info(f"Executor generated explanation response. Segment {next_segment_index}/{max_segments-1}. Is Last: {is_last}. Status: {STATUS_AWAITING_INPUT}")
            status = STATUS_AWAITING_INPUT

        else: # Last segment explained, move to quiz
            logger.info("Executor: Calling create_quiz")
            quiz_question = await invoke(
                create_quiz,
                ctx,
                topic=topic,
                instructions=f"Generate a multiple-choice question about the main concepts of {topic} covered so far."
            )
            if not isinstance(quiz_question, QuizQuestion):
                logger.error("Executor: create_quiz did NOT return a valid QuizQuestion object!")
                 # Return an ErrorResponse
                error_payload = ErrorResponse(response_type="error", message="Failed to create a quiz question.")
                response_to_send = InteractionResponseData(content_type="error", data=error_payload, user_model_state=ctx.user_model_state)
                logger.warning(f"Executor failed to create quiz. Status: {STATUS_AWAITING_INPUT}")
                status = STATUS_AWAITING_INPUT # Keep waiting, maybe try again later?
            else:
                logger.info(f"Executor: Storing quiz question in context: {quiz_question.question[:50]}...")
                ctx.current_quiz_question = quiz_question
                # Draw MCQ actions
                actions = await invoke(draw_mcq_actions, ctx, question=quiz_question)
                question_payload = QuestionResponse(
                    response_type="question",
                    question=quiz_question,
                    topic=topic
                )
                response_to_send = InteractionResponseData(
                    content_type="question",
                    data=question_payload,
                    user_model_state=ctx.user_model_state,
                    whiteboard_actions=actions
                )
                logger.info(f"Executor generated question response. Status: {STATUS_AWAITING_INPUT}")
                status = STATUS_AWAITING_INPUT

    elif event_type == 'answer':
        if not ctx.current_quiz_question:
            logger.warning("Executor received 'answer' event but no question in context.")
            error_payload = ErrorResponse(response_type="error", message="I wasn't expecting an answer right now. Did I ask a question?")
            response_to_send = InteractionResponseData(content_type="error", data=error_payload, user_model_state=ctx.user_model_state)
            logger.warning(f"Executor received answer when no question active. Status: {STATUS_AWAITING_INPUT}")
            status = STATUS_AWAITING_INPUT
        elif not event_data or 'answer_index' not in event_data:
            logger.error("Executor received 'answer' event but 'answer_index' is missing in event_data.")
            error_payload = ErrorResponse(response_type="error", message="Something went wrong. I didn't receive your answer index correctly.")
            response_to_send = InteractionResponseData(content_type="error", data=error_payload, user_model_state=ctx.user_model_state)
            logger.error(f"Executor missing answer_index. Status: {STATUS_AWAITING_INPUT}")
            status = STATUS_AWAITING_INPUT
        else:
            answer_index = event_data['answer_index']
            logger.info(f"Executor handling 'answer': Index = {answer_index}. Evaluating question: {ctx.current_quiz_question.question[:50]}...")

            # Call evaluate_quiz
            feedback: Optional[QuizFeedbackItem] = await invoke(
                evaluate_quiz,
                ctx,
                user_answer_index=answer_index
                # evaluate_quiz should internally get the question from ctx.current_quiz_question
            )

            if not feedback:
                 logger.error("Executor: evaluate_quiz did not return feedback.")
                 error_payload = ErrorResponse(response_type="error", message="Sorry, I couldn't evaluate your answer.")
                 response_to_send = InteractionResponseData(content_type="error", data=error_payload, user_model_state=ctx.user_model_state)
                 logger.error(f"Executor evaluate_quiz failed. Status: {STATUS_AWAITING_INPUT}")
                 status = STATUS_AWAITING_INPUT
            else:
                logger.info(f"Executor: evaluate_quiz returned. Correct: {feedback.is_correct}")
                # Call update_user_model based on feedback
                outcome = 'correct' if feedback.is_correct else 'incorrect'
                logger.info(f"Executor: Calling update_user_model with topic='{topic}', outcome='{outcome}'")
                updated_model_state: Optional[UserModelState] = await invoke(
                    update_user_model,
                    ctx,
                    topic=topic,
                    outcome=outcome,
                    details=f"Answered question on {topic}. Correct: {feedback.is_correct}."
                    # update_user_model should update ctx.user_model_state internally
                )

                # --- FIX: Wrap feedback in FeedbackResponse --- 
                feedback_payload = FeedbackResponse(
                    response_type="feedback", # Ensure inner type is set
                    item=feedback # Nest the actual feedback item
                )
                # --- END FIX ---

                if not updated_model_state:
                    logger.error("Executor: update_user_model did not return an updated state.")
                    # Proceed with feedback but log error
                    response_to_send = InteractionResponseData(
                        content_type="feedback",
                        data=feedback_payload, # Pass the wrapped payload
                        user_model_state=ctx.user_model_state # Return old state
                    )
                    logger.error(f"Executor update_user_model failed. Proceeding with feedback. Status: {status}")
                else:
                     # Ensure the context reflects the updated state returned by the skill
                     ctx.user_model_state = updated_model_state
                     logger.info(f"Executor: User model updated. New mastery for {topic}: {ctx.user_model_state.concepts.get(topic).mastery if ctx.user_model_state.concepts.get(topic) else 'N/A'}")
                     response_to_send = InteractionResponseData(
                        content_type="feedback",
                        data=feedback_payload, # Pass the wrapped payload
                        user_model_state=ctx.user_model_state # Use updated state
                    )

                # Clear the question now that it's answered and evaluated
                logger.info("Executor: Clearing current_quiz_question from context.")
                ctx.current_quiz_question = None

                # Check if objective is complete (Example condition: mastery >= target)
                target_mastery = ctx.current_focus_objective.target_mastery
                current_mastery = ctx.user_model_state.concepts.get(topic).mastery if ctx.user_model_state.concepts.get(topic) else 0.0
                if current_mastery >= target_mastery:
                    logger.info(f"Executor: Objective '{topic}' complete! Mastery {current_mastery:.2f} >= Target {target_mastery:.2f}")
                    status = STATUS_OBJECTIVE_COMPLETE
                    # Optionally, modify the feedback response to indicate completion?
                    # Or let the WebSocket handler manage the transition message.
                    logger.info(f"Executor set status: {STATUS_OBJECTIVE_COMPLETE}")
                else:
                    logger.info(f"Executor: Objective '{topic}' not yet complete. Mastery {current_mastery:.2f} < Target {target_mastery:.2f}")
                    logger.info(f"Executor set status: {STATUS_AWAITING_INPUT}")
                    status = STATUS_AWAITING_INPUT


    elif event_type == 'user_message':
        logger.info(f"Executor handling 'user_message': '{user_input}'")
        # Determine topic reference (fallback if no focus)
        topic = ctx.current_focus_objective.topic if ctx.current_focus_objective else "General Chat"
        user_lower = (user_input or "").lower()
        wants_quiz = any(k in user_lower for k in ["quiz", "question", "test", "mcq"])
        wants_explain = any(k in user_lower for k in ["explain", "clarify", "help"])

        if wants_quiz:
            # --- User explicitly asked for a question --- #
            logger.info("Executor: Detected request for quiz question. Calling create_quiz.")
            quiz_question = await invoke(
                create_quiz,
                ctx,
                topic=topic,
                instructions=f"Generate a short multiple-choice question about {topic} that checks understanding of the main idea."
            )
            if not isinstance(quiz_question, QuizQuestion):
                logger.error("Executor: create_quiz did NOT return a valid QuizQuestion object when requested by user.")
                error_payload = ErrorResponse(response_type="error", message="Failed to create a quiz question.")
                response_to_send = InteractionResponseData(content_type="error", data=error_payload, user_model_state=ctx.user_model_state)
                status = STATUS_AWAITING_INPUT
            else:
                logger.info("Executor: Successfully generated quiz via user request. Creating whiteboard actions.")
                actions = await invoke(draw_mcq_actions, ctx, question=quiz_question)
                ctx.current_quiz_question = quiz_question
                question_payload = QuestionResponse(response_type="question", question=quiz_question, topic=topic)
                response_to_send = InteractionResponseData(
                    content_type="question",
                    data=question_payload,
                    user_model_state=ctx.user_model_state,
                    whiteboard_actions=actions
                )
                status = STATUS_AWAITING_INPUT
        else:
            # Existing behaviour: explanation or dialogue
            if wants_explain:
                logger.info("Executor: Detected clarification request. Using explain_concept.")
                skill_details = f"Provide a concise clarification regarding: '{user_input}'."
            else:
                logger.info("Executor: Treating as general dialogue/explanation.")
                skill_details = f"Respond conversationally to the user message: '{user_input}' regarding {topic}."

            response_text = await invoke(
                explain_concept,
                ctx,
                topic=topic,
                details=skill_details
            )
            message_payload = MessageResponse(response_type="message", text=response_text or "Got it.")
            response_to_send = InteractionResponseData(content_type="message", data=message_payload, user_model_state=ctx.user_model_state)
            status = STATUS_AWAITING_INPUT

    else:
        logger.error(f"Executor received unknown event_type: '{event_type}'")
        error_payload = ErrorResponse(response_type="error", message=f"Received unknown event type '{event_type}'.")
        response_to_send = InteractionResponseData(content_type="error", data=error_payload, user_model_state=ctx.user_model_state)
        logger.error(f"Executor unknown event type. Status: {STATUS_AWAITING_INPUT}")
        status = STATUS_AWAITING_INPUT


    # Log final decision before returning
    logger.info(f"Executor finished turn. Response Type: {response_to_send.content_type}, Status: {status}")
    # Ensure response includes the latest state AFTER any updates from skills
    response_to_send.user_model_state = ctx.user_model_state
    logger.info(f"Executor returning: response_type={response_to_send.content_type}, status={status}")
    return response_to_send, status


# --- Removed old run_executor implementation ---


# Note: The actual skill implementations (explain_concept, create_quiz, evaluate_quiz, update_user_model)
# need to be implemented correctly in the skills/ directory and registered for invoke() to work.
# This refactored executor RELIES on those skills performing their described actions,
# including potentially modifying the ctx or returning specific data structures.
# Example: update_user_model is expected to modify ctx.user_model_state AND return the modified state.
# Example: evaluate_quiz is expected to use ctx.current_quiz_question and return QuizFeedbackItem.
# Example: create_quiz returns QuizQuestion, which is then stored in ctx.current_quiz_question.
# Example: explain_concept returns a string, index is updated in ctx.user_model_state here.

def extract_whiteboard_actions(response_json: dict, logger: logging.Logger = logger):
    """
    Extract and validate the optional whiteboard_actions field from a parsed LLM JSON response.
    Returns a list if valid, or None. Logs a warning if present but not a list.
    """
    actions = response_json.get("whiteboard_actions")
    if actions is not None:
        if isinstance(actions, list):
            return actions
        else:
            logger.warning(f"LLM returned non-list for whiteboard_actions: {type(actions)}. Ignoring.")
    return None

# Example usage in LLM integration (pseudo-code):
# response_json = ... # parsed LLM output
# validated_actions = extract_whiteboard_actions(response_json)
# interaction_response = InteractionResponseData(
#     content_type=..., data=..., user_model_state=..., whiteboard_actions=validated_actions
# )