from __future__ import annotations
import logging
from typing import Optional, Union, Dict, Tuple, List, Any

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
from ai_tutor.skills.draw_mcq_feedback import draw_mcq_feedback
from ai_tutor.skills.draw_diagram import draw_diagram_actions  # Import diagram skill
from ai_tutor.skills.clear_whiteboard import clear_whiteboard  # Import clear skill

# NEW low-level drawing helpers
from ai_tutor.skills.drawing_tools import draw_text, draw_shape, style_token, clear_board

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
    *   `explain_concept(topic: str, details: str)`: To explain a new part of the topic or clarify something. This skill may itself return whiteboard_actions for diagrams or labels.
    *   `create_quiz(topic: str, instructions: str)`: To create a single multiple-choice question to check understanding. This skill may itself return whiteboard_actions to visually present the MCQ.
    *   `evaluate_quiz(user_answer_index: int)`: To evaluate the user's answer to the *most recently asked question*. This skill may itself return whiteboard_actions for feedback.
    *   `remediate_concept(topic: str, remediation_details: str)`: To provide targeted help if the user is struggling.
    *   `update_user_model(topic: str, outcome: str, details: Optional[str] = None)`: Call this *after* evaluating an answer or determining understanding/struggle. Use outcomes like 'correct', 'incorrect', 'unsure', 'clarification_needed'.
    *   `draw_text(id: str, text: str, x?: int, y?: int, fontSize?: int, width?: int, color_token?: str)`: Draw a textbox at (x,y). You can call this directly to add a label or annotation.
    *   `draw_shape(id: str, kind: "rect"|"circle"|"arrow", x?: int, y?: int, w?: int, h?: int, radius?: int, points?: List[Dict], label?: str, color_token?: str)`: Draw a basic shape. You can call this directly for visual emphasis or to illustrate a concept.
    *   `style_token(token: "default"|"primary"|"accent"|"muted"|"success"|"error")`: Resolve a semantic colour token to a hex colour string.
    *   `clear_board()`: Clear previous assistant drawings.
    *   `draw_diagram_actions(topic: str, description: str)`: Produce a simple diagram when helpful.
    *   `clear_whiteboard()`: (Alias of clear_board) Emit a reset action list to clear old visuals.
    *   `draw_mcq_feedback(question_id: str, option_id: int, is_correct: bool)`: After evaluating an MCQ answer, draw ✓/✗ and recolour the selected option. Also locks further clicks.
    *   `draw_table_actions(headers: List[str], rows: List[List[str]])`: Render small comparison or summary tables when tabular presentation aids understanding.
    *   `draw_flowchart_actions(steps: List[str])`: Render a simple left-to-right flowchart to outline processes.
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
    *   Always include correct metadata: `{ "source": "assistant" }` and, for MCQs, add role, question_id, and option_id as appropriate.
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

**Additional Examples (new skills)**

*Table Example* – using `draw_table_actions`:
```json
{
  "content_type": "explanation",
  "data": {
    "response_type": "explanation",
    "text": "Here's a quick comparison of prokaryotic vs eukaryotic cells.",
    "topic": "Cell Types",
    "segment_index": 1,
    "is_last_segment": false
  },
  "user_model_state": { ... },
  "whiteboard_actions": [
    { "id": "table-1-header-0", "kind": "rect", "x": 50, "y": 50, "width": 140, "height": 40, "fill": "#BBDEFB", "stroke": "#0D47A1", "strokeWidth": 1, "metadata": {"source":"assistant","role":"table_header","table_id":"table-1","col":0}},
    { "id": "table-1-header-0-text", "kind": "text", "x": 60, "y": 70, "text": "Feature", "fontSize": 14, "fill": "#0D47A1", "metadata": {"source":"assistant","role":"table_header_text","table_id":"table-1","col":0}},
    { "id": "table-1-header-1", "kind": "rect", "x": 200, "y": 50, "width": 140, "height": 40, "fill": "#BBDEFB", "stroke": "#0D47A1", "strokeWidth": 1, "metadata": {"source":"assistant","role":"table_header","table_id":"table-1","col":1}},
    { "id": "table-1-header-1-text", "kind": "text", "x": 210, "y": 70, "text": "Prokaryotic", "fontSize": 14, "fill": "#0D47A1", "metadata": {"source":"assistant","role":"table_header_text","table_id":"table-1","col":1}},
    { "id": "table-1-header-2", "kind": "rect", "x": 350, "y": 50, "width": 140, "height": 40, "fill": "#BBDEFB", "stroke": "#0D47A1", "strokeWidth": 1, "metadata": {"source":"assistant","role":"table_header","table_id":"table-1","col":2}},
    { "id": "table-1-header-2-text", "kind": "text", "x": 360, "y": 70, "text": "Eukaryotic", "fontSize": 14, "fill": "#0D47A1", "metadata": {"source":"assistant","role":"table_header_text","table_id":"table-1","col":2}},
    // ... additional body cells ...
  ]
}
```

*Flowchart Example* – using `draw_flowchart_actions`:
```json
{
  "content_type": "explanation",
  "data": {
    "response_type": "explanation",
    "text": "Follow these steps for the water cycle.",
    "topic": "Water Cycle",
    "segment_index": 2,
    "is_last_segment": false
  },
  "user_model_state": { ... },
  "whiteboard_actions": [
    { "id": "flow-1-box-0", "kind": "rect", "x": 50, "y": 50, "width": 140, "height": 60, "fill": "#E8F5E9", "stroke": "#1B5E20", "strokeWidth": 1, "metadata": {"source":"assistant","role":"flow_box","chart_id":"flow-1","step":0}},
    { "id": "flow-1-box-0-text", "kind": "text", "x": 120, "y": 80, "text": "Evaporation", "fontSize": 14, "fill": "#1B5E20", "textAnchor":"middle", "metadata": {"source":"assistant","role":"flow_box_text","chart_id":"flow-1","step":0}},
    { "id": "flow-1-arrow-0-1", "kind": "line", "points": [190, 80, 260, 80], "stroke": "#000000", "strokeWidth": 2, "metadata": {"source":"assistant","role":"flow_arrow","chart_id":"flow-1","from":0,"to":1}},
    // ... next boxes & arrows ...
  ]
}
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

async def run_executor(ctx: TutorContext, user_input: Optional[str], event_type: str, event_data: Optional[Dict] = None) -> Tuple[dict, str]:
    """
    Refactored Executor: Uses LLM to select next skill, its parameters, and status.
    Returns a dict: {"skill_name": ..., "skill_params": {...}, "status": ...}
    """
    logger.info(f"[LLM-Driven Executor] Running for session {ctx.session_id}. Event Type: '{event_type}', User input: '{user_input}', Event Data: {event_data}")

    if not ctx.current_focus_objective or not ctx.current_focus_objective.topic:
        logger.error(f"Executor run failed: Missing or invalid focus objective in context for session {ctx.session_id}.")
        return {"skill_name": None, "skill_params": {}, "status": "awaiting_user_input", "error": "No focus objective."}, "awaiting_user_input"

    # Prepare context for LLM
    user_model_state_summary = _get_user_model_state_summary(ctx.user_model_state)
    objective = ctx.current_focus_objective
    system_prompt = f"""
You are an AI Tutor Executor. Your job is to decide the next pedagogical action by selecting a skill and its parameters.

Current Focus Objective:
- Topic: {objective.topic}
- Learning Goal: {objective.learning_goal}
- Target Mastery: {objective.target_mastery}

Current User State:
{user_model_state_summary}

User's Last Message: {user_input or ''}

Event Type: {event_type}
Event Data: {event_data or {}}

Choose ONE skill to call next, and provide its parameters. Also, return the interaction status (awaiting_user_input or objective_complete).
Respond ONLY with a JSON object like:
{{
  "skill_name": "...",
  "skill_params": {{ ... }},
  "status": "awaiting_user_input" // or "objective_complete"
}}
"""
    llm = LLMClient()
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    try:
        llm_response = await llm.chat(messages)
        logger.info(f"[LLM-Driven Executor] LLM response: {llm_response}")
        # Parse response
        import json
        if isinstance(llm_response, str):
            # Try to extract JSON
            start = llm_response.find('{')
            end = llm_response.rfind('}')
            if start != -1 and end != -1:
                llm_response = llm_response[start:end+1]
            parsed = json.loads(llm_response)
        elif isinstance(llm_response, dict):
            parsed = llm_response
        else:
            raise ValueError("LLM did not return a valid JSON object.")
        skill_name = parsed.get("skill_name")
        skill_params = parsed.get("skill_params", {})
        status = parsed.get("status", "awaiting_user_input")
        return {"skill_name": skill_name, "skill_params": skill_params, "status": status}, status
    except Exception as e:
        from ai_tutor.api_models import InteractionResponseData, ErrorResponse
        from ai_tutor.core_models import UserModelState
        logger.error(f"[LLM-Driven Executor] Error in LLM or parsing: {e}")
        error_data = ErrorResponse(
            error_message="An unexpected error occurred. Please try again.",
            error_code="UNEXPECTED_EXECUTOR_ERROR",
            technical_details=str(e)
        )
        user_model_state_dict = ctx.user_model_state if hasattr(ctx, 'user_model_state') and ctx.user_model_state else UserModelState()
        return InteractionResponseData(
            content_type="error",
            data=error_data,
            user_model_state=user_model_state_dict
        ), "error"


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