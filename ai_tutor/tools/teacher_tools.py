from agents import function_tool
from agents.run_context import RunContextWrapper
from typing import List, Dict, Any, Literal, cast
import uuid

# Import necessary response models from api_models.py
from ai_tutor.api_models import ExplanationResponse, QuestionResponse, MessageResponse # Add others if needed
from ai_tutor.agents.models import QuizQuestion # If asking structured questions
from ai_tutor.context import TutorContext # Import TutorContext for type hint

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