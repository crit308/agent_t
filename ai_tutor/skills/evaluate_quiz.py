# from agents import function_tool # No longer used
from ai_tutor.skills import skill # Import correct decorator
# from ai_tutor.utils.agent_callers import call_quiz_teacher_evaluate # No longer used
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem # Import models
from ai_tutor.api_models import FeedbackResponse # Import FeedbackResponse
# from ai_tutor.errors import ToolExecutionError # Using exceptions module now
import logging
from typing import Any, Dict, List, Optional # Added Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError # Added BaseModel, Field, ValidationError
from ai_tutor.exceptions import ToolInputError, ExecutorError # Added ToolInputError, ExecutorError

logger = logging.getLogger(__name__)

class EvaluateQuizArgs(BaseModel):
    user_answer_index: int = Field(..., ge=0, description="The 0-based index of the user's selected answer.")

@skill
async def evaluate_quiz(ctx: RunContextWrapper[TutorContext], **kwargs) -> FeedbackResponse:
    """Evaluates the user's answer against the current QuizQuestion stored in context and generates whiteboard actions for feedback."""
    try:
        args = EvaluateQuizArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for evaluate_quiz: {e}")

    user_answer_idx = args.user_answer_index
    logger.info(f"Evaluating quiz answer. User selected index: {user_answer_idx}")
    
    question = ctx.context.current_quiz_question
    
    if not question or not isinstance(question, QuizQuestion):
        logger.error("evaluate_quiz skill called but no valid QuizQuestion found in context.")
        # This is more of an execution state error than a tool input error
        raise ExecutorError("No valid quiz question found in context to evaluate.") 

    try:
        if not (0 <= user_answer_idx < len(question.options)):
            error_msg = f"Selected answer index ({user_answer_idx}) is out of bounds for question options (count={len(question.options)})."
            logger.error(error_msg)
            # Raise ToolInputError as it relates to the validity of the input index against the context
            raise ToolInputError(error_msg)
            
        is_correct = (user_answer_idx == question.correct_answer_index)
        selected_option = question.options[user_answer_idx]
        correct_option = question.options[question.correct_answer_index]
        
        logger.info(f"Answer evaluation: User selected '{selected_option}' (index {user_answer_idx}). Correct: '{correct_option}' (index {question.correct_answer_index}). Correct: {is_correct}")

        feedback = QuizFeedbackItem(
            question_index=0,
            question_text=question.question,
            user_selected_option=selected_option,
            is_correct=is_correct,
            correct_option=correct_option,
            explanation=question.explanation,
            improvement_suggestion="Consider reviewing the explanation and related concepts." if not is_correct else "Great job!"
        )
        
        # TODO: Generate whiteboard actions based on the actual MCQ drawing elements if they exist
        # For now, creating simple feedback indicators assuming generic element IDs
        radio_id = f"mcq-q1-opt-{user_answer_idx}-radio"
        check_id = f"mcq-q1-opt-{user_answer_idx}-check"
        whiteboard_actions = [
            {
                "id": radio_id, 
                "kind": "style_update", # Use style_update to change existing element
                "updates": { "fill": "#4CAF50" if is_correct else "#F44336" },
                "metadata": { "role": "option_selector_feedback" }
            },
            {
                "id": check_id,
                "kind": "text", # Create a new text element for check/cross
                "text": "✓" if is_correct else "✗",
                "x": 0, # Placeholder - FE might need to position this near the radio button
                "y": 0, 
                "fontSize": 18,
                "fill": "#4CAF50" if is_correct else "#F44336",
                "metadata": {
                    "source": "assistant",
                    "role": "option_feedback_mark",
                    "question_id": "q1", # Needs actual question id
                    "option_id": user_answer_idx
                }
            }
        ]
        
        # Clear the question from context *after* successful evaluation
        if ctx.context.user_model_state:
            ctx.context.user_model_state.pending_interaction_type = None
        ctx.context.current_quiz_question = None
        logger.info("Cleared current_quiz_question from context.")

        logger.info(f"Quiz evaluation complete. Correct: {feedback.is_correct}")
        payload = FeedbackResponse(
            feedback_items=[feedback],
            overall_assessment=None,
            suggested_next_step=None
        )
        setattr(payload, 'whiteboard_actions', whiteboard_actions)
        return payload

    except ToolInputError: # Re-raise ToolInputError specifically
        raise
    except Exception as e:
        logger.error(f"Error during quiz evaluation logic: {e}", exc_info=True)
        # Wrap other internal errors in ExecutorError
        raise ExecutorError(f"Failed during quiz evaluation logic: {e}") from e 