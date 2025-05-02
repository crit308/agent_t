# from agents import function_tool # No longer used
from ai_tutor.skills import skill # Import correct decorator
# from ai_tutor.utils.agent_callers import call_quiz_teacher_evaluate # No longer used
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem # Import models
from ai_tutor.api_models import FeedbackResponse # Import FeedbackResponse
from ai_tutor.errors import ToolExecutionError # Import custom error
import logging

logger = logging.getLogger(__name__)

@skill
async def evaluate_quiz(ctx: RunContextWrapper[TutorContext], user_answer_index: int) -> FeedbackResponse:
    """Evaluates the user's answer against the current QuizQuestion stored in context and generates whiteboard actions for feedback."""
    logger.info(f"Evaluating quiz answer. User selected index: {user_answer_index}")
    
    # Access the underlying TutorContext via the .context attribute of the wrapper
    question = ctx.context.current_quiz_question
    
    if not question or not isinstance(question, QuizQuestion):
        logger.error("evaluate_quiz skill called but no valid QuizQuestion found in context.")
        raise ToolExecutionError("No valid quiz question found in context to evaluate.", code="missing_question")

    try:
        # Ensure user_answer_index is within bounds
        if not (0 <= user_answer_index < len(question.options)):
            logger.error(f"Invalid user_answer_index ({user_answer_index}) for question options count ({len(question.options)}).")
            raise ValueError(f"Selected answer index ({user_answer_index}) is out of bounds.")
            
        is_correct = (user_answer_index == question.correct_answer_index)
        selected_option = question.options[user_answer_index]
        correct_option = question.options[question.correct_answer_index]
        
        logger.info(f"Answer evaluation: User selected '{selected_option}' (index {user_answer_index}). Correct: '{correct_option}' (index {question.correct_answer_index}). Correct: {is_correct}")

        # Construct feedback - using 0 for question_index assuming one question at a time
        feedback = QuizFeedbackItem(
            question_index=0, # Assuming single question context for now
            question_text=question.question,
            user_selected_option=selected_option,
            is_correct=is_correct,
            correct_option=correct_option,
            explanation=question.explanation,
            improvement_suggestion="Consider reviewing the explanation and related concepts." if not is_correct else "Great job!"
        )
        
        # Generate whiteboard actions for feedback
        radio_id = f"mcq-q1-opt-{user_answer_index}-radio"
        check_id = f"mcq-q1-opt-{user_answer_index}-check"
        whiteboard_actions = [
            {
                "id": radio_id,
                "kind": "circle",
                "fill": "#4CAF50" if is_correct else "#F44336",
                "stroke": "#222222",
                "strokeWidth": 2,
                "metadata": {
                    "source": "assistant",
                    "role": "option_selector_feedback",
                    "question_id": "q1",
                    "option_id": user_answer_index
                }
            },
            {
                "id": check_id,
                "kind": "text",
                "x": 0,  # Let FE auto-layout or you can set coordinates if you have them
                "y": 0,
                "text": "✓" if is_correct else "✗",
                "fontSize": 18,
                "fill": "#4CAF50" if is_correct else "#F44336",
                "metadata": {
                    "source": "assistant",
                    "role": "option_feedback_mark",
                    "question_id": "q1",
                    "option_id": user_answer_index
                }
            }
        ]
        
        # Clear the question from context *after* successful evaluation
        if ctx.context.user_model_state:
            # Access the underlying context via the .context attribute
            ctx.context.user_model_state.pending_interaction_type = None
        ctx.context.current_quiz_question = None
        logger.info("Cleared current_quiz_question from context.")

        logger.info(f"Quiz evaluation complete. Correct: {feedback.is_correct}")
        # Build FeedbackResponse payload
        payload = FeedbackResponse(
            feedback_items=[feedback],
            overall_assessment=None,
            suggested_next_step=None
        )
        # Attach whiteboard_actions as a non-model attribute for the executor to pick up
        setattr(payload, 'whiteboard_actions', whiteboard_actions)
        return payload

    except Exception as e:
        logger.error(f"Error during quiz evaluation: {e}", exc_info=True)
        # Re-raise to allow Executor/handler to potentially return an ErrorResponse
        raise ToolExecutionError(f"Failed during quiz evaluation: {e}", code="evaluation_error") 