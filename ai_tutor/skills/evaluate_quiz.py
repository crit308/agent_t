# from agents import function_tool # No longer used
from ai_tutor.skills import skill # Import correct decorator
# from ai_tutor.utils.agent_callers import call_quiz_teacher_evaluate # No longer used
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem # Import models
from ai_tutor.errors import ToolExecutionError # Import custom error
import logging

logger = logging.getLogger(__name__)

@skill
async def evaluate_quiz(ctx: RunContextWrapper[TutorContext], user_answer_index: int) -> QuizFeedbackItem:
    """Evaluates the user's answer against the current QuizQuestion stored in context.
    
    Reads `ctx.current_quiz_question`, compares the answer, constructs
    a QuizFeedbackItem, clears `ctx.current_quiz_question`, and returns
    the feedback item.
    """
    logger.info(f"Evaluating quiz answer. User selected index: {user_answer_index}")
    
    question = ctx.current_quiz_question
    
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
        
        # Clear the question from context *after* successful evaluation
        if ctx.user_model_state:
            ctx.user_model_state.pending_interaction_type = None
        ctx.current_quiz_question = None
        logger.info("Cleared current_quiz_question from context.")

        logger.info(f"Quiz evaluation complete. Correct: {feedback.is_correct}")
        return feedback

    except Exception as e:
        logger.error(f"Error during quiz evaluation: {e}", exc_info=True)
        # Re-raise to allow Executor/handler to potentially return an ErrorResponse
        raise ToolExecutionError(f"Failed during quiz evaluation: {e}", code="evaluation_error") 