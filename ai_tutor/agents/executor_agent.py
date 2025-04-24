"""
ExecutorAgent: picks and executes skills to fulfill a learning objective.
"""

from ai_tutor.context import TutorContext
from ai_tutor.agents.models import FocusObjective
from ai_tutor.api_models import ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse
from agents.run_context import RunContextWrapper
from typing import Any

# Define custom exception for budget issues
class BudgetExceededError(Exception):
    pass

async def choose_tactic(ctx: TutorContext, objective: FocusObjective):
    """
    Choose the next skill (tactic) based on Phase 3 micro-policy:
    1) Direct question → answer_question
    2) First explanation segment → explain_concept
    3) Quiz failed → remediate_concept
    4) After ≥3 explanation segments → create_quiz
    5) Default → continue explanation
    """
    # Examine the last event for direct questions
    last_event = getattr(ctx, 'last_event', {}) or {}
    event_type = last_event.get('event_type')
    # 1. User answered a quiz question: evaluate answer
    if event_type == 'user_answer':
        data = last_event.get('data', {}) or {}
        answer_index = data.get('user_answer_index')
        return 'evaluate_quiz', {'user_answer_index': answer_index}
    # 2. Student asked a direct question
    if event_type == 'user_question':
        data = last_event.get('data', {}) or {}
        question_text = data.get('question') or data.get('question_text', '')
        return 'answer_question', {'question': question_text}
    # First segment: explanation
    if ctx.user_model_state.current_topic_segment_index == 0:
        return 'explain_concept', {'topic': objective.topic, 'explanation_details': f'Explain the concept: {objective.topic}'}
    # Quiz failure remediation
    summary = getattr(ctx, 'last_interaction_summary', '') or ''
    if isinstance(summary, str) and any(kw in summary.lower() for kw in ['incorrect', 'struggled']):
        return 'remediate_concept', {'topic': objective.topic, 'remediation_details': f'Review concept due to difficulty: {summary}'}
    # After several explanation segments, quiz
    if ctx.user_model_state.current_topic_segment_index >= 3:
        return 'create_quiz', {'topic': objective.topic, 'instructions': f'Generate a 3-question quiz for {objective.topic}'}
    # Default: continue explanation
    return 'explain_concept', {'topic': objective.topic, 'explanation_details': f'Continue explanation for {objective.topic}'}


def objective_completed(ctx: TutorContext, objective: FocusObjective) -> bool:
    """
    Return True if the user has answered the quiz for this topic and their mastery >= target_mastery.
    """
    # Ensure a quiz answer event triggered
    last_event = getattr(ctx, 'last_event', {}) or {}
    if last_event.get('event_type') != 'user_answer':
        return False
    # Check mastery threshold
    mastery_state = ctx.user_model_state.concepts.get(objective.topic)
    if not mastery_state:
        return False
    return mastery_state.mastery >= objective.target_mastery


def stuck(ctx: TutorContext) -> bool:
    """
    Determine if executor is stuck (unable to make progress).
    Here: never stuck for initial phase.
    """
    return False

class ExecutorAgent:
    """Executor that runs skills to fulfill a FocusObjective."""

    @staticmethod
    async def run(objective: FocusObjective, ctx: TutorContext) -> Any:
        """
        Execute one skill tactic for the given objective and return the result payload.
        Raises BudgetExceededError if high-cost skill budget is exhausted.
        Raises ValueError if wrapping the result fails.
        """
        # Wrap context for tool invocation
        wrapper = RunContextWrapper(ctx)
        # Choose and run a single tactic
        tactic, params = await choose_tactic(ctx, objective)
        # Import get_tool here to avoid circular import at module load
        from ai_tutor.skills import get_tool
        skill_fn = get_tool(tactic)
        # Enforce high-cost skill budget if tagged
        cost = getattr(skill_fn, '_skill_cost', None)
        if cost == 'high':
            if ctx.high_cost_calls >= ctx.max_high_cost_calls:
                # Budget exhausted: raise exception
                msg = f"High-cost skill budget exceeded ({ctx.high_cost_calls}/{ctx.max_high_cost_calls}) for tactic '{tactic}'"
                print(f"[Executor Error] {msg}") # Log it too
                raise BudgetExceededError(msg)
            ctx.high_cost_calls += 1
        result = await skill_fn(wrapper, **params)

        # Update context for next evaluation - Use a structured summary if possible
        # For now, keep simple string conversion, but TODO: improve this
        ctx.last_interaction_summary = str(result)

        # --- Always wrap the result based on the tactic --- 
        output: Any = None
        try:
            if tactic == 'explain_concept' or tactic == 'remediate_concept' or tactic == 'answer_question':
                # Assuming result is the explanation text string
                # Increment segment index *before* checking for last segment
                ctx.user_model_state.current_topic_segment_index += 1
                output = ExplanationResponse(
                    response_type='explanation',
                    text=str(result), # Ensure it's a string
                    topic=objective.topic, # Use topic from objective
                    segment_index=ctx.user_model_state.current_topic_segment_index, # Use updated segment
                    # TODO: Add more robust logic to determine is_last_segment (e.g., based on planner/KB structure)
                    is_last_segment= (ctx.user_model_state.current_topic_segment_index >= 3) # Placeholder: assume quiz after 3 segments
                )
            elif tactic == 'create_quiz':
                # Assuming result is a QuizQuestion object from the skill
                # TODO: Verify the actual return type of create_quiz skill
                from ai_tutor.agents.models import QuizQuestion # Import locally if needed
                if isinstance(result, QuizQuestion):
                     output = QuestionResponse(
                          response_type='question',
                          question=result,
                          topic=objective.topic
                     )
                else:
                     # Handle unexpected result type from create_quiz
                     print(f"[Executor Warning] Unexpected result type from create_quiz: {type(result)}. Expected QuizQuestion.")
                     # Return an error response instead of raw data
                     output = ErrorResponse(response_type='error', message=f"Internal error: Unexpected quiz format from '{tactic}'.")

            elif tactic == 'evaluate_quiz':
                 # Assuming result is a QuizFeedbackItem object from the skill
                 # TODO: Verify the actual return type of evaluate_quiz skill
                 from ai_tutor.agents.models import QuizFeedbackItem # Import locally if needed
                 if isinstance(result, QuizFeedbackItem):
                      output = FeedbackResponse(
                           response_type='feedback',
                           feedback=result,
                           topic=objective.topic
                           # TODO: Add correct_answer, explanation if available from skill
                      )
                 else:
                      # Handle unexpected result type
                      print(f"[Executor Warning] Unexpected result type from evaluate_quiz: {type(result)}. Expected QuizFeedbackItem.")
                      # Return an error response
                      output = ErrorResponse(response_type='error', message=f"Internal error: Unexpected feedback format from '{tactic}'.")
            else:
                # Fallback for unknown tactic
                print(f"[Executor Warning] Unknown tactic '{tactic}' executed. Returning raw result as MessageResponse.")
                output = MessageResponse(response_type='message', text=f"Tutor response: {str(result)}")

        except Exception as e:
             # Handle potential errors during response model creation
             msg = f"Failed to wrap result for tactic '{tactic}': {e}"
             print(f"[Executor Error] {msg}")
             raise ValueError(msg) from e # Raise an error if wrapping fails

        # Return the wrapped payload directly
        return output 