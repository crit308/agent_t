"""
ExecutorAgent: picks and executes skills to fulfill a learning objective.
"""

from ai_tutor.core.enums import ExecutorStatus
from ai_tutor.context import TutorContext
from ai_tutor.agents.models import FocusObjective
from ai_tutor.skills import get_tool
from agents.run_context import RunContextWrapper
from typing import Any

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
    async def run(objective: FocusObjective, ctx: TutorContext) -> tuple[ExecutorStatus, Any]:
        """
        Execute one skill tactic for the given objective and return (status, result).
        """
        # Wrap context for tool invocation
        wrapper = RunContextWrapper(ctx)
        # Choose and run a single tactic
        tactic, params = await choose_tactic(ctx, objective)
        skill_fn = get_tool(tactic)
        # Enforce high-cost skill budget if tagged
        cost = getattr(skill_fn, '_skill_cost', None)
        if cost == 'high':
            if ctx.high_cost_calls >= ctx.max_high_cost_calls:
                # Budget exhausted: treat as stuck and skip this tactic
                return ExecutorStatus.STUCK, f"High-cost skill budget exceeded ({ctx.high_cost_calls}/{ctx.max_high_cost_calls})"
            ctx.high_cost_calls += 1
        result = await skill_fn(wrapper, **params)
        # Update context for next evaluation
        ctx.last_interaction_summary = str(result)
        # Determine status: completed on mastery, stuck on fallback, otherwise continue
        if objective_completed(ctx, objective):
            status = ExecutorStatus.COMPLETED
        elif stuck(ctx):
            status = ExecutorStatus.STUCK
        else:
            status = ExecutorStatus.CONTINUE
        return status, result 