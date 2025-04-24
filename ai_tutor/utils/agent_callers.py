"""Helper functions for invoking specific agents."""

from __future__ import annotations
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, ExplanationResult, QuizCreationResult
from ai_tutor.errors import ToolExecutionError
from ai_tutor.utils.tool_helpers import invoke

async def call_quiz_teacher_evaluate(ctx: RunContextWrapper[TutorContext], user_answer_index: int) -> QuizFeedbackItem:
    """Evaluate the user's answer against the current QuizQuestion in context."""
    question = ctx.context.current_quiz_question
    if not question or not isinstance(question, QuizQuestion):
        raise ToolExecutionError("No valid QuizQuestion in context to evaluate.", code="missing_question")
    is_correct = (user_answer_index == question.correct_answer_index)
    selected = question.options[user_answer_index] if 0 <= user_answer_index < len(question.options) else ""
    correct = question.options[question.correct_answer_index]
    feedback = QuizFeedbackItem(
        question_index=user_answer_index,
        question_text=question.question,
        user_selected_option=selected,
        is_correct=is_correct,
        correct_option=correct,
        explanation=question.explanation,
        improvement_suggestion="Review the concept again." if not is_correct else ""
    )
    # Clear pending question state
    if ctx.context.user_model_state:
        ctx.context.user_model_state.pending_interaction_type = None
    ctx.context.current_quiz_question = None
    return feedback

async def call_teacher_agent(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    explanation_details: str
) -> ExplanationResult:
    """Invokes the Teacher Agent to generate an explanation for a topic."""
    print(f"[Agent Caller] Calling Teacher Agent for topic: {topic}")
    teacher_agent = None # Placeholder - needs actual agent retrieval
    if not teacher_agent:
        print("[Agent Caller] Warning: Teacher agent runner not found directly in context. Invocation might rely on global registry.")
        agent_name = "teacher_agent" # Assuming agent name
    else:
        agent_name = teacher_agent

    try:
        result = await invoke(agent_name, input={"topic": topic, "details": explanation_details}, context=ctx.context)
        if isinstance(result, dict) and "explanation" in result:
            return ExplanationResult(**result)
        elif isinstance(result, str):
            return ExplanationResult(explanation=result, segment_index=0)
        raise ToolExecutionError(f"Unexpected result format from teacher agent: {type(result)}", code="agent_response_error")
    except Exception as e:
        print(f"[Agent Caller] Error invoking teacher agent: {e}")
        raise ToolExecutionError(f"Failed to invoke teacher agent: {e}", code="agent_invocation_error")

async def call_quiz_creator_agent(ctx: RunContextWrapper[TutorContext], topic: str, instructions: str) -> QuizCreationResult:
    """Invokes the Quiz Creator Agent to generate a quiz question."""
    print(f"[Agent Caller] Calling Quiz Creator Agent for topic: {topic}")
    quiz_creator_agent = None # Placeholder
    if not quiz_creator_agent:
        print("[Agent Caller] Warning: Quiz Creator agent runner not found directly in context. Invocation might rely on global registry.")
        agent_name = "quiz_creator_agent"
    else:
        agent_name = quiz_creator_agent

    try:
        result = await invoke(agent_name, input={"topic": topic, "instructions": instructions}, context=ctx.context)
        if isinstance(result, dict) and "question" in result and "options" in result:
            return QuizCreationResult(**result)
        raise ToolExecutionError(f"Unexpected result format from quiz creator agent: {type(result)}", code="agent_response_error")
    except Exception as e:
        print(f"[Agent Caller] Error invoking quiz creator agent: {e}")
        raise ToolExecutionError(f"Failed to invoke quiz creator agent: {e}", code="agent_invocation_error") 