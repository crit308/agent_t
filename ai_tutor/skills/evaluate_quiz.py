from ai_tutor.tools import call_quiz_teacher_evaluate
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.skills import tool

@tool()
async def evaluate_quiz(ctx: RunContextWrapper[TutorContext], user_answer_index: int):
    """Skill wrapper that delegates to the Quiz Teacher evaluation tool."""
    return await call_quiz_teacher_evaluate(ctx, user_answer_index) 