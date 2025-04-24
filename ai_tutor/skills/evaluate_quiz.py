from agents import function_tool
from ai_tutor.utils.agent_callers import call_quiz_teacher_evaluate
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper

@function_tool
async def evaluate_quiz(ctx: RunContextWrapper[TutorContext], user_answer_index: int):
    """Skill wrapper that delegates to the Quiz Teacher evaluation tool."""
    return await call_quiz_teacher_evaluate(ctx, user_answer_index) 