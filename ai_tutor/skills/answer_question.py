from agents import function_tool
from ai_tutor.utils.agent_callers import call_teacher_agent
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper

@function_tool
async def answer_question(ctx: RunContextWrapper[TutorContext], question: str) -> str:
    """Skill wrapper that delegates to the Teacher agent for answering a direct student question."""
    # Use the current focus objective's topic if available, else generic
    topic = ctx.current_focus_objective.topic if ctx.current_focus_objective else 'General'
    return await call_teacher_agent(ctx, topic, question) 