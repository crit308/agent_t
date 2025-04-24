from ai_tutor.tools import call_teacher_agent
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.skills import skill, as_tool

@skill(cost="high")
@as_tool
async def answer_question(ctx: RunContextWrapper[TutorContext], question: str) -> str:
    """Skill wrapper that delegates to the Teacher agent for answering a direct student question."""
    # Use the current focus objective's topic if available, else generic
    topic = ctx.context.current_focus_objective.topic if ctx.context.current_focus_objective else 'General'
    return await call_teacher_agent(ctx, topic, question) 