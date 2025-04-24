from ai_tutor.tools import call_teacher_agent
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.skills import skill, as_tool

@skill(cost="medium")
@as_tool
async def remediate_concept(ctx: RunContextWrapper[TutorContext], topic: str, remediation_details: str):
    """Skill wrapper that delegates to the Teacher agent for remediation after a quiz failure."""
    # In Phase 3, reuse existing teacher agent for remediation
    return await call_teacher_agent(ctx, topic, remediation_details) 