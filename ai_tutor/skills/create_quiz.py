from ai_tutor.tools import call_quiz_creator_agent
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.skills import tool

@tool(cost="high")
async def create_quiz(ctx: RunContextWrapper[TutorContext], topic: str, instructions: str):
    """Skill wrapper that delegates to the Quiz Creator Agent to generate quiz questions."""
    return await call_quiz_creator_agent(ctx, topic, instructions) 