from agents import function_tool
from ai_tutor.agents.planner_agent import run_planner
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext

@function_tool
async def plan_next(ctx: RunContextWrapper[TutorContext]):
    """Generates or updates the lesson plan based on the current context."""
    # Note: run_planner might need the raw TutorContext, not the wrapper
    # Adjust if needed based on run_planner's signature
    return await run_planner(ctx.context) 