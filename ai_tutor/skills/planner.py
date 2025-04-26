from agents import function_tool
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext

@function_tool
async def plan_next(ctx: RunContextWrapper[TutorContext]):
    """
    Placeholder skill. The actual planning logic lives in planner_agent.py
    and is invoked by the main loop/agent, not directly by this skill.
    This skill might evolve to *request* a replan if needed.
    """
    # This skill might not actually *do* anything itself in the new model,
    # or it might formulate a request for the main agent loop to replan.
    logger.info("[Skill plan_next] Triggered. Placeholder. Requesting main loop to potentially re-plan.")
    # For now, just return a simple message.
    # The actual call to run_planner happens elsewhere (e.g., tutor_ws.py or executor deciding to replan)
    return "Planning check requested." 