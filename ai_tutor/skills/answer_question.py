from agents import function_tool
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper

@function_tool
async def answer_question(ctx: RunContextWrapper[TutorContext], question: str) -> str:
    """Skill wrapper that delegates to the Teacher agent for answering a direct student question."""
    # Use the current focus objective's topic if available, else generic
    topic = ctx.current_focus_objective.topic if ctx.current_focus_objective else 'General'
    # TODO: replace with direct LLM explanation or new skill implementation
    return f"[placeholder answer for topic '{topic}' and question '{question}']" 