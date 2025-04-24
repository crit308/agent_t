# ai_tutor/skills/update_user_model.py
from agents import function_tool
from typing import Optional, Literal
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper

@function_tool
async def update_user_model_skill(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained'],
    confusion_point: Optional[str] = None,
    last_accessed: Optional[str] = None,
    mastered_objective_title: Optional[str] = None
) -> str:
    """Skill wrapper that updates the user model state."""
    # Implement the logic here or call the appropriate method on ctx
    # Placeholder: return a dummy update message
    return f"User model updated for {topic} with outcome {outcome}." 