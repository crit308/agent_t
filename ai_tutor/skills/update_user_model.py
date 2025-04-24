# ai_tutor/skills/update_user_model.py
from typing import Optional
from typing import Literal
from typing import Union
from ai_tutor.tools import update_user_model
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.skills import skill, as_tool

@skill(cost="low")
@as_tool
async def update_user_model_skill(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained'],
    confusion_point: Optional[str] = None,
    last_accessed: Optional[str] = None,
    mastered_objective_title: Optional[str] = None
) -> str:
    """Skill wrapper that updates the user model state."""
    return await update_user_model(
        ctx,
        topic=topic,
        outcome=outcome,
        confusion_point=confusion_point,
        last_accessed=last_accessed,
        mastered_objective_title=mastered_objective_title
    ) 