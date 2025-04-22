from typing import Optional, Dict, Any
from ai_tutor.tools import get_user_model_status as _gums
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.skills import tool

@tool()
async def get_user_model_status(ctx: RunContextWrapper[TutorContext], topic: Optional[str] = None) -> Dict[str, Any]:
    """Skill wrapper for retrieving the user model status."""
    return await _gums(ctx, topic) 