from agents import function_tool
from typing import Optional, Dict, Any
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper

@function_tool
async def get_user_model_status(ctx: RunContextWrapper[TutorContext], topic: Optional[str] = None) -> Dict[str, Any]:
    """Skill wrapper for retrieving the user model status."""
    # Implement the logic here or call the appropriate method on ctx
    # Placeholder: return a dummy status
    return {"status": "ok"} 