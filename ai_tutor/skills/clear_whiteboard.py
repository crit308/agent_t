from typing import List, Dict, Any

from ai_tutor.skills import skill


@skill
async def clear_whiteboard() -> List[Dict[str, Any]]:
    """Returns a whiteboard action list containing only a 'reset' action.

    This signals the frontend to clear all existing elements drawn by the assistant.
    """
    return [
        {
            "id": "global-reset-0",  # ID is somewhat arbitrary for reset, but good practice
            "kind": "reset",
            "metadata": {"source": "assistant", "reason": "new_diagram"},
        }
    ] 