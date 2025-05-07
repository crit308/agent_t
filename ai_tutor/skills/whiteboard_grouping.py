"""Skills for grouping objects on the whiteboard."""

from typing import Any, Dict, List

from педагог.skill_library.skill import skill


@skill
async def group_objects(ctx: Any, group_id: str, object_ids: List[str]) -> Dict[str, Any]:
    """Creates a group of specified objects on the whiteboard."""
    return {"type": "GROUP_OBJECTS", "groupId": group_id, "objectIds": object_ids}


@skill
async def move_group(ctx: Any, group_id: str, dx_pct: float, dy_pct: float) -> Dict[str, Any]:
    """Moves a group of objects on the whiteboard by a percentage of canvas dimensions."""
    # Assuming dx, dy were meant to be percentages as per the new coord system.
    # If they are absolute, the type hint and description might need adjustment.
    # For now, let's assume they are percentages to be consistent.
    return {"type": "MOVE_GROUP", "groupId": group_id, "dxPct": dx_pct, "dyPct": dy_pct}


@skill
async def delete_group(ctx: Any, group_id: str) -> Dict[str, Any]:
    """Deletes a group of objects (and its members) from the whiteboard."""
    return {"type": "DELETE_GROUP", "groupId": group_id} 