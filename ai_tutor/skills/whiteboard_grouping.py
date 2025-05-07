"""Skills for grouping objects on the whiteboard."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, conlist, ValidationError
from ai_tutor.skills import skill
from ai_tutor.exceptions import ToolInputError

class GroupObjectsArgs(BaseModel):
    group_id: str = Field(..., min_length=1)
    object_ids: conlist(str, min_length=1)

class MoveGroupArgs(BaseModel):
    group_id: str = Field(..., min_length=1)
    dx_pct: float
    dy_pct: float

class DeleteGroupArgs(BaseModel):
    group_id: str = Field(..., min_length=1)

@skill
async def group_objects(ctx: Any, **kwargs) -> Dict[str, Any]:
    """Creates a group of specified objects on the whiteboard."""
    try:
        validated_args = GroupObjectsArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for group_objects: {e}")
    return {"type": "GROUP_OBJECTS", "groupId": validated_args.group_id, "objectIds": validated_args.object_ids}


@skill
async def move_group(ctx: Any, **kwargs) -> Dict[str, Any]:
    """Moves a group of objects on the whiteboard by a percentage of canvas dimensions."""
    try:
        validated_args = MoveGroupArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for move_group: {e}")
    return {"type": "MOVE_GROUP", "groupId": validated_args.group_id, "dxPct": validated_args.dx_pct, "dyPct": validated_args.dy_pct}


@skill
async def delete_group(ctx: Any, **kwargs) -> Dict[str, Any]:
    """Deletes a group of objects (and its members) from the whiteboard."""
    try:
        validated_args = DeleteGroupArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for delete_group: {e}")
    return {"type": "DELETE_GROUP", "groupId": validated_args.group_id} 