from ai_tutor.skills import skill
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator, ValidationError
from ai_tutor.exceptions import ToolInputError

class DrawCoordinatePlaneArgs(BaseModel):
    plane_id: str = Field(..., min_length=1)
    x_range: Tuple[float, float] = (-10.0, 10.0)
    y_range: Tuple[float, float] = (-10.0, 10.0)
    x_label: str = "X"
    y_label: str = "Y"
    num_ticks_x: int = Field(default=5, ge=0) # Cannot be negative
    num_ticks_y: int = Field(default=5, ge=0) # Cannot be negative
    show_grid: bool = False
    x: Optional[int] = 50
    y: Optional[int] = 300
    width: int = Field(default=250, gt=0) # Must be positive
    height: int = Field(default=200, gt=0) # Must be positive

    @validator('x_range', 'y_range')
    def check_range_order(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Min value in range must be less than max value")
        return v

@skill
async def draw_coordinate_plane(ctx: Any, **kwargs) -> Dict[str, Any]:
    """Generates specs to draw a 2D Cartesian coordinate plane."""
    try:
        args = DrawCoordinatePlaneArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for draw_coordinate_plane: {e}")

    objects: List[Dict[str, Any]] = []
    origin_x, origin_y = args.x, args.y
    min_x, max_x = args.x_range
    min_y, max_y = args.y_range

    # Ensure division by zero does not occur if min_x == max_x or min_y == max_y
    # This should be caught by the validator, but as a safeguard:
    if max_x == min_x or max_y == min_y:
        # This case should ideally not happen due to validator but good to be defensive
        raise ToolInputError("Range min and max values cannot be identical for draw_coordinate_plane.")

    pixel_per_unit_x = args.width / (max_x - min_x)
    pixel_per_unit_y = args.height / (max_y - min_y)

    # --- X-Axis Line ---
    objects.append({
        "id": f"{args.plane_id}-xaxis", "kind": "line",
        "points": [origin_x + min_x * pixel_per_unit_x, origin_y, origin_x + max_x * pixel_per_unit_x, origin_y],
        "stroke": "black", "strokeWidth": 2, "metadata": {"id": f"{args.plane_id}-xaxis", "groupId": args.plane_id, "role": "axis"}
    })
    # --- Y-Axis Line (Inverted Y) ---
    objects.append({
        "id": f"{args.plane_id}-yaxis", "kind": "line",
        "points": [origin_x, origin_y - max_y * pixel_per_unit_y, origin_x, origin_y - min_y * pixel_per_unit_y],
        "stroke": "black", "strokeWidth": 2, "metadata": {"id": f"{args.plane_id}-yaxis", "groupId": args.plane_id, "role": "axis"}
    })
    # --- Labels ---
    objects.append({
        "id": f"{args.plane_id}-xlabel", "kind": "textbox", "text": args.x_label,
        "x": origin_x + max_x * pixel_per_unit_x + 5, "y": origin_y - 10,
        "fontSize": 12, "metadata": {"id": f"{args.plane_id}-xlabel", "groupId": args.plane_id, "role": "label"}
    })
    objects.append({
        "id": f"{args.plane_id}-ylabel", "kind": "textbox", "text": args.y_label,
        "x": origin_x - 10, "y": origin_y - max_y * pixel_per_unit_y - 15,
        "fontSize": 12, "metadata": {"id": f"{args.plane_id}-ylabel", "groupId": args.plane_id, "role": "label"}
    })

    # --- Ticks (Example for X-axis) ---
    if args.num_ticks_x > 0:
        tick_step_x = (max_x - min_x) / (args.num_ticks_x + 1)
        for i in range(1, args.num_ticks_x + 1):
            tick_val = min_x + i * tick_step_x
            tick_x = origin_x + tick_val * pixel_per_unit_x
            objects.append({
                "id": f"{args.plane_id}-xtick-{i}", "kind": "line",
                "points": [tick_x, origin_y - 3, tick_x, origin_y + 3],
                "stroke": "black", "strokeWidth": 1, "metadata": {"id": f"{args.plane_id}-xtick-{i}", "groupId": args.plane_id, "role": "tick"}
            })
    # TODO: Add Y-axis ticks and optional grid lines

    return {"type": "ADD_OBJECTS", "objects": objects} 