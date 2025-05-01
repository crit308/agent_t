from typing import List, Dict, Any

from ai_tutor.skills import skill


@skill
async def draw_axis_actions(
    *,
    axis_id: str = "axis-1",
    start_x: int = 50,
    start_y: int = 300,
    width: int = 250,
    height: int = 200,
    show_arrows: bool = True,
    label_x: str = "X",
    label_y: str = "Y",
) -> List[Dict[str, Any]]:
    """Return actions for a simple X/Y axis (Cartesian plane)."""
    actions: List[Dict[str, Any]] = []

    # X-axis line
    actions.append(
        {
            "id": f"{axis_id}-x-line",
            "kind": "line",
            "points": [start_x, start_y, start_x + width, start_y],
            "stroke": "#000",
            "strokeWidth": 2,
            "metadata": {"source": "assistant", "role": "axis_x", "axis_id": axis_id},
        }
    )

    # Y-axis line
    actions.append(
        {
            "id": f"{axis_id}-y-line",
            "kind": "line",
            "points": [start_x, start_y, start_x, start_y - height],
            "stroke": "#000",
            "strokeWidth": 2,
            "metadata": {"source": "assistant", "role": "axis_y", "axis_id": axis_id},
        }
    )

    if show_arrows:
        # Arrow heads – simple small lines
        actions.append(
            {
                "id": f"{axis_id}-x-arrow",
                "kind": "line",
                "points": [start_x + width, start_y, start_x + width - 10, start_y - 5],
                "stroke": "#000",
                "strokeWidth": 2,
                "metadata": {"source": "assistant", "role": "axis_x_arrow", "axis_id": axis_id},
            }
        )
        actions.append(
            {
                "id": f"{axis_id}-x-arrow2",
                "kind": "line",
                "points": [start_x + width, start_y, start_x + width - 10, start_y + 5],
                "stroke": "#000",
                "strokeWidth": 2,
                "metadata": {"source": "assistant", "role": "axis_x_arrow", "axis_id": axis_id},
            }
        )
        actions.append(
            {
                "id": f"{axis_id}-y-arrow",
                "kind": "line",
                "points": [start_x, start_y - height, start_x - 5, start_y - height + 10],
                "stroke": "#000",
                "strokeWidth": 2,
                "metadata": {"source": "assistant", "role": "axis_y_arrow", "axis_id": axis_id},
            }
        )
        actions.append(
            {
                "id": f"{axis_id}-y-arrow2",
                "kind": "line",
                "points": [start_x, start_y - height, start_x + 5, start_y - height + 10],
                "stroke": "#000",
                "strokeWidth": 2,
                "metadata": {"source": "assistant", "role": "axis_y_arrow", "axis_id": axis_id},
            }
        )

    # Axis labels
    actions.append(
        {
            "id": f"{axis_id}-label-x",
            "kind": "text",
            "x": start_x + width + 10,
            "y": start_y - 5,
            "text": label_x,
            "fontSize": 14,
            "fill": "#000000",
            "metadata": {"source": "assistant", "role": "axis_label_x", "axis_id": axis_id},
        }
    )
    actions.append(
        {
            "id": f"{axis_id}-label-y",
            "kind": "text",
            "x": start_x - 10,
            "y": start_y - height - 15,
            "text": label_y,
            "fontSize": 14,
            "fill": "#000000",
            "metadata": {"source": "assistant", "role": "axis_label_y", "axis_id": axis_id},
        }
    )

    return actions 