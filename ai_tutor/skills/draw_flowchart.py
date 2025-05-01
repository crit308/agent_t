from typing import List, Dict, Any, Sequence

from ai_tutor.skills import skill
from ai_tutor.skills.utils.layout import flow_positions


@skill
async def draw_flowchart_actions(
    steps: Sequence[str],
    *,
    start_x: int = 50,
    start_y: int = 50,
    box_width: int = 140,
    box_height: int = 60,
    h_gap: int = 80,
    chart_id: str = "flow-1",
) -> List[Dict[str, Any]]:
    """Return a list of actions drawing a simple left-to-right flowchart."""
    positions = flow_positions(
        len(steps),
        start_x=start_x,
        start_y=start_y,
        box_width=box_width,
        h_gap=h_gap,
    )

    actions: List[Dict[str, Any]] = []

    # Draw boxes
    for i, (x, y) in enumerate(positions):
        actions.append(
            {
                "id": f"{chart_id}-box-{i}",
                "kind": "rect",
                "x": x,
                "y": y,
                "width": box_width,
                "height": box_height,
                "fill": "#E8F5E9",
                "stroke": "#1B5E20",
                "strokeWidth": 1,
                "metadata": {"source": "assistant", "role": "flow_box", "chart_id": chart_id, "step": i},
            }
        )
        actions.append(
            {
                "id": f"{chart_id}-box-{i}-text",
                "kind": "text",
                "x": x + box_width / 2,
                "y": y + box_height / 2,
                "text": steps[i],
                "fontSize": 14,
                "fill": "#1B5E20",
                "textAnchor": "middle",
                "metadata": {"source": "assistant", "role": "flow_box_text", "chart_id": chart_id, "step": i},
            }
        )

    # Draw arrows between boxes
    for i in range(len(steps) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        actions.append(
            {
                "id": f"{chart_id}-arrow-{i}-{i+1}",
                "kind": "line",
                "points": [x1 + box_width, y1 + box_height / 2, x2 - 10, y2 + box_height / 2],
                "stroke": "#000000",
                "strokeWidth": 2,
                "metadata": {"source": "assistant", "role": "flow_arrow", "chart_id": chart_id, "from": i, "to": i + 1},
            }
        )

    return actions 