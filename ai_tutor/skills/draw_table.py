from typing import List, Dict, Any, Sequence

from ai_tutor.skills import skill
from ai_tutor.skills.utils.layout import grid_positions


@skill
async def draw_table_actions(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    start_x: int = 50,
    start_y: int = 50,
    cell_width: int = 140,
    cell_height: int = 40,
    col_gap: int = 10,
    row_gap: int = 10,
    table_id: str = "table-1",
) -> List[Dict[str, Any]]:
    """Return a list of CanvasObjectSpec actions that draw a simple table.

    Args:
        headers: column header labels.
        rows: list of table rows (each a list of column strings).  Lengths do *not* have to be equal; shorter
              rows are padded with empty strings.
    """
    n_cols = len(headers)
    n_rows = len(rows) + 1  # +1 for header row

    positions = grid_positions(
        n_cols,
        n_rows,
        start_x=start_x,
        start_y=start_y,
        cell_width=cell_width,
        cell_height=cell_height,
        col_gap=col_gap,
        row_gap=row_gap,
    )

    actions: List[Dict[str, Any]] = []

    # Draw header cells (simple bold text)
    for c, header in enumerate(headers):
        x, y = positions[c]
        actions.append(
            {
                "id": f"{table_id}-header-{c}",
                "kind": "rect",
                "x": x,
                "y": y,
                "width": cell_width,
                "height": cell_height,
                "fill": "#BBDEFB",
                "stroke": "#0D47A1",
                "strokeWidth": 1,
                "metadata": {"source": "assistant", "role": "table_header", "table_id": table_id, "col": c},
            }
        )
        actions.append(
            {
                "id": f"{table_id}-header-{c}-text",
                "kind": "text",
                "x": x + 10,
                "y": y + cell_height / 2,
                "text": str(header),
                "fontSize": 14,
                "fill": "#0D47A1",
                "metadata": {"source": "assistant", "role": "table_header_text", "table_id": table_id, "col": c},
            }
        )

    # Draw body cells
    for r, row_values in enumerate(rows):
        for c in range(n_cols):
            x, y = positions[(r + 1) * n_cols + c]
            text_value = str(row_values[c]) if c < len(row_values) else ""
            actions.append(
                {
                    "id": f"{table_id}-cell-{r}-{c}",
                    "kind": "rect",
                    "x": x,
                    "y": y,
                    "width": cell_width,
                    "height": cell_height,
                    "fill": "#FFFFFF",
                    "stroke": "#9E9E9E",
                    "strokeWidth": 1,
                    "metadata": {"source": "assistant", "role": "table_cell", "table_id": table_id, "row": r, "col": c},
                }
            )
            actions.append(
                {
                    "id": f"{table_id}-cell-{r}-{c}-text",
                    "kind": "text",
                    "x": x + 10,
                    "y": y + cell_height / 2,
                    "text": text_value,
                    "fontSize": 14,
                    "fill": "#000000",
                    "metadata": {"source": "assistant", "role": "table_cell_text", "table_id": table_id, "row": r, "col": c},
                }
            )

    return actions 