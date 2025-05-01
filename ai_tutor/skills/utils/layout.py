# Utility functions for laying out whiteboard objects
from typing import List, Tuple


def grid_positions(
    n_cols: int,
    n_rows: int,
    *,
    start_x: int = 50,
    start_y: int = 50,
    cell_width: int = 120,
    cell_height: int = 40,
    col_gap: int = 20,
    row_gap: int = 20,
) -> List[Tuple[int, int]]:
    """Return a list of (x, y) positions for a simple grid layout.

    The positions are returned row-major.  They correspond to the *top-left* corner of each cell.
    """
    positions: List[Tuple[int, int]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = start_x + c * (cell_width + col_gap)
            y = start_y + r * (cell_height + row_gap)
            positions.append((x, y))
    return positions


def flow_positions(
    n_items: int,
    *,
    start_x: int = 50,
    start_y: int = 50,
    box_width: int = 140,
    box_height: int = 60,
    h_gap: int = 80,
) -> List[Tuple[int, int]]:
    """Return positions for a horizontal flow layout suitable for simple flowcharts."""
    return [
        (start_x + i * (box_width + h_gap), start_y) for i in range(n_items)
    ] 