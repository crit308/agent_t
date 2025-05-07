from ai_tutor.skills import skill
from typing import List, Dict, Any, TypedDict, Optional

class NodeSpec(TypedDict):
    id: str
    width: int
    height: int
    label: Optional[str]

class EdgeSpec(TypedDict):
    id: str
    source: str # Source node ID
    target: str # Target node ID
    label: Optional[str]

@skill
async def draw_graph(
    ctx,
    graph_id: str,
    nodes: List[NodeSpec],
    edges: List[EdgeSpec],
    layout_type: str = 'elk', # Default to elk
    x: Optional[int] = None, # Position for the whole graph group
    y: Optional[int] = None,
    xPct: Optional[float] = None,
    yPct: Optional[float] = None,
) -> Dict[str, Any]:
    """Generates the spec to automatically lay out and draw a graph."""
    spec = {
        "id": graph_id,
        "kind": "graph_layout",
        "metadata": {
            "id": graph_id,
            "layoutSpec": {
                "nodes": nodes,
                "edges": edges,
                "layoutType": layout_type,
            }
        },
         # Pass coordinate info for the top-left of the graph area
        **({"x": x} if x is not None else {}),
        **({"y": y} if y is not None else {}),
        **({"xPct": xPct} if xPct is not None else {}),
        **({"yPct": yPct} if yPct is not None else {}),
    }
    return {"type": "ADD_OBJECTS", "objects": [spec]} 