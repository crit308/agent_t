from ai_tutor.skills import skill
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from ai_tutor.exceptions import ToolInputError

class NodeSpec(BaseModel):
    id: str = Field(..., min_length=1)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    label: Optional[str] = None

class EdgeSpec(BaseModel):
    id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1) # Source node ID
    target: str = Field(..., min_length=1) # Target node ID
    label: Optional[str] = None

class DrawGraphArgs(BaseModel):
    graph_id: str = Field(..., min_length=1)
    nodes: List[NodeSpec]
    edges: List[EdgeSpec]
    layout_type: str = 'elk'
    x: Optional[int] = None
    y: Optional[int] = None
    xPct: Optional[float] = None
    yPct: Optional[float] = None

@skill
async def draw_graph(ctx: Any, **kwargs) -> Dict[str, Any]:
    """Generates the spec to automatically lay out and draw a graph."""
    try:
        validated_args = DrawGraphArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for draw_graph: {e}")

    spec = {
        "id": validated_args.graph_id,
        "kind": "graph_layout",
        "metadata": {
            "id": validated_args.graph_id,
            "layoutSpec": {
                # Pydantic models need to be converted to dicts for JSON serialization
                "nodes": [node.model_dump() for node in validated_args.nodes],
                "edges": [edge.model_dump() for edge in validated_args.edges],
                "layoutType": validated_args.layout_type,
            }
        },
        **({"x": validated_args.x} if validated_args.x is not None else {}),
        **({"y": validated_args.y} if validated_args.y is not None else {}),
        **({"xPct": validated_args.xPct} if validated_args.xPct is not None else {}),
        **({"yPct": validated_args.yPct} if validated_args.yPct is not None else {}),
    }
    return {"type": "ADD_OBJECTS", "objects": [spec]} 