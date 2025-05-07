from ai_tutor.skills import skill
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from ai_tutor.exceptions import ToolInputError

class DrawLatexArgs(BaseModel):
    latex_string: str = Field(..., min_length=1)
    object_id: str = Field(..., min_length=1)
    x: Optional[int] = None
    y: Optional[int] = None
    xPct: Optional[float] = None
    yPct: Optional[float] = None
    # Add other optional layout params like color if needed

@skill
async def draw_latex(ctx: Any, **kwargs) -> Dict[str, Any]:
    """Generates the spec to render a LaTeX string on the whiteboard."""
    try:
        validated_args = DrawLatexArgs(**kwargs)
    except ValidationError as e:
        # Option B: Raise specific exception caught by Runner
        raise ToolInputError(f"Invalid arguments for draw_latex: {e}")

    spec = {
        "id": validated_args.object_id,
        "kind": "latex_svg",
        "metadata": { "latex": validated_args.latex_string, "id": validated_args.object_id },
        **({"x": validated_args.x} if validated_args.x is not None else {}),
        **({"y": validated_args.y} if validated_args.y is not None else {}),
        **({"xPct": validated_args.xPct} if validated_args.xPct is not None else {}),
        **({"yPct": validated_args.yPct} if validated_args.yPct is not None else {}),
    }
    return {"type": "ADD_OBJECTS", "objects": [spec]} 