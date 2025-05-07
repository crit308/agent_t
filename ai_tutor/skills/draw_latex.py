from педагог.skill_library.skill import skill
from typing import Optional, List, Dict, Any

@skill
async def draw_latex(
    ctx: Any, # Added type hint for ctx
    latex_string: str,
    object_id: str, # Require an ID from the LLM
    x: Optional[int] = None,
    y: Optional[int] = None,
    xPct: Optional[float] = None,
    yPct: Optional[float] = None,
    # Add other optional layout params like color if needed
) -> Dict[str, Any]: # Returns ONE action spec to add ONE object
    """Generates the spec to render a LaTeX string on the whiteboard."""
    spec = {
        "id": object_id,
        "kind": "latex_svg", # Assuming FE can handle this kind
        "metadata": { "latex": latex_string, "id": object_id }, # Ensure id is also in metadata if FE needs it there
        # Pass coordinate info
        **({"x": x} if x is not None else {}),
        **({"y": y} if y is not None else {}),
        **({"xPct": xPct} if xPct is not None else {}),
        **({"yPct": yPct} if yPct is not None else {}),
    }
    # Return the action structure the FE dispatcher expects
    return {"type": "ADD_OBJECTS", "objects": [spec]} 