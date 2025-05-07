from __future__ import annotations

"""Generic low-level drawing helper skills.

These skills expose primitive whiteboard drawing capabilities (text, basic
shapes, colour tokens …) that higher-level tutors/agents can reuse when they
need *just* a textbox or rectangle rather than a full diagram/MCQ/table.

Phase-1 MVP keeps the logic intentionally simple:
• If the caller does not supply coordinates we fall back to a fixed layout
  stub so nothing crashes while a proper layout service is still under
  development.
• We do *not* attempt to detect collisions/overlap – that will be refined in
  Phase-2.

All functions are registered as ADK FunctionTools via the ``@skill`` decorator
from :pymod:`ai_tutor.skills`.
"""

from typing import List, Dict, Any, Optional, Tuple, Literal as TypingLiteral

from ai_tutor.skills import skill  # Re-exported decorator
from ai_tutor.exceptions import ToolInputError
from pydantic import BaseModel, Field, ValidationError

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

# Simple static colour palette that loosely mirrors the FE theme.
_PALETTE: dict[str, str] = {
    "default": "#000000",
    "primary": "#1976D2",  # Blue-600
    "accent": "#FF5722",   # Deep-orange-500
    "muted": "#9E9E9E",    # Grey-500
    "success": "#2ECC71",  # Green-400
    "error": "#E74C3C",    # Red-400
}

# Define a Literal type for Pydantic validation based on _PALETTE keys
PaletteColor = TypingLiteral["default", "primary", "accent", "muted", "success", "error"]

def _get_layout_position(w: int | None, h: int | None) -> Tuple[int, int]:
    """Very naïve placeholder until a real layout engine exists.

    For now we simply return a fixed offset so that something renders on the
    canvas.  In a later phase we will call a proper layout service.
    """
    return 100, 100


# --------------------------------------------------------------------------- #
# Pydantic Models for Skill Arguments
# --------------------------------------------------------------------------- #

class StyleTokenArgs(BaseModel):
    token: PaletteColor

class DrawTextArgs(BaseModel):
    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    x: Optional[int] = None
    y: Optional[int] = None
    fontSize: Optional[int] = Field(default=None, gt=0)
    width: Optional[int] = Field(default=None, gt=0)
    color_token: PaletteColor = "default"

class PointSpec(BaseModel):
    x: int
    y: int

class DrawShapeArgs(BaseModel):
    id: str = Field(..., min_length=1)
    kind: TypingLiteral["rect", "circle", "arrow"]
    x: Optional[int] = None
    y: Optional[int] = None
    w: Optional[int] = Field(default=None, gt=0)
    h: Optional[int] = Field(default=None, gt=0)
    radius: Optional[int] = Field(default=None, gt=0)
    points: Optional[List[PointSpec]] = None # List of Pydantic PointSpec models
    label: Optional[str] = None
    color_token: PaletteColor = "default"

# --------------------------------------------------------------------------- #
# Public skills
# --------------------------------------------------------------------------- #

@skill
async def style_token(**kwargs) -> str:
    """Resolve a semantic *token* (e.g. ``"primary"``) to a hex colour string.

    This utility is intentionally synchronous-like but kept ``async`` for a
    consistent skill interface.
    """
    try:
        validated_args = StyleTokenArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for style_token: {e}")
    return _PALETTE.get(validated_args.token, _PALETTE["default"])


@skill
async def draw_text(ctx: Any, **kwargs) -> Dict[str, Any]:
    """Return a *single* CanvasObjectSpec describing a text label."""
    try:
        args = DrawTextArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for draw_text: {e}")

    x_coord, y_coord = args.x, args.y
    if x_coord is None or y_coord is None:
        x_coord, y_coord = _get_layout_position(args.width, args.fontSize)

    # Use semantic colour token utility so FE palette remains in-sync.
    fill_colour = await style_token(token=args.color_token)

    return {
        "id": args.id,
        "kind": "text",
        "x": x_coord,
        "y": y_coord,
        "text": args.text,
        **({"fontSize": args.fontSize} if args.fontSize else {}),
        **({"width": args.width} if args.width else {}),
        "fill": fill_colour,
        "metadata": {"source": "assistant", "id": args.id},
    }


@skill
async def draw_shape(ctx: Any, **kwargs) -> List[Dict[str, Any]]:
    """Draw a primitive shape (rect, circle, arrow). May return multiple specs."""
    try:
        args = DrawShapeArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for draw_shape: {e}")

    x_coord, y_coord = args.x, args.y
    if x_coord is None or y_coord is None:
        x_coord, y_coord = _get_layout_position(args.w, args.h or args.radius)

    # Use semantic colour token utility so FE palette remains in-sync.
    stroke_colour = await style_token(token=args.color_token)

    actions: List[Dict[str, Any]] = []

    if args.kind == "rect":
        actions.append(
            {
                "id": args.id,
                "kind": "rect",
                "x": x_coord,
                "y": y_coord,
                "width": args.w or 100,
                "height": args.h or 60,
                "stroke": stroke_colour,
                "strokeWidth": 2,
                "fill": "#FFFFFF",
                "metadata": {"source": "assistant", "id": args.id},
            }
        )
    elif args.kind == "circle":
        actions.append(
            {
                "id": args.id,
                "kind": "circle",
                "x": x_coord,
                "y": y_coord,
                "radius": args.radius or 30,
                "stroke": stroke_colour,
                "strokeWidth": 2,
                "fill": "#FFFFFF",
                "metadata": {"source": "assistant", "id": args.id},
            }
        )
    elif args.kind == "arrow":
        flat_points: List[int] = []
        if args.points:
            for pt_spec in args.points: # pt_spec is a PointSpec model
                flat_points.extend([pt_spec.x, pt_spec.y])
        else:
            flat_points = [x_coord, y_coord, x_coord + (args.w or 60), y_coord] # Default arrow if no points
        
        actions.append(
            {
                "id": args.id,
                "kind": "line", # Arrows are represented as lines
                "points": flat_points,
                "stroke": stroke_colour,
                "strokeWidth": 2,
                "metadata": {"source": "assistant", "id": args.id, "role": "arrow"},
            }
        )
    else:
        raise ValueError(f"Unsupported shape kind: {args.kind}")

    # Optional label just below the shape centre / rect top-left.
    if args.label:
        label_id = f"{args.id}-label"
        label_x_val = x_coord + (args.w or args.radius or 0) / 2
        label_y_val = y_coord + (args.h or args.radius or 0) + 20
        label_spec = {
            "id": label_id,
            "kind": "text",
            "x": label_x_val,
            "y": label_y_val,
            "text": args.label,
            "fontSize": 14,
            "fill": stroke_colour,
            "metadata": {"source": "assistant", "linked_to": args.id, "role": "label"},
        }
        actions.append(label_spec)

    return actions


# --------------------------------------------------------------------------- #
# *Alias* for clearing the board so older/other code can still import the
# existing skill name ``clear_whiteboard`` while newer code uses ``clear_board``.
# --------------------------------------------------------------------------- #

from ai_tutor.skills.clear_whiteboard import clear_whiteboard as _clear_whiteboard_existing


@skill(name_override="clear_board")
async def clear_board() -> List[Dict[str, Any]]:  # noqa: D401 – simple verb
    """Return a single **reset** CanvasObjectSpec to wipe previous drawings."""

    return await _clear_whiteboard_existing()  # Re-use implementation 