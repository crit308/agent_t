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

from typing import List, Dict, Any, Optional, Tuple, Literal

from ai_tutor.skills import skill  # Re-exported decorator

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


def _get_layout_position(w: int | None, h: int | None) -> Tuple[int, int]:
    """Very naïve placeholder until a real layout engine exists.

    For now we simply return a fixed offset so that something renders on the
    canvas.  In a later phase we will call a proper layout service.
    """
    return 100, 100


# --------------------------------------------------------------------------- #
# Public skills
# --------------------------------------------------------------------------- #

@skill
async def style_token(token: Literal[
    "default", "primary", "accent", "muted", "success", "error"
]) -> str:
    """Resolve a semantic *token* (e.g. ``"primary"``) to a hex colour string.

    This utility is intentionally synchronous-like but kept ``async`` for a
    consistent skill interface.
    """
    return _PALETTE.get(token, _PALETTE["default"])


@skill
async def draw_text(
    ctx,  # type: ignore[arg-type] – accepts TutorContext or RunContextWrapper via invoke()
    *,
    id: str,
    text: str,
    x: Optional[int] = None,
    y: Optional[int] = None,
    fontSize: Optional[int] = None,
    width: Optional[int] = None,
    color_token: Optional[str] = "default",
) -> Dict[str, Any]:
    """Return a *single* CanvasObjectSpec describing a text label."""

    if x is None or y is None:
        x, y = _get_layout_position(width, fontSize)

    # Use semantic colour token utility so FE palette remains in-sync.
    fill_colour = await style_token(token=color_token or "default")

    return {
        "id": id,
        "kind": "text",
        "x": x,
        "y": y,
        "text": text,
        **({"fontSize": fontSize} if fontSize else {}),
        **({"width": width} if width else {}),
        "fill": fill_colour,
        "metadata": {"source": "assistant", "id": id},
    }


@skill
async def draw_shape(
    ctx,  # type: ignore[arg-type]
    *,
    id: str,
    kind: Literal["rect", "circle", "arrow"],
    x: Optional[int] = None,
    y: Optional[int] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    radius: Optional[int] = None,
    points: Optional[List[Dict[str, int]]] = None,
    label: Optional[str] = None,
    color_token: Optional[str] = "default",
) -> List[Dict[str, Any]]:
    """Draw a primitive shape (rect, circle, arrow).  May return multiple specs."""

    if x is None or y is None:
        x, y = _get_layout_position(w, h or radius)

    # Use semantic colour token utility so FE palette remains in-sync.
    stroke_colour = await style_token(token=color_token or "default")

    actions: List[Dict[str, Any]] = []

    if kind == "rect":
        actions.append(
            {
                "id": id,
                "kind": "rect",
                "x": x,
                "y": y,
                "width": w or 100,
                "height": h or 60,
                "stroke": stroke_colour,
                "strokeWidth": 2,
                "fill": "#FFFFFF",
                "metadata": {"source": "assistant", "id": id},
            }
        )
    elif kind == "circle":
        actions.append(
            {
                "id": id,
                "kind": "circle",
                "x": x,
                "y": y,
                "radius": radius or 30,
                "stroke": stroke_colour,
                "strokeWidth": 2,
                "fill": "#FFFFFF",
                "metadata": {"source": "assistant", "id": id},
            }
        )
    elif kind == "arrow":
        # Represent arrow as a simple line for now.  If explicit *points* are
        # provided we flatten them; otherwise default to a horizontal arrow.
        if points:
            flat = [coord for pt in points for coord in (pt["x"], pt["y"])]
        else:
            flat = [x, y, x + (w or 60), y]
        actions.append(
            {
                "id": id,
                "kind": "line",
                "points": flat,
                "stroke": stroke_colour,
                "strokeWidth": 2,
                "metadata": {"source": "assistant", "id": id, "role": "arrow"},
            }
        )
    else:
        raise ValueError(f"Unsupported shape kind: {kind}")

    # Optional label just below the shape centre / rect top-left.
    if label:
        label_id = f"{id}-label"
        label_x = x + (w or radius or 0) / 2
        label_y = y + (h or radius or 0) + 20
        label_spec = {
            "id": label_id,
            "kind": "text",
            "x": label_x,
            "y": label_y,
            "text": label,
            "fontSize": 14,
            "fill": stroke_colour,
            "metadata": {"source": "assistant", "linked_to": id, "role": "label"},
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