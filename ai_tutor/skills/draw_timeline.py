from ai_tutor.skills import skill
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, conlist, ValidationError
from ai_tutor.exceptions import ToolInputError

class EventSpec(BaseModel):
    date: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)

class DrawTimelineArgs(BaseModel):
    timeline_id: str = Field(..., min_length=1)
    events: conlist(EventSpec, min_length=1) # Must have at least one event
    start_x: int = 50
    start_y: int = 150
    length: int = Field(default=600, gt=0) # Length must be positive
    event_label_offset_y: int = 20
    tick_height: int = Field(default=5, ge=0) # Tick height can be 0 but not negative
    line_color: str = "black"
    line_stroke_width: int = Field(default=2, gt=0)
    tick_color: str = "black"
    tick_stroke_width: int = Field(default=1, ge=0)
    label_font_size: int = Field(default=10, gt=0)
    label_color: str = "black"

@skill
async def draw_timeline(ctx: Any, **kwargs) -> Dict[str, Any]:
    """Generates specs to draw a horizontal timeline with events."""
    try:
        args = DrawTimelineArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for draw_timeline: {e}")

    objects: List[Dict[str, Any]] = []

    # The conlist(min_length=1) for events ensures this condition is met by Pydantic
    # if not args.events:
    #     return {"type": "ADD_OBJECTS", "objects": []}

    # --- Main Timeline Line ---
    objects.append({
        "id": f"{args.timeline_id}-mainline",
        "kind": "line",
        "points": [args.start_x, args.start_y, args.start_x + args.length, args.start_y],
        "stroke": args.line_color,
        "strokeWidth": args.line_stroke_width,
        "metadata": {"id": f"{args.timeline_id}-mainline", "groupId": args.timeline_id, "role": "timeline-axis"}
    })

    num_events = len(args.events)
    # spacing division by num_events + 1 handles num_events=0 if it were possible, 
    # but Pydantic's conlist(min_length=1) ensures num_events >= 1.
    # If num_events is 1, spacing is length / 2. If 2, length / 3 etc.
    spacing = args.length / (num_events + 1)

    for i, event_spec in enumerate(args.events):
        event_x = args.start_x + (i + 1) * spacing
        event_id_base = f"{args.timeline_id}-event-{i}"

        # --- Event Tick ---
        objects.append({
            "id": f"{event_id_base}-tick",
            "kind": "line",
            "points": [event_x, args.start_y - args.tick_height, event_x, args.start_y + args.tick_height],
            "stroke": args.tick_color,
            "strokeWidth": args.tick_stroke_width,
            "metadata": {"id": f"{event_id_base}-tick", "groupId": args.timeline_id, "role": "timeline-tick"}
        })

        label_y = args.start_y - args.event_label_offset_y if i % 2 == 0 else args.start_y + args.event_label_offset_y + args.label_font_size

        objects.append({
            "id": f"{event_id_base}-label",
            "kind": "textbox",
            "text": event_spec.label, # Access Pydantic model field
            "x": event_x - 20, 
            "y": label_y,
            "fontSize": args.label_font_size,
            "fill": args.label_color,
            "metadata": {"id": f"{event_id_base}-label", "groupId": args.timeline_id, "role": "timeline-label", "date": event_spec.date} # Access Pydantic model field
        })
        
        date_label_y = label_y + args.label_font_size + 2
        if i % 2 == 0: 
             date_label_y = args.start_y - args.event_label_offset_y + args.label_font_size + 2
        else: 
             date_label_y = args.start_y + args.event_label_offset_y + args.label_font_size + args.label_font_size + 4

        objects.append({
            "id": f"{event_id_base}-date",
            "kind": "textbox",
            "text": f"({event_spec.date})", # Access Pydantic model field
            "x": event_x - 15, 
            "y": date_label_y,
            "fontSize": args.label_font_size - 2 if args.label_font_size > 2 else 1, # Ensure font size is positive
            "fill": args.label_color,
            "metadata": {"id": f"{event_id_base}-date", "groupId": args.timeline_id, "role": "timeline-date"}
        })

    return {"type": "ADD_OBJECTS", "objects": objects} 