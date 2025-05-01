from ai_tutor.skills import skill
from typing import List, Dict, Any

@skill
async def draw_diagram_actions(topic: str, description: str) -> List[Dict[str, Any]]:
    """
    Generates a list of CanvasObjectSpec dictionaries to draw a diagram for the given topic/description.
    This is a stub. In production, call an LLM or diagram model here.
    """
    # Example: Draw a labeled circle for the topic
    return [
        {
            "id": f"diagram-{topic}-circle",
            "kind": "circle",
            "x": 200,
            "y": 200,
            "radius": 60,
            "stroke": "#1976D2",
            "strokeWidth": 2,
            "fill": "#E3F2FD",
            "metadata": {"source": "assistant", "role": "main_shape", "topic": topic}
        },
        {
            "id": f"diagram-{topic}-label",
            "kind": "text",
            "x": 200,
            "y": 200,
            "text": topic,
            "fontSize": 20,
            "fill": "#1976D2",
            "metadata": {"source": "assistant", "role": "label", "topic": topic}
        }
    ] 