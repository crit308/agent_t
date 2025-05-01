from ai_tutor.skills import skill
from typing import Dict

@skill
async def explain_diagram_part(object_id: str) -> str:
    """Return a deeper explanation associated with a clicked diagram object.

    For MVP this is a stub that turns the object_id into a readable message.
    Future versions could map IDs to knowledge or parse metadata.
    """
    # Simple heuristic: strip prefixes/suffixes
    topic = object_id.replace("diagram-", "").split("-")[0]
    return f"Let's dive deeper into {topic}: … (detailed explanation placeholder)." 