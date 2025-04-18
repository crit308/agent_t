from typing import Any

from .tool import FunctionTool
from .models.openai_chatcompletions import ToolConverter
from .handoffs import Handoff


def build_openai_schema(tool: Any) -> dict[str, Any]:
    """
    Convert a FunctionTool or Handoff into an OpenAI function schema dictionary.
    """
    if isinstance(tool, FunctionTool):
        return ToolConverter.to_openai(tool)
    if isinstance(tool, Handoff):
        return ToolConverter.convert_handoff_tool(tool)
    # Fallback to attempting conversion as a FunctionTool
    return ToolConverter.to_openai(tool) 