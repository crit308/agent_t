from agents.tool import FunctionTool as _FT
from typing import Dict, List, Any

_REGISTRY: Dict[str, _FT] = {}

def add_tool_to_registry(tool: _FT):
    """Adds a FunctionTool instance to the registry."""
    if isinstance(tool, _FT):
        if tool.name in _REGISTRY:
            print(f"Warning: Tool '{tool.name}' already exists in registry. Overwriting.")
        _REGISTRY[tool.name] = tool
    else:
        print(f"Warning: Attempted to register non-FunctionTool object: {type(tool)}")

def list_tools() -> List[_FT]:
    """Return every FunctionTool the Tutor Agent can call."""
    if not _REGISTRY:
        print("Warning: list_tools called but registry is empty. Ensure skills are imported.")
    return list(_REGISTRY.values()) 