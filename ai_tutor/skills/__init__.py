from __future__ import annotations
import inspect
import importlib
import pkgutil
from typing import Dict, List, Any

from agents import function_tool
from ai_tutor.telemetry import log_tool
from agents.tool import FunctionTool as _FT

# --- Define Registry and Helpers Directly Here ---
_REGISTRY: Dict[str, _FT] = {}

def add_tool_to_registry(tool: _FT):
    """Adds a FunctionTool instance to the registry."""
    if isinstance(tool, _FT):
        tool_name = getattr(tool, 'name', None) # Get the tool name reliably
        if not tool_name:
             print(f"Warning: Trying to register a tool without a name: {tool}")
             return
        if tool_name in _REGISTRY:
            print(f"Warning: Tool '{tool_name}' already exists in registry. Overwriting.")
        _REGISTRY[tool_name] = tool
        print(f"    - Registered tool: {tool_name}") # Log registration
    else:
        print(f"Warning: Attempted to register non-FunctionTool object: {type(tool)}")

def list_tools() -> List[_FT]:
    """Return every FunctionTool the Tutor Agent can call."""
    if not _REGISTRY:
        print("Warning: list_tools called but registry is empty. Ensure skills were imported and decorated correctly.")
    return list(_REGISTRY.values())

# --- Skill Decorator (unchanged) ---
def skill(cost: str = "low", *ft_args, **ft_kwargs):
    """Decorator: (1) logs, (2) registers FunctionTool, (3) tracks cost."""
    def decorator(fn):
        wrapped = log_tool(fn)
        wrapped._skill_cost = cost  # noqa: SLF001 (used by ExecutorAgent)
        return wrapped
    return decorator

# --- Dynamic Import Loop (calls local add_tool_to_registry) ---
print("--- Importing Skill Modules ---")
for finder, module_name, ispkg in pkgutil.iter_modules(__path__):
    if module_name != '__init__' and not ispkg:
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            print(f"  - Imported: {module_name}")
            for name, obj in inspect.getmembers(module):
                if isinstance(obj, _FT):
                    print(f"    - Found SDK FunctionTool: {obj.name}")
                    add_tool_to_registry(obj)
                elif inspect.isfunction(obj) and hasattr(obj, "__ai_function_spec__"):
                    print(f"    - Manually creating FunctionTool for: {name}")
                    tool_instance = _FT(python_fn=obj, name=name)
                    add_tool_to_registry(tool_instance)
        except Exception as e:
            print(f"Error importing/registering skills from module '{module_name}': {e}")
            import traceback
            traceback.print_exc()
print("--- Finished Importing Skill Modules ---")

__all__ = ['list_tools'] 