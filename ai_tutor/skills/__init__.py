from __future__ import annotations

from agents import function_tool
from ai_tutor.telemetry import log_tool
from agents.tool import FunctionTool as _FT
from .registration import add_tool_to_registry, list_tools

# Registry map: name  2 callable
# SKILL_REGISTRY: dict[str, callable] = {} # Replaced by _REGISTRY for tools

def skill(cost: str = "low", *ft_args, **ft_kwargs):
    """Decorator: (1) logs, (2) registers FunctionTool, (3) tracks cost."""
    def decorator(fn):
        wrapped = log_tool(fn)
        wrapped._skill_cost = cost  # noqa: SLF001 (used by ExecutorAgent)
        return wrapped
    return decorator

# # re-export helper for ExecutorAgent (legacy)
# def get_tool(name: str):
#     return SKILL_REGISTRY[name]

# --- Import all skill modules *after* defining the registry helpers ---

"""
Automatically import all skill modules in this package so they register with the tool registry.
"""
import importlib
import pkgutil
import inspect

print("--- Importing Skill Modules ---")
for finder, module_name, ispkg in pkgutil.iter_modules(__path__):
    if module_name != '__init__' and module_name != 'registration' and not ispkg:
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            print(f"  - Imported: {module_name}")
            # Auto-register FunctionTool instances
            for name, obj in inspect.getmembers(module):
                if isinstance(obj, _FT):
                    print(f"    - Registering tool: {obj.name} from {module_name}")
                    add_tool_to_registry(obj)
        except Exception as e:
            print(f"Error importing/registering skills from module '{module_name}': {e}")
print("--- Finished Importing Skill Modules ---")

__all__ = ['list_tools']

def list_tools():
    """Return every FunctionTool the Tutor Agent can call."""
    # Make sure all modules are imported before listing
    # (The loop above should handle this on initial load)
    return list(_REGISTRY.values()) 