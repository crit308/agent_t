from __future__ import annotations

from agents import function_tool
from ai_tutor.telemetry import log_tool
from agents.tool import FunctionTool as _FT

# Registry map: name  2 callable
# SKILL_REGISTRY: dict[str, callable] = {} # Replaced by _REGISTRY for tools
_REGISTRY = {}

def skill(cost: str = "low", *ft_args, **ft_kwargs):
    """Decorator: (1) logs, (2) registers FunctionTool, (3) tracks cost."""

    def decorator(fn):
        wrapped = log_tool(fn)

        # Attach metadata
        wrapped._skill_cost = cost  # noqa: SLF001 (used by ExecutorAgent)

        # Register with Agents SDK (legacy, keep for now? Or rely on @as_tool?)
        # function_tool(strict_mode=True, *ft_args, **ft_kwargs)(wrapped)

        # Add to python-side registry (legacy, keep for now?)
        # SKILL_REGISTRY[wrapped.__name__] = wrapped
        return wrapped

    return decorator

# # re-export helper for ExecutorAgent (legacy)
# def get_tool(name: str):
#     return SKILL_REGISTRY[name]

# --- Import all skill modules *after* defining the registry helpers ---

"""
Automatically import all skill modules in this package so they register with the tool registry.
"""
import importlib, pkgutil
# __path__ = __import__(__name__).__path__  # REMOVE THIS LINE. __path__ is implicitly available.
for finder, module_name, ispkg in pkgutil.iter_modules(__path__): # Use the existing __path__
    # Add a check to prevent importing __init__ or the package itself recursively
    if module_name != '__init__' and not ispkg: # Only import .py files, not subdirectories
        try:
            # Use relative import within the package
            importlib.import_module(f".{module_name}", package=__name__)
        except Exception as e:
            # Add some logging/printing to see if specific skills fail to import
            print(f"Error importing skill module '{module_name}': {e}")

# All modules under ai_tutor/skills are now imported and registered

def as_tool(fn=None, *, name: str | None = None):
    """Decorator to register a skill function as an agents.tool.FunctionTool."""
    def decorator(func):
        tool = _FT(python_fn=func, name=name or func.__name__)
        _REGISTRY[tool.name] = tool
        # If the original function was also decorated with @skill, ensure it's still callable
        # This might need adjustment based on how @skill and @as_tool interact
        return func # Return the original function, tool is registered separately
    return decorator(fn) if fn else decorator

def list_tools():
    """Return every FunctionTool the Tutor Agent can call."""
    # Make sure all modules are imported before listing
    # (The loop above should handle this on initial load)
    return list(_REGISTRY.values()) 