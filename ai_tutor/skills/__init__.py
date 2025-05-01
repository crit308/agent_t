from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, Dict, List, Any
import logging
from agents import function_tool
from agents.tool import FunctionTool as ADKFunctionTool

from ai_tutor.telemetry import log_tool

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 1.  In-memory registry (used by ExecutorAgent and tests)
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, ADKFunctionTool] = {}
SKILL_REGISTRY = _REGISTRY          # <- exported for tests


# --------------------------------------------------------------------------- #
# 2.  Smart decorator – works as @skill or @skill(cost="high")
# --------------------------------------------------------------------------- #
def skill(_fn: Callable | None = None, *, cost: str = "low", name_override: str | None = None):
    """
    Decorator for registering an async skill using ADK's @function_tool.

    - Wraps the function with ADK's @function_tool for SDK compatibility.
    - Optionally stores the resulting FunctionTool object in a local registry.
    - Adds a custom _skill_cost attribute (ADK doesn't use this).
    """
    def decorator(fn: Callable) -> ADKFunctionTool:
        tool_name = name_override or fn.__name__
        adk_tool_instance = function_tool(name_override=tool_name, strict_mode=False)(fn)

        setattr(adk_tool_instance, '_skill_cost', cost)

        _REGISTRY[adk_tool_instance.name] = adk_tool_instance
        logger.debug(f"Registered skill '{adk_tool_instance.name}' as ADK FunctionTool.")

        # Ensure invoke() can access the wrapped coroutine even if the ADK FunctionTool
        # implementation does not expose it. The `invoke` helper expects a
        # `__original_func__` attribute pointing to the underlying async def.
        # Attach it once here for every tool created so downstream callers
        # (e.g. planner_agent.determine_session_focus) work reliably.
        if not hasattr(adk_tool_instance, "__original_func__"):
            setattr(adk_tool_instance, "__original_func__", fn)

        return adk_tool_instance

    if callable(_fn):
        return decorator(_fn)
    else:
        return decorator


# --------------------------------------------------------------------------- #
# 3.  Convenience helper used by Planner / Executor
# --------------------------------------------------------------------------- #
def list_tools() -> List[ADKFunctionTool]:
    """Return all registered ADK FunctionTool skills."""
    return list(_REGISTRY.values())


# --------------------------------------------------------------------------- #
# 4.  Auto-import every module under skills/ so their decorators run
# --------------------------------------------------------------------------- #
for *_ , module_name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg and module_name != "__init__":
        importlib.import_module(f".{module_name}", package=__name__)

def auto_import_skills():
    """Dynamically imports skills from modules in the same directory."""
    logger.debug("Attempting to auto-import skills...")
    package = __name__.split('.')[0]
    package_dir = __path__

    for _, module_name, _ in pkgutil.iter_modules(package_dir):
        full_module_path = f"{package}.skills.{module_name}"
        if full_module_path != __name__:
            try:
                module = importlib.import_module(full_module_path)
                logger.debug(f"Successfully imported module: {full_module_path}")
            except ImportError as e:
                logger.error(f"Failed to import skill module '{full_module_path}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error importing '{full_module_path}': {e}", exc_info=True)
    logger.debug(f"Finished auto-importing skills. Registry size: {len(_REGISTRY)}")

auto_import_skills()

# Export commonly used skills for convenience
from .draw_diagram import draw_diagram_actions
from .draw_mcq import draw_mcq_actions
from .draw_mcq_feedback import draw_mcq_feedback
from .clear_whiteboard import clear_whiteboard
from .draw_table import draw_table_actions
from .draw_flowchart import draw_flowchart_actions
from .draw_axis import draw_axis_actions
from .explain_diagram_part import explain_diagram_part 