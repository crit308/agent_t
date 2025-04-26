from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, Dict, List

from agents.tool import FunctionTool as _FT
from ai_tutor.telemetry import log_tool


# --------------------------------------------------------------------------- #
# 1.  In-memory registry (used by ExecutorAgent and tests)
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, Callable] = {}
SKILL_REGISTRY = _REGISTRY          # <- exported for tests


# --------------------------------------------------------------------------- #
# 2.  Smart decorator – works as @skill or @skill(cost="high")
# --------------------------------------------------------------------------- #
def skill(_fn: Callable | None = None, *, cost: str = "low"):
    """
    Decorator for registering an async skill.

    • Use as @skill                 → defaults to cost="low"
    • Or  @skill(cost="medium")     → sets custom cost tag

    The wrapped function is:
    * telemetry-logged
    * annotated with _skill_cost
    * inserted into SKILL_REGISTRY
    """
    def _wrap(fn: Callable) -> Callable:
        wrapped = log_tool(fn)
        wrapped._skill_cost = cost
        _REGISTRY[wrapped.__name__] = wrapped
        return wrapped

    # Called as @skill
    if callable(_fn):
        return _wrap(_fn)

    # Called as @skill(...)
    return _wrap


# --------------------------------------------------------------------------- #
# 3.  Convenience helper used by Planner / Executor
# --------------------------------------------------------------------------- #
def list_tools() -> List[_FT]:
    """Return every FunctionTool that has been auto-registered."""
    return [v for v in _REGISTRY.values() if isinstance(v, _FT) or callable(v)]


# --------------------------------------------------------------------------- #
# 4.  Auto-import every module under skills/ so their decorators run
# --------------------------------------------------------------------------- #
for *_ , module_name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg and module_name != "__init__":
        importlib.import_module(f".{module_name}", package=__name__) 