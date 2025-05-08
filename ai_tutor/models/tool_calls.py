from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List

# --------------------------------------------------------------------------- #
# 1.  Define the set of front-end display tools that always exist
# --------------------------------------------------------------------------- #
_FE_TOOL_NAMES: List[str] = [
    "ask_question",     # Renders MCQ / input UI
    "explain",          # Renders text explanation
    "draw",             # Legacy SVG render (may be deprecated)
    "reflect",          # Internal thought process, no UI change
    "summarise_context",# Internal history management
    "end_session",      # Signals session completion
]

# --------------------------------------------------------------------------- #
# 2.  Dynamically load backend skill names registered via ai_tutor.skills
# --------------------------------------------------------------------------- #
try:
    from ai_tutor.skills import list_tools  # Local import to avoid circular deps in most cases

    # Importing ai_tutor.skills triggers auto-loading of all modules and their
    # decorators, so by the time we call list_tools() the registry should be
    # populated.  If, for any reason, skill loading fails we fall back to an
    # empty list – validation will then only allow FE tools.
    _BACKEND_TOOL_NAMES: List[str] = [tool.name for tool in list_tools()]
except Exception:  # pragma: no cover – defensive guard
    _BACKEND_TOOL_NAMES = []

# Merge and deduplicate while preserving order (FE tools first for readability)
_ALLOWED_TOOL_NAMES: List[str] = _FE_TOOL_NAMES + [
    name for name in _BACKEND_TOOL_NAMES if name not in _FE_TOOL_NAMES
]


class ToolCall(BaseModel):
    """Unified envelope produced by the lean Executor LLM each turn.

    The `name` field is validated at runtime against the union of:

    1. Front-end display tools (static list)
    2. All backend skills registered via `ai_tutor.skills`

    This approach avoids having to manually keep a giant Literal[...] in sync
    whenever new skills are added while still providing early validation and
    clear error messages if the LLM attempts to invoke an unknown tool.
    """

    name: str = Field(..., description="Tool/function the tutor wants to invoke")
    args: Dict[str, Any] = Field(default_factory=dict, description="JSON args for the tool")

    # ------------------------------ Validators ----------------------------- #
    @field_validator("name")
    @classmethod
    def _check_name_allowed(cls, v: str) -> str:
        if v not in _ALLOWED_TOOL_NAMES:
            raise ValueError(
                f"Unsupported tool name '{v}'. Allowed names: {_ALLOWED_TOOL_NAMES}"
            )
        return v 