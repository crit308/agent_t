from pydantic import BaseModel

class ToolExecutionError(RuntimeError):
    """Raise inside any tool when the call failed but the process should continue."""
    def __init__(self, detail: str, *, code: str | None = None):
        super().__init__(detail)
        self.detail = detail
        self.code = code or "tool_error"

class ErrorResponse(BaseModel):
    response_type: str = "error"
    tool: str
    detail: str
    code: str 