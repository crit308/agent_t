import json
from agents.run_context import RunContextWrapper
from typing import Any
from agents.tool import FunctionTool

async def invoke(tool, ctx, **kwargs) -> Any:
    """
    Uniformly call an Agents-SDK FunctionTool from plain Python.
    """
    # If this is a FunctionTool, use its on_invoke_tool interface
    if isinstance(tool, FunctionTool):
        payload_json = json.dumps(kwargs)
        ctx_wrapper = RunContextWrapper(ctx)
        return await tool.on_invoke_tool(ctx_wrapper, payload_json)
    # Otherwise, assume it's a direct stub or coroutine function and invoke with ctx and kwargs
    return await tool(ctx, **kwargs) 