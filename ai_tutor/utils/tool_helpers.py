import json
from agents.run_context import RunContextWrapper
from typing import Any

async def invoke(tool, ctx, **kwargs) -> Any:
    """
    Uniformly call an Agents-SDK FunctionTool from plain Python.
    """
    payload_json = json.dumps(kwargs)
    ctx_wrapper = RunContextWrapper(ctx)
    return await tool.on_invoke_tool(ctx_wrapper, payload_json) 