import pytest
import ai_tutor.tools
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext
from agents.tool import FunctionTool
from agents.exceptions import ModelBehaviorError
from uuid import uuid4

# Collect all FunctionTool instances from ai_tutor.tools
TOOLS = [t for t in ai_tutor.tools.__dict__.values() if isinstance(t, FunctionTool)]

@pytest.mark.asyncio
@pytest.mark.parametrize("tool", TOOLS)
async def test_tool_schema(tool):
    # Build minimal context
    ctx = TutorContext(session_id=uuid4(), user_id=uuid4())
    wrapper = RunContextWrapper(ctx)
    payload = "{}"  # empty or stub
    try:
        result = await tool.on_invoke_tool(wrapper, payload)
    except ModelBehaviorError as e:
        pytest.fail(f"Tool {tool.name} raised ModelBehaviorError: {e}")
    assert result is not None 