import pytest
import ai_tutor.tools
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext

# List of (tool_name, kwargs) pairs to test
TOOL_CASES = [
    ("call_quiz_teacher_evaluate", {"user_answer_index": 0}),
    ("update_explanation_progress", {"segment_index": 0}),
    ("update_user_model", {"topic": "test", "outcome": "correct"}),
    ("get_user_model_status", {}),
    ("reflect_on_interaction", {"topic": "test", "interaction_summary": "summary"}),
    ("call_planner_agent", {}),
    ("call_teacher_agent", {"topic": "test", "explanation_details": "details"}),
    ("call_quiz_creator_agent", {"topic": "test", "instructions": "instructions"}),
]

@pytest.mark.parametrize("tool_name,args", TOOL_CASES)
@pytest.mark.asyncio
async def test_tool_contract(tool_name, args):
    tool = getattr(ai_tutor.tools, tool_name)
    ctx = RunContextWrapper(TutorContext(session_id="dummy", user_id="dummy"))
    res = await tool(ctx, **args)
    assert res is not None 