import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from ai_tutor.context import TutorContext
from ai_tutor.agents.orchestrator_agent import run_orchestrator
from ai_tutor.agents.models import PlannerOutput, ActionSpec, FocusObjective

@pytest.mark.asyncio
async def test_orchestrator_calls_subagent():
    ctx = TutorContext(user_id=uuid4(), session_id=uuid4())
    # Create a fake PlannerOutput for the test
    focus_obj = FocusObjective(
        topic="Fractions",
        learning_goal="Understand fractions",
        priority=5,
        relevant_concepts=["numerator", "denominator"],
        suggested_approach=None,
        target_mastery=0.8,
        initial_difficulty="easy"
    )
    action_spec = ActionSpec(
        agent="teacher",
        params={"topic": "Fractions", "explanation_details": "Segment 0"},
        success_criteria="delivered",
        max_steps=1
    )
    planner_output = PlannerOutput(objective=focus_obj, next_action=action_spec)

    with (
        patch("ai_tutor.tools.call_planner_agent", new_callable=AsyncMock) as planner,
        patch("ai_tutor.tools.call_teacher_agent", new_callable=AsyncMock) as teacher
    ):
        planner.return_value = planner_output
        teacher.return_value = "EXPLAIN_OK"

        result = await run_orchestrator(ctx, last_event={"event_type": "system_tick", "data": {}})

        planner.assert_called_once()
        teacher.assert_called_once()
        assert result == "EXPLAIN_OK"

def test_all_tools_validate():
    import ai_tutor.tools.orchestrator_tools as tools
    from agents.utils import build_openai_schema
    for tool in tools.__all__:
        build_openai_schema(getattr(tools, tool))  # raises on error 