import pytest
from unittest.mock import AsyncMock, patch

from ai_tutor.fsm import TutorFSM
from ai_tutor.context import TutorContext, UserModelState
from ai_tutor.agents.models import PlannerOutput, ActionSpec, FocusObjective

@pytest.mark.asyncio
async def test_explanation_to_quiz_flow():
    ctx = TutorContext(session_id="s1",
                       user_id="u1",
                       vector_store_id="vs1",
                       user_model_state=UserModelState())

    # Create a fake PlannerOutput for the test
    focus_obj = FocusObjective(
        topic="Test Topic",
        learning_goal="Test Goal",
        priority=5,
        relevant_concepts=["A"],
        suggested_approach=None,
        target_mastery=0.8,
        initial_difficulty="medium"
    )
    action_spec = ActionSpec(
        agent="teacher",
        params={"topic": "Test Topic", "explanation_details": "Segment 0"},
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

        # Use the FSM instead of the deprecated orchestrator
        fsm = TutorFSM(ctx)
        await fsm.on_user_message(None)

        planner.assert_called_once()
        teacher.assert_called_once()  # executor should call teacher on explain 