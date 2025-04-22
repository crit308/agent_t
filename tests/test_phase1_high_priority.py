import pytest
from uuid import UUID
from ai_tutor.context import TutorContext, UserModelState, UserConceptMastery
from ai_tutor.core.enums import ExecutorStatus
from ai_tutor.agents.models import FocusObjective
from ai_tutor.agents.executor_agent import ExecutorAgent
from ai_tutor.fsm import TutorFSM
from ai_tutor.agents.planner_agent import run_planner
from ai_tutor.core.schema import PlannerOutput

@ pytest.mark.asyncio
async def test_planner_output_schema():
    """PlannerAgent should return a valid PlannerOutput with at least one objective"""
    ctx = TutorContext(session_id=UUID(int=0), user_id='test_user')
    output = await run_planner(ctx)
    assert isinstance(output, PlannerOutput)
    assert output.objectives and len(output.objectives) >= 1

@ pytest.mark.asyncio
async def test_executor_continue_and_completed_states():
    """ExecutorAgent.run should return CONTINUE when mastery below target, COMPLETED when meeting target"""
    base_ctx = TutorContext(session_id=UUID(int=0), user_id='test_user')
    objective = FocusObjective(topic='topic1', learning_goal='Test', target_mastery=0.8)
    # No mastery -> CONTINUE
    status, result = await ExecutorAgent.run(objective, base_ctx)
    assert status == ExecutorStatus.CONTINUE
    # Simulate user mastery above target
    base_ctx.user_model_state.concepts['topic1'] = UserConceptMastery(alpha=8, beta=2)
    status, result = await ExecutorAgent.run(objective, base_ctx)
    assert status == ExecutorStatus.COMPLETED

@ pytest.mark.asyncio
async def test_fsm_transitions_continue_to_awaiting():
    """TutorFSM should move to 'awaiting_user' state on CONTINUE"""
    ctx = TutorContext(session_id=UUID(int=0), user_id='test_user')
    fsm = TutorFSM(ctx)
    # Trigger first transition: planning -> executing -> CONTINUE
    result = await fsm.on_user_message({'event_type': 'start', 'data': {}})
    assert ctx.state == 'awaiting_user'
    # Ensure result from FSM is the executor output
    assert result is not None 