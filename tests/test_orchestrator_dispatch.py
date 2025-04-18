import pytest
from unittest.mock import AsyncMock, patch

from ai_tutor.agents.orchestrator_agent import run_orchestrator
from ai_tutor.context import TutorContext, UserModelState

@pytest.mark.asyncio
async def test_explanation_to_quiz_flow():
    ctx = TutorContext(session_id="s1",
                       user_id="u1",
                       vector_store_id="vs1",
                       user_model_state=UserModelState())

    with patch("ai_tutor.tools.call_teacher_agent", new_callable=AsyncMock) as teacher,\
         patch("ai_tutor.tools.call_quiz_creator_agent", new_callable=AsyncMock) as quiz_creator:

        # pretend teacher finishes explaining
        teacher.return_value = "EXPLAIN_OK"
        quiz_creator.return_value = "QUIZ_OK"

        await run_orchestrator(ctx, last_event=None)

        teacher.assert_called_once()
        quiz_creator.assert_called_once()                # ← our guarantee
        assert ctx.user_model_state.pending_interaction_type == "checking_question" 