import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from ai_tutor.context import TutorContext
from ai_tutor.policy import InteractionEvent
from ai_tutor.agents.orchestrator_agent import run_orchestrator
import ai_tutor.tools.orchestrator_tools as tools
from ai_tutor.api_models import QuestionResponse, FeedbackResponse, MessageResponse
from ai_tutor.agents.models import QuizFeedbackItem, QuizQuestion


@pytest.mark.asyncio
async def test_loop_exits_on_ask_mcq(monkeypatch):
    """The orchestrator loop should exit only when an AskMCQ action is returned."""
    # Create a minimal TutorContext with required fields
    ctx = TutorContext(user_id=uuid4(), session_id=uuid4())
    # Define a sequence of policy actions: explain, evaluate, then ask_mcq
    seq = [
        {"type": "explain", "topic": "Fractions", "segment_index": 0},
        {"type": "evaluate", "user_answer_index": 1, "question_id": "q1"},
        {"type": "ask_mcq", "topic": "Fractions", "difficulty": "easy"},
    ]
    # Patch the policy choose_action to pop from our seq
    monkeypatch.setattr(
        "ai_tutor.policy.choose_action", 
        lambda *_: seq.pop(0)
    )
    # Stub teacher explanation tool
    monkeypatch.setattr(
        tools, "call_teacher_agent",
        AsyncMock(return_value=MessageResponse(
            response_type="message", text="stub exp", message_type="info"
        ))
    )
    # Stub evaluative feedback tool
    feedback_item = QuizFeedbackItem(
        question_index=0,
        question_text="Q?",
        user_selected_option="A",
        is_correct=True,
        correct_option="A",
        explanation="Good",
        improvement_suggestion=""
    )
    monkeypatch.setattr(
        tools, "call_quiz_teacher_evaluate",
        AsyncMock(return_value=FeedbackResponse(
            response_type="feedback",
            feedback=feedback_item,
            topic="Fractions",
            correct_answer="A",
            explanation="Good"
        ))
    )
    # Stub quiz creation tool to return a QuestionResponse with a valid QuizQuestion
    quiz_question = QuizQuestion(
        question="Sample question?",
        options=["A", "B", "C", "D"],
        correct_answer_index=0,
        explanation="Sample explanation",
        difficulty="easy",
        related_section="Fractions"
    )
    question_resp = QuestionResponse(
        response_type="question",
        question=quiz_question,
        topic="Fractions",
        context=None
    )
    monkeypatch.setattr(
        tools, "call_quiz_creator_agent",
        AsyncMock(return_value=question_resp)
    )

    # Run the orchestrator loop
    result = await run_orchestrator(
        ctx,
        last_event={"event_type": "system_tick", "data": {}}
    )

    # Verify we exited on ask_mcq and returned a QuestionResponse
    assert isinstance(result, QuestionResponse)
    assert result.response_type == "question" 