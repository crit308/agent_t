import pytest
from ai_tutor.policy import choose_action, Explain, AskMCQ, Evaluate, Advance, InteractionEvent


class FakeContext:
    """Dummy context for testing choose_action stub."""
    pass


def test_policy_structures_exist():
    """Test that TypedDict classes for policy actions and events exist and accept the right keys."""
    # Instantiate each action dict
    exp: Explain = {"type": "explain", "topic": "TestTopic", "segment_index": 0, "style": "standard"}
    ask: AskMCQ = {"type": "ask_mcq", "topic": "TestTopic", "difficulty": "easy", "misconception_focus": None}
    eval_act: Evaluate = {"type": "evaluate", "user_answer_index": 1, "question_id": "qid123"}
    adv: Advance = {"type": "advance", "mastered_topic": "TestTopic", "next_focus": None}
    evt: InteractionEvent = {"event_type": "system_tick", "data": {}}

    assert isinstance(exp, dict)
    assert exp["type"] == "explain"
    assert ask["difficulty"] == "easy"
    assert eval_act["question_id"] == "qid123"
    assert adv["mastered_topic"] == "TestTopic"
    assert evt["event_type"] == "system_tick"


def test_choose_action_stub_not_implemented():
    """choose_action should raise NotImplementedError in its stub form."""
    ctx = FakeContext()
    evt: InteractionEvent = {"event_type": "system_tick", "data": {}}
    with pytest.raises(NotImplementedError):
        choose_action(ctx, evt) 