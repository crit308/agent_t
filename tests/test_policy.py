import pytest
from ai_tutor.policy import choose_action, Explain, AskMCQ, Evaluate, Advance, InteractionEvent
from ai_tutor.utils.difficulty import bloom_difficulty
import asyncio
from unittest.mock import MagicMock
from ai_tutor.tools import call_planner_agent


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


@pytest.mark.parametrize("m,pace,exp", [(0.2,1,"easy"), (0.4,1,"medium"), (0.4,1.3,"hard"), (0.8,0.6,"medium")])
def test_bloom_difficulty(m, pace, exp):
    assert bloom_difficulty(m, pace) == exp 


@pytest.mark.asyncio
def test_call_planner_agent_excludes_mastered_topics(monkeypatch):
    class DummyConcept:
        def __init__(self, mastery, confidence):
            self.mastery = mastery
            self.confidence = confidence
    class DummyUserModelState:
        def __init__(self):
            self.concepts = {
                'A': DummyConcept(0.9, 6),
                'B': DummyConcept(0.7, 6),
                'C': DummyConcept(0.85, 4),
                'D': DummyConcept(0.81, 5),
            }
    class DummyContext:
        def __init__(self):
            self.user_model_state = DummyUserModelState()
            self.session_goal = 'goal'
            self.vector_store_id = 'v'
            self.session_id = 's'
    class DummyRunContextWrapper:
        def __init__(self):
            self.context = DummyContext()
    # Patch create_planner_agent and Runner.run to capture payload
    monkeypatch.setattr('ai_tutor.agents.planner_agent.create_planner_agent', lambda x: None)
    captured = {}
    async def dummy_run(agent, prompt, context, run_config):
        captured['prompt'] = prompt
        return type('Result', (), {'final_output': 'dummy'})()
    monkeypatch.setattr('ai_tutor.tools.Runner.run', dummy_run)
    # Run
    ctx = DummyRunContextWrapper()
    asyncio.get_event_loop().run_until_complete(call_planner_agent(ctx))
    # Check that only D and A are excluded (mastery >= 0.8 and confidence >= 5)
    assert 'D' in captured['prompt'] and 'A' in captured['prompt']
    assert 'B' not in captured['prompt'] and 'C' not in captured['prompt'] 