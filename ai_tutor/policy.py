from typing import TypedDict, Literal, Union, Optional
from ai_tutor.context import TutorContext
from ai_tutor.agents.models import FocusObjective
import random
from ai_tutor.dependencies import SUPABASE_CLIENT  # Supabase client for policy weights
from ai_tutor.utils.difficulty import bloom_difficulty


class Explain(TypedDict):
    type: Literal["explain"]
    topic: str
    segment_index: int
    style: Optional[Literal["standard", "analogy", "code", "visual"]]


class AskMCQ(TypedDict):
    type: Literal["ask_mcq"]
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    misconception_focus: Optional[str]


class Evaluate(TypedDict):
    type: Literal["evaluate"]
    user_answer_index: int
    question_id: str


class Advance(TypedDict):
    type: Literal["advance"]
    mastered_topic: str
    next_focus: Optional[FocusObjective]


# Union of all possible orchestrator actions
Action = Union[Explain, AskMCQ, Evaluate, Advance]


class InteractionEvent(TypedDict):
    event_type: Literal[
        "user_answer",
        "user_question",
        "system_tick",
        "system_explanation",
        "system_feedback",
        "system_advance",
    ]
    data: dict


def choose_action(ctx: TutorContext, last_event: InteractionEvent) -> Action:
    """Decide the next micro-action for the orchestrator using learned action weights and event context."""
    # If user requested a summary prompt (e.g., "I'm stuck" button), handle immediately
    if ctx.user_model_state.pending_interaction_type == "summary_prompt":
        # Determine segment to re-explain (rewind by one)
        prev_index = max(0, ctx.user_model_state.current_topic_segment_index - 1)
        # Clear pending flags
        ctx.user_model_state.pending_interaction_type = None
        ctx.user_model_state.pending_interaction_details = None
        return {
            "type": "explain", 
            "topic": ctx.current_focus_objective.topic, 
            "segment_index": prev_index, 
            "style": "analogy"
        }
    # Build default weight map
    weights_map = {"explain": 1.0, "ask_mcq": 1.0, "evaluate": 1.0, "advance": 1.0}
    # Fetch weights from Supabase if available
    if SUPABASE_CLIENT:
        try:
            res = SUPABASE_CLIENT.table("action_weights").select("action_type", "weight").execute()
            for row in getattr(res, 'data', []):
                weights_map[row.get("action_type")] = row.get("weight", 1.0)
        except Exception as e:
            print(f"[Policy] Failed to fetch action_weights: {e}")
    # Determine candidate actions based on last event
    et = last_event.get("event_type")
    if et == "system_tick":
        # Initial turn: start with explanation
        return {"type": "explain", "topic": ctx.current_focus_objective.topic, "segment_index": ctx.user_model_state.current_topic_segment_index, "style": "standard"}
    # Map event to possible actions
    if et == "system_explanation":
        candidates = ["explain", "ask_mcq", "advance"]
    elif et in ("user_answer", "system_feedback"):
        candidates = ["explain", "advance"]
    elif et == "system_advance":
        return {"type": "explain", "topic": ctx.current_focus_objective.topic, "segment_index": ctx.user_model_state.current_topic_segment_index, "style": "standard"}
    elif et == "user_question":
        candidates = ["explain"]
    else:
        candidates = ["ask_mcq"]
    # Sample action type
    weights = [weights_map.get(a, 1.0) for a in candidates]
    selected = random.choices(candidates, weights, k=1)[0]
    # Construct full action dict
    if selected == "explain":
        return {"type": "explain", "topic": ctx.current_focus_objective.topic, "segment_index": ctx.user_model_state.current_topic_segment_index, "style": None}
    if selected == "ask_mcq":
        s = ctx.user_model_state
        topic = ctx.current_focus_objective.topic
        c = s.concepts.get(topic)
        if c and c.mastery < 0.8:
            diff = bloom_difficulty(c.mastery, s.learning_pace_factor)
            return {"type": "ask_mcq", "topic": topic, "difficulty": diff, "misconception_focus": None}
        return {"type": "ask_mcq", "topic": topic, "difficulty": "medium", "misconception_focus": None}
    if selected == "evaluate":
        data = last_event.get("data", {})
        return {"type": "evaluate", "user_answer_index": data.get("user_answer_index"), "question_id": data.get("question_id")}
    if selected == "advance":
        return {"type": "advance", "mastered_topic": ctx.current_teaching_topic or ctx.current_focus_objective.topic, "next_focus": None}
    # Fallback
    return {"type": "ask_mcq", "topic": ctx.current_focus_objective.topic, "difficulty": "medium", "misconception_focus": None}
