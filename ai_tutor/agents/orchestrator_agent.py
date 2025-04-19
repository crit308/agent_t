from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING, Union, Any
import json
from functools import lru_cache

from agents import Agent, Runner, RunConfig, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper

# Use TYPE_CHECKING for TutorContext import
if TYPE_CHECKING:
    from ai_tutor.context import TutorContext # Use the enhanced context
# Import tool functions (assuming they will exist in a separate file)
from ai_tutor.tools import (
    call_planner_agent,
    call_teacher_agent,
    call_quiz_creator_agent,
    call_quiz_teacher_evaluate,
    update_user_model,
    get_user_model_status,
    reflect_on_interaction,
)
from ai_tutor.policy import InteractionEvent, Action
from ai_tutor.utils.tool_helpers import invoke

# Import models needed for type hints if tools return them
# Also import models needed for the output Union type
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, LessonContent, FocusObjective
from ai_tutor.api_models import (
    TutorInteractionResponse, ExplanationResponse, QuestionResponse,
    FeedbackResponse, MessageResponse, ErrorResponse
)
from ai_tutor.dependencies import SUPABASE_CLIENT, get_openai  # Supabase client for logging actions
from ai_tutor.errors import ToolExecutionError, ErrorResponse

# Import model name from settings, with fallback
def get_orchestrator_model_name():
    try:
        from ai_tutor.settings import ORCHESTRATOR_MODEL_NAME
        return ORCHESTRATOR_MODEL_NAME
    except (ImportError, AttributeError):
        return "gpt-4o-2024-08-06"  # fallback

def create_orchestrator_agent(api_key: str = None, client=None) -> Agent['TutorContext']:
    """Creates the Orchestrator Agent for the AI Tutor. Optionally accepts an OpenAI client for reuse."""

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Always prefer a shared async OpenAI client for HTTP pool reuse
    if client is None:
        try:
            client = get_openai()
        except Exception:
            client = None
    provider: ModelProvider = OpenAIProvider(openai_client=client) if client else OpenAIProvider()
    # Use the specified model version
    base_model = provider.get_model("gpt-4o-2024-08-06")  # Using exact model version

    orchestrator_tools = [
        call_planner_agent,
        call_teacher_agent,
        call_quiz_creator_agent,
        call_quiz_teacher_evaluate,
        update_user_model,
        get_user_model_status,
        # reflect_on_interaction tool temporarily disabled due to schema validation issue
    ]

    orchestrator_agent = Agent['TutorContext'](
        name="TutorOrchestrator",
        instructions="You are the runtime wrapper of the tutor; you will *not* generate text yourself.",
        tools=orchestrator_tools,
        output_type=TutorInteractionResponse,
        model=base_model,
    )

    return orchestrator_agent 

# --- Add persist_ctx helper ---
def persist_ctx(ctx, last_event):
    """Persist the current TutorContext back to the sessions table."""
    if SUPABASE_CLIENT:
        # Build update query for context_data
        query = SUPABASE_CLIENT.table("sessions").update(
            {"context_data": ctx.model_dump(mode="json")}  # JSON-serializable dict
        ).eq("id", str(ctx.session_id))
        # Filter by user_id if present to enforce ownership
        if hasattr(ctx, "user_id"):
            query = query.eq("user_id", str(ctx.user_id))
        # Execute the update
        query.execute()
    # Note: last_event persistence is handled in the /interact endpoint if needed

async def run_orchestrator(ctx: TutorContext, last_event: InteractionEvent):
    """Run micro-steps of the orchestrator until awaiting learner input, then return that response."""
    # Import tools at runtime to allow monkeypatched stubs
    from ai_tutor.policy import choose_action
    from ai_tutor.tools import (
        call_teacher_agent,
        call_quiz_teacher_evaluate,
        call_quiz_creator_agent,
        update_user_model,
    )
    ctx_wrapper = RunContextWrapper(ctx)
    event = last_event

    while True:
        action: Action = choose_action(ctx, event)

        if action["type"] == "explain":
            segment_index = ctx.user_model_state.current_topic_segment_index or 0
            try:
                explanation_result = await invoke(
                    call_teacher_agent,
                    ctx,
                    topic=action["topic"],
                    explanation_details=f"Segment {segment_index}",
                )
                persist_ctx(ctx, event)
            except ToolExecutionError as e:
                return ErrorResponse(tool="call_teacher_agent", detail=e.detail, code=e.code)
            ctx.user_model_state.current_topic_segment_index = segment_index + 1
            event = {"event_type": "system_explanation", "data": explanation_result.model_dump()}
            continue

        if action["type"] == "evaluate":
            try:
                feedback = await invoke(
                    call_quiz_teacher_evaluate,
                    ctx,
                    user_answer_index=action["user_answer_index"],
                )
                persist_ctx(ctx, event)
            except ToolExecutionError as e:
                return ErrorResponse(tool="call_quiz_teacher_evaluate", detail=e.detail, code=e.code)
            event = {"event_type": "system_feedback", "data": feedback.model_dump()}
            continue

        if action["type"] == "ask_mcq":
            instructions = f"Create a {action['difficulty']} difficulty question"
            if action.get("misconception_focus"):
                instructions += f" probing {action['misconception_focus']}"
            try:
                quiz_resp = await invoke(
                    call_quiz_creator_agent,
                    ctx,
                    topic=action["topic"],
                    instructions=instructions,
                )
                persist_ctx(ctx, event)
            except ToolExecutionError as e:
                return ErrorResponse(tool="call_quiz_creator_agent", detail=e.detail, code=e.code)
            if isinstance(quiz_resp, QuestionResponse):
                # Persist the question as current_question_json for resume
                if SUPABASE_CLIENT:
                    SUPABASE_CLIENT.table("sessions").update({
                        "current_question_json": quiz_resp.model_dump_json()
                    }).eq("id", str(ctx.session_id)).execute()
                return quiz_resp
            ctx.user_model_state.pending_interaction_type = "checking_question"
            if hasattr(quiz_resp, "question") and quiz_resp.question:
                try:
                    qid = quiz_resp.question.id
                except AttributeError:
                    qid = None
                ctx.user_model_state.pending_interaction_details = {"question_id": qid}
            return QuestionResponse(
                response_type="question",
                question=quiz_resp.question,
                topic=action["topic"],
                context=None,
            )

        if action["type"] == "advance":
            try:
                await invoke(
                    update_user_model,
                    ctx,
                    topic=action["mastered_topic"],
                    outcome="mastered",
                )
                persist_ctx(ctx, event)
            except ToolExecutionError as e:
                return ErrorResponse(tool="update_user_model", detail=e.detail, code=e.code)
            ctx.user_model_state.current_topic_segment_index = 0
            return MessageResponse(
                response_type="message",
                text="Great! Let's tackle the next concept.",
                message_type="info",
            )

        # Fallback for unknown actions
        return ErrorResponse(
            response_type="error",
            message=f"Unknown action type: {action.get('type')}",
        ) 

def get_orchestrator():
    """Return a cached orchestrator agent instance (singleton per process)."""
    return create_orchestrator_agent(client=get_openai()) 

@lru_cache(maxsize=8)
def get_orchestrator_cached():
    """Return a cached orchestrator agent instance (singleton per process)."""
    return create_orchestrator_agent() 