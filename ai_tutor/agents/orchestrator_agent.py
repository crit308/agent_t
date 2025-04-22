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
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, LessonContent, FocusObjective, PlannerOutput, ActionSpec
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
    """Run the orchestrator: get the planner's output, then execute the next_action as a blocking call until completion/failure."""
    from ai_tutor.tools import call_planner_agent, call_teacher_agent, call_quiz_creator_agent, call_quiz_teacher_evaluate, update_user_model
    from ai_tutor.api_models import ErrorResponse
    from ai_tutor.utils.tool_helpers import invoke

    ctx_wrapper = RunContextWrapper(ctx)

    # 1. Call the planner to get the next action spec
    planner_output = await invoke(call_planner_agent, ctx)
    if not isinstance(planner_output, PlannerOutput):
        return ErrorResponse(tool="call_planner_agent", detail="Planner did not return PlannerOutput", code="planner_output_error")

    action_spec: ActionSpec = planner_output.next_action
    agent = action_spec.agent
    params = action_spec.params or {}
    success_criteria = action_spec.success_criteria
    max_steps = action_spec.max_steps

    # 2. Map agent name to function
    agent_tool_map = {
        "teacher": call_teacher_agent,
        "quiz_creator": call_quiz_creator_agent,
        # Add more agents as needed
    }
    agent_tool = agent_tool_map.get(agent)
    if not agent_tool:
        return ErrorResponse(tool="orchestrator", detail=f"Unknown agent: {agent}", code="unknown_agent")

    # 3. Loop until sub-agent returns completed/failed or max_steps is reached
    result = None
    for step in range(1, max_steps + 1):
        try:
            # Map sub-agent invocation based on the agent string
            if agent == "teacher":
                result = await invoke(
                    call_teacher_agent,
                    ctx,
                    topic=planner_output.objective.topic,
                    explanation_details=success_criteria
                )
            elif agent == "quiz_creator":
                result = await invoke(
                    call_quiz_creator_agent,
                    ctx,
                    topic=planner_output.objective.topic,
                    instructions=success_criteria
                )
            else:
                # Default: pass through any params from the planner
                result = await invoke(agent_tool, ctx, **params)
        except ToolExecutionError as e:
            return ErrorResponse(tool=agent_tool.__name__, detail=e.detail, code=e.code)
        # Check for terminal status
        status = getattr(result, 'status', None)
        if status in ("completed", "failed", "delivered", "created"):
            break
    else:
        # Max steps reached without completion
        return ErrorResponse(tool=agent_tool.__name__, detail="max_steps exceeded without completion", code="max_steps_exceeded")

    # 4. Emit the final event (here: just return it)
    # In a websocket context, you would send this event to the client
    return result

def get_orchestrator():
    """Return a cached orchestrator agent instance (singleton per process)."""
    return create_orchestrator_agent(client=get_openai()) 

@lru_cache(maxsize=8)
def get_orchestrator_cached():
    """Return a cached orchestrator agent instance (singleton per process)."""
    return create_orchestrator_agent() 