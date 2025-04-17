from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING, Union
import json

from agents import Agent, Runner, RunConfig, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper

# Use TYPE_CHECKING for TutorContext import
if TYPE_CHECKING:
    from ai_tutor.context import TutorContext # Use the enhanced context
# Import tool functions (assuming they will exist in a separate file)
from ai_tutor.tools.orchestrator_tools import (
    call_planner_agent,
    call_teacher_agent,
    call_quiz_creator_agent,
    call_quiz_teacher_evaluate,
    update_user_model,
    get_user_model_status,
    reflect_on_interaction,
)
from ai_tutor.policy import choose_action, InteractionEvent, Action

# Import models needed for type hints if tools return them
# Also import models needed for the output Union type
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, LessonContent, FocusObjective
from ai_tutor.api_models import (
    TutorInteractionResponse, ExplanationResponse, QuestionResponse,
    FeedbackResponse, MessageResponse, ErrorResponse
)
from ai_tutor.dependencies import SUPABASE_CLIENT  # Supabase client for logging actions

def create_orchestrator_agent(api_key: str = None) -> Agent['TutorContext']:
    """Creates the Orchestrator Agent for the AI Tutor."""

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    provider: ModelProvider = OpenAIProvider()
    # Use the specified model version
    base_model = provider.get_model("gpt-4o-2024-08-06")  # Using exact model version

    orchestrator_tools = [
        call_planner_agent,
        call_teacher_agent,
        call_quiz_creator_agent,
        call_quiz_teacher_evaluate,
        update_user_model,
        get_user_model_status,
        reflect_on_interaction,
    ]

    orchestrator_agent = Agent['TutorContext'](
        name="TutorOrchestrator",
        instructions="""
        Call the `choose_action` tool with the context and last event each turn; then immediately execute the resulting action via the corresponding tool. Only exit when waiting for learner input.
        """,
        tools=orchestrator_tools,
        output_type=TutorInteractionResponse,
        model=base_model,
    )

    return orchestrator_agent 

async def run_orchestrator(ctx: TutorContext, last_event: InteractionEvent):
    """Run a single orchestrator step and return the corresponding API model response."""
    ctx_wrapper = RunContextWrapper(ctx)
    # If user clicked 'next', directly generate a quiz question
    if last_event.get("event_type") == "next":
        instructions = f"Create a medium difficulty question"
        payload_q = json.dumps({"topic": ctx.current_focus_objective.topic, "instructions": instructions})
        quiz_creation = await call_quiz_creator_agent.on_invoke_tool(ctx_wrapper, payload_q)
        ctx.user_model_state.pending_interaction_type = "checking_question"
        if hasattr(quiz_creation, "question") and quiz_creation.question:
            ctx.user_model_state.pending_interaction_details = {"question_id": quiz_creation.question.id}
        return QuestionResponse(
            response_type="question",
            question=quiz_creation.question,
            topic=ctx.current_focus_objective.topic,
            context=None
        )
    # Map client 'start' to system_tick
    et = last_event.get("event_type")
    if et == "start":
        event = InteractionEvent(event_type="system_tick", data={})
    else:
        event = last_event
    # Decide next action via policy
    action: Action = choose_action(ctx, event)  # type: ignore
    # Execute and format response
    if action["type"] == "explain":
        # Explain current segment
        segment_index = ctx.user_model_state.current_topic_segment_index or 0
        payload = json.dumps({"topic": action["topic"], "explanation_details": f"Segment {segment_index}"})
        explanation_result = await call_teacher_agent.on_invoke_tool(ctx_wrapper, payload)
        # Advance to next segment
        ctx.user_model_state.current_topic_segment_index = segment_index + 1
        return ExplanationResponse(
            response_type="explanation",
            text=(explanation_result.details if hasattr(explanation_result, "details") else str(explanation_result)),
            topic=action["topic"],
            segment_index=segment_index,
            is_last_segment=False,
            references=None
        )
    if action["type"] == "ask_mcq":
        # Create a quiz question
        instructions = f"Create a {action['difficulty']} difficulty question"
        if action.get("misconception_focus"):
            instructions += f" probing {action['misconception_focus']}"
        quiz_payload = json.dumps({"topic": action["topic"], "instructions": instructions})
        quiz_creation = await call_quiz_creator_agent.on_invoke_tool(ctx_wrapper, quiz_payload)
        ctx.user_model_state.pending_interaction_type = "checking_question"
        if hasattr(quiz_creation, 'question') and quiz_creation.question:
            ctx.user_model_state.pending_interaction_details = {"question_id": quiz_creation.question.id}
        return QuestionResponse(
            response_type="question",
            question=quiz_creation.question,
            topic=action["topic"],
            context=None
        )
    if action["type"] == "evaluate":
        # Evaluate learner's answer
        eval_payload = json.dumps({"user_answer_index": action["user_answer_index"]})
        feedback = await call_quiz_teacher_evaluate.on_invoke_tool(ctx_wrapper, eval_payload)
        return FeedbackResponse(
            response_type="feedback",
            feedback=feedback,
            topic=(ctx.current_teaching_topic or action.get("topic", "")),
            correct_answer=getattr(feedback, "correct_option", None),
            explanation=getattr(feedback, "explanation", None)
        )
    if action["type"] == "advance":
        # Mark topic mastered
        um_payload = json.dumps({"topic": action["mastered_topic"], "outcome": "mastered"})
        await update_user_model.on_invoke_tool(ctx_wrapper, um_payload)
        return MessageResponse(
            response_type="message",
            text=f"Advanced to next focus.",
            message_type="info"
        )
    # Fallback unknown action
    return ErrorResponse(response_type="error", message=f"Unknown action type: {action.get('type')}") 