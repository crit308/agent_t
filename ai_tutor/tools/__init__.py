"""Tools for the AI Tutor system.""" 

from __future__ import annotations
from agents import function_tool
from agents.run_context import RunContextWrapper
from typing import Any, Optional, Literal, Union, cast, Dict, List
import os
from datetime import datetime
import traceback
from ai_tutor.dependencies import SUPABASE_CLIENT
from ai_tutor.context import TutorContext, UserConceptMastery, UserModelState, is_mastered
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, ExplanationResult, QuizCreationResult
from ai_tutor.core.schema import PlannerOutput
from ai_tutor.api_models import (
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse
)
from ai_tutor.errors import ToolExecutionError
import json
from ai_tutor.telemetry import log_tool
from pydantic import BaseModel
from ai_tutor.utils.decorators import function_tool_logged
from ai_tutor.core.llm import LLMClient

_exports: List[str] = []
def _export(obj):
    # Support both function and FunctionTool objects
    name = getattr(obj, '__name__', None)
    if name is None and hasattr(obj, '__original_func__'):
        name = getattr(obj.__original_func__, '__name__', None)
    if name:
        _exports.append(name)
    return obj

# --- Orchestrator Tool Implementations ---

@_export
@function_tool_logged()
async def healthz(ctx) -> str:
    """Ping‑pong tool so orchestrator traces always include a baseline call."""
    return "ok"

@_export
@function_tool_logged()
def update_explanation_progress(ctx: RunContextWrapper[TutorContext], segment_index: int) -> str:
    """DEPRECATED: The Orchestrator manages micro-steps directly."""
    return "Error: This tool is deprecated. Orchestrator manages micro-steps."

@_export
@function_tool_logged()
async def call_quiz_teacher_evaluate(ctx: RunContextWrapper[TutorContext], user_answer_index: int) -> QuizFeedbackItem:
    """Evaluate the user's answer against the current QuizQuestion in context."""
    question = ctx.context.current_quiz_question
    if not question or not isinstance(question, QuizQuestion):
        raise ToolExecutionError("No valid QuizQuestion in context to evaluate.", code="missing_question")
    is_correct = (user_answer_index == question.correct_answer_index)
    selected = question.options[user_answer_index] if 0 <= user_answer_index < len(question.options) else ""
    correct = question.options[question.correct_answer_index]
    feedback = QuizFeedbackItem(
        question_index=user_answer_index,
        question_text=question.question,
        user_selected_option=selected,
        is_correct=is_correct,
        correct_option=correct,
        explanation=question.explanation,
        improvement_suggestion="Review the concept again." if not is_correct else ""
    )
    # Clear pending question state
    ctx.context.user_model_state.pending_interaction_type = None
    ctx.context.current_quiz_question = None
    return feedback

@_export
@function_tool_logged()
async def update_user_model(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained'],
    confusion_point: Optional[str] = None,
    last_accessed: Optional[str] = None,
    mastered_objective_title: Optional[str] = None,
) -> str:
    """Updates the user model state with interaction outcomes and temporal data."""
    print(f"[Tool update_user_model] Updating '{topic}' with outcome '{outcome}'")
    if not ctx.context or not ctx.context.user_model_state:
        raise ToolExecutionError("TutorContext or UserModelState not found.", code="missing_context")
    if not topic or not isinstance(topic, str):
        raise ToolExecutionError("Invalid topic provided for user model update.", code="invalid_topic")
    if topic not in ctx.context.user_model_state.concepts:
        ctx.context.user_model_state.concepts[topic] = UserConceptMastery()
    concept_state = ctx.context.user_model_state.concepts[topic]
    old_mastery = concept_state.mastery
    previously_mastered = is_mastered(concept_state)
    concept_state.last_interaction_outcome = outcome
    concept_state.last_accessed = last_accessed or datetime.now().isoformat()
    if confusion_point and confusion_point not in concept_state.confusion_points:
        concept_state.confusion_points.append(confusion_point)
    if outcome in ['correct', 'incorrect', 'mastered', 'struggled']:
        concept_state.attempts += 1
        if outcome in ['correct', 'mastered']:
            concept_state.alpha += 1
        else:
            concept_state.beta += 1
            if len(concept_state.confusion_points) > 2:
                ctx.context.user_model_state.learning_pace_factor = max(
                    0.5, ctx.context.user_model_state.learning_pace_factor - 0.1
                )
    if mastered_objective_title and mastered_objective_title not in ctx.context.user_model_state.mastered_objectives_current_section:
        ctx.context.user_model_state.mastered_objectives_current_section.append(mastered_objective_title)
        print(f"[Tool] Marked objective '{mastered_objective_title}' as mastered for current section.")
    print(f"[Tool] Updated '{topic}' - Mastery: {concept_state.mastery:.2f}, "
          f"Confidence: {concept_state.confidence}, "
          f"Pace: {ctx.context.user_model_state.learning_pace_factor:.2f}")
    if not previously_mastered and is_mastered(concept_state):
        print(f"[Tool update_user_model] Mastery achieved for '{topic}'. Triggering planner agent.")
        from ai_tutor.tools import call_planner_agent
        if ctx.context.current_focus_objective is None:
            await invoke(call_planner_agent, ctx)
    mastery = concept_state.mastery
    confidence = concept_state.confidence
    if hasattr(ctx.context, "ws"):
        try:
            await ctx.context.ws.send_json({"type": "mastery_update", "topic": topic, "mastery": mastery, "confidence": confidence})
        except Exception:
            pass
    new_mastery = mastery
    delta_mastery = new_mastery - old_mastery
    if SUPABASE_CLIENT:
        try:
            SUPABASE_CLIENT.table("concept_events").insert({
                "session_id": str(ctx.context.session_id),
                "user_id": str(ctx.context.user_id),
                "concept": topic,
                "outcome": outcome,
                "delta_mastery": delta_mastery
            }).execute()
        except Exception as e:
            print(f"[Tool update_user_model] Failed to log concept event: {e}")
    return f"User model updated for {topic}."

@_export
@function_tool_logged()
async def get_user_model_status(ctx: RunContextWrapper[TutorContext], topic: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves detailed user model state, optionally for a specific topic."""
    print(f"[Tool] Retrieving user model status for topic '{topic}'")
    if not ctx.context.user_model_state:
        return {"error": "No user model state found in context."}
    state = ctx.context.user_model_state
    if topic:
        if topic not in state.concepts:
            return {
                "topic": topic,
                "exists": False,
                "message": "Topic not found in user model."
            }
        concept = state.concepts[topic]
        return {
            "topic": topic,
            "exists": True,
            "mastery_level": concept.mastery,
            "attempts": concept.attempts,
            "last_outcome": concept.last_interaction_outcome,
            "confusion_points": concept.confusion_points,
            "last_accessed": concept.last_accessed
        }
    return state.model_dump(mode='json')

@_export
@function_tool_logged()
async def reflect_on_interaction(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    interaction_summary: str,
    user_response: Optional[str] = None,
    feedback_provided: Optional[QuizFeedbackItem] = None
) -> Dict[str, Any]:
    class _Params(BaseModel):
        topic: str
        interaction_summary: str
        user_response: Optional[str] = None
        feedback_provided: Optional[QuizFeedbackItem] = None
        model_config = {"extra": "forbid"}
    print(f"[Tool reflect_on_interaction] Called for topic '{topic}'. Summary: {interaction_summary}")
    suggestions = []
    analysis = f"Reflection on interaction regarding '{topic}': {interaction_summary}. "
    if feedback_provided and not feedback_provided.is_correct:
        analysis += f"User incorrectly selected '{feedback_provided.user_selected_option}' when the correct answer was '{feedback_provided.correct_option}'. "
        analysis += f"Explanation: {feedback_provided.explanation}. "
        suggestions.append(f"Re-explain the core concept using the provided explanation: '{feedback_provided.explanation}'.")
        if feedback_provided.improvement_suggestion:
            suggestions.append(f"Focus on the improvement suggestion: '{feedback_provided.improvement_suggestion}'.")
        suggestions.append(f"Try asking a slightly different checking question on the same concept.")
    elif "incorrect" in interaction_summary.lower() or "struggled" in interaction_summary.lower():
        suggestions.append(f"Consider re-explaining the last segment of '{topic}' using a different approach or analogy.")
        suggestions.append(f"Ask a simpler checking question focused on the specific confusion points for '{topic}'.")
    else:
        analysis += "Interaction seems positive or neutral."
        suggestions.append("Proceed with the next logical step in the micro-plan (e.g., next segment, checking question).")
    print(f"[Tool reflect_on_interaction] Analysis: {analysis}. Suggestions: {suggestions}")
    return {"analysis": analysis, "suggested_next_steps": suggestions}

@_export
@function_tool_logged()
async def call_planner_agent(
    ctx: RunContextWrapper[TutorContext],
    user_state_summary: Optional[str] = None
) -> PlannerOutput:
    """Delegate to the direct run_planner function."""
    from ai_tutor.agents.planner_agent import run_planner
    return await run_planner(ctx.context)

@_export
@function_tool_logged()
async def call_teacher_agent(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    explanation_details: str
) -> ExplanationResult:
    """Provide a concept explanation using LLMClient and wrap in ExplanationResult."""
    llm = LLMClient()
    system_msg = {"role": "system", "content": "You are an AI tutor. Provide a clear, concise explanation."}
    user_msg = {"role": "user", "content": f"Topic: {topic}. Details: {explanation_details}"}
    response = await llm.chat([system_msg, user_msg])
    # Wrap in ExplanationResult
    return ExplanationResult(status="delivered", details=response)

@_export
@function_tool_logged()
async def call_quiz_creator_agent(ctx: RunContextWrapper[TutorContext], topic: str, instructions: str) -> QuizCreationResult:
    """Generate quiz questions using LLM and return a QuizCreationResult."""
    llm = LLMClient()
    prompt = (
        f"Generate a JSON object matching the QuizCreationResult schema with a quiz of 3 multiple choice questions. "
        f"Topic: {topic}. Instructions: {instructions}"
    )
    response = await llm.chat([
        {"role": "system", "content": "You are a quiz creator that outputs valid JSON for QuizCreationResult."},
        {"role": "user", "content": prompt}
    ])
    try:
        result = QuizCreationResult.parse_raw(response)
    except Exception:
        # Fallback on failure
        result = QuizCreationResult(status="failed", quiz=None, question=None, details=response)
    return result

__all__ = _exports 