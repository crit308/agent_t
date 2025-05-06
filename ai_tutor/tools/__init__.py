"""Tools for the AI Tutor system.""" 

from __future__ import annotations
from agents import function_tool
from agents.run_context import RunContextWrapper
from typing import Any, Optional, Literal, Union, cast, Dict, List
import os
from datetime import datetime
import traceback
from ai_tutor.dependencies import SUPABASE_CLIENT, get_supabase_client
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
from ai_tutor.skills import skill

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
@skill(cost="low")
async def healthz(ctx) -> str:
    """Ping‑pong tool so orchestrator traces always include a baseline call."""
    return "ok"

@_export
@skill(cost="low")
def update_explanation_progress(ctx: RunContextWrapper[TutorContext], segment_index: int) -> str:
    """DEPRECATED: The Orchestrator manages micro-steps directly."""
    return "Error: This tool is deprecated. Orchestrator manages micro-steps."

@_export
@skill(cost="medium")
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

__all__ = _exports 