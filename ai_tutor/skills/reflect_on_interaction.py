from agents import function_tool
from typing import Any, Optional, Dict
from ai_tutor.context import TutorContext
from ai_tutor.agents.models import QuizFeedbackItem
from agents.run_context import RunContextWrapper
from pydantic import BaseModel

@function_tool
async def reflect_on_interaction(ctx: RunContextWrapper[TutorContext],
                                topic: str,
                                interaction_summary: str,
                                user_response: Optional[str] = None,
                                feedback_provided: Optional[QuizFeedbackItem] = None) -> Dict[str, Any]:
    """Analyze an interaction and optional feedback to suggest next tutoring steps."""
    # Validate parameters
    class _Params(BaseModel):
        topic: str
        interaction_summary: str
        user_response: Optional[str] = None
        feedback_provided: Optional[QuizFeedbackItem] = None
        model_config = {"extra": "forbid"}
    _Params(
        topic=topic,
        interaction_summary=interaction_summary,
        user_response=user_response,
        feedback_provided=feedback_provided
    )
    # Perform reflection logic
    print(f"[Skill reflect_on_interaction] Called for topic '{topic}'. Summary: {interaction_summary}")
    suggestions: list[str] = []
    analysis = f"Reflection on interaction regarding '{topic}': {interaction_summary}. "
    if feedback_provided and not feedback_provided.is_correct:
        analysis += (
            f"User incorrectly selected '{feedback_provided.user_selected_option}' "
            f"when the correct answer was '{feedback_provided.correct_option}'. "
        )
        analysis += f"Explanation: {feedback_provided.explanation}. "
        suggestions.append(
            f"Re-explain the core concept using the provided explanation: '{feedback_provided.explanation}'."
        )
        if getattr(feedback_provided, 'improvement_suggestion', None):
            suggestions.append(
                f"Focus on the improvement suggestion: '{feedback_provided.improvement_suggestion}'."
            )
        suggestions.append(
            f"Try asking a slightly different checking question on the same concept."
        )
    elif any(kw in interaction_summary.lower() for kw in ['incorrect', 'struggled']):
        suggestions.append(
            f"Consider re-explaining the last segment of '{topic}' using a different approach or analogy."
        )
        suggestions.append(
            f"Ask a simpler checking question focused on the specific confusion points for '{topic}'."
        )
    else:
        analysis += "Interaction seems positive or neutral."
        suggestions.append(
            "Proceed with the next logical step in the micro-plan (e.g., next segment, checking question)."
        )
    print(f"[Skill reflect_on_interaction] Analysis: {analysis}. Suggestions: {suggestions}")
    return {"analysis": analysis, "suggested_next_steps": suggestions} 