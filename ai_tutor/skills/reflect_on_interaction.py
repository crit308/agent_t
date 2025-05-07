from typing import Any, Optional, Dict, List
from ai_tutor.skills import skill
from ai_tutor.context import TutorContext
from ai_tutor.agents.models import QuizFeedbackItem
from agents.run_context import RunContextWrapper
from pydantic import BaseModel, Field, ValidationError, validator
from ai_tutor.exceptions import ToolInputError
import logging

logger = logging.getLogger(__name__)

class ReflectOnInteractionArgs(BaseModel):
    topic: str = Field(..., min_length=1, description="The topic of the interaction.")
    interaction_summary: str = Field(..., min_length=1, description="A summary of the recent interaction.")
    user_response: Optional[str] = Field(default=None, description="The user's specific response, if any.")
    feedback_provided: Optional[QuizFeedbackItem] = Field(default=None, description="Feedback item if a question was just answered.")

    @validator('topic', 'interaction_summary', 'user_response')
    def check_not_just_whitespace(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Field cannot be only whitespace if provided.")
        return v

@skill
async def reflect_on_interaction(ctx: RunContextWrapper[TutorContext],
                                **kwargs) -> Dict[str, Any]:
    """Analyze an interaction and optional feedback to suggest next tutoring steps."""
    try:
        args = ReflectOnInteractionArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for reflect_on_interaction: {e}")

    logger.info(f"[Skill reflect_on_interaction] Called for topic '{args.topic}'. Summary: {args.interaction_summary}")
    
    suggestions: List[str] = []
    analysis = f"Reflection on interaction regarding '{args.topic}': {args.interaction_summary}. "
    
    if args.feedback_provided and not args.feedback_provided.is_correct:
        analysis += (
            f"User incorrectly selected '{args.feedback_provided.user_selected_option}' "
            f"when the correct answer was '{args.feedback_provided.correct_option}'. "
        )
        analysis += f"Explanation: {args.feedback_provided.explanation}. "
        suggestions.append(
            f"Re-explain the core concept using the provided explanation: '{args.feedback_provided.explanation}'."
        )
        if getattr(args.feedback_provided, 'improvement_suggestion', None):
            suggestions.append(
                f"Focus on the improvement suggestion: '{args.feedback_provided.improvement_suggestion}'."
            )
        suggestions.append(
            f"Try asking a slightly different checking question on the same concept."
        )
    elif args.user_response and any(kw in args.user_response.lower() for kw in ['dunno', "don't know", 'clueless', 'no idea']):
        analysis += f"User expressed significant uncertainty: '{args.user_response}'. "
        suggestions.append(f"Simplify the explanation for '{args.topic}', or break it down further.")
        suggestions.append(f"Offer a concrete example or analogy for '{args.topic}'.")
    elif args.interaction_summary and any(kw in args.interaction_summary.lower() for kw in ['incorrect', 'struggled', 'confused']):
        analysis += f"Interaction summary indicates user struggled: '{args.interaction_summary}'. "
        suggestions.append(
            f"Consider re-explaining the last segment of '{args.topic}' using a different approach or analogy."
        )
        suggestions.append(
            f"Ask a simpler checking question focused on the specific confusion points for '{args.topic}'."
        )
    else:
        analysis += "Interaction seems positive or neutral."
        suggestions.append(
            "Proceed with the next logical step in the micro-plan (e.g., next segment, checking question)."
        )
    
    logger.info(f"[Skill reflect_on_interaction] Analysis: {analysis}. Suggestions: {suggestions}")
    return {"analysis": analysis, "suggested_next_steps": suggestions} 