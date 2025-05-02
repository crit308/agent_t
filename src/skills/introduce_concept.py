"""Alias skill for introducing a new concept using the existing explain_concept skill."""

from ai_tutor.skills import skill
from ai_tutor.api_models import ExplanationResponse
from ai_tutor.utils.tool_helpers import invoke
from ai_tutor.skills.explain_concept import explain_concept as explain_concept_tool
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext


@skill(name_override="introduce_concept")
async def introduce_concept(
    ctx: RunContextWrapper[TutorContext],
    topic: str | None = None,
    details: str | None = None,
    concept: str | None = None,
    **unused_kwargs,
) -> ExplanationResponse:
    """Provide an introductory explanation for a concept by delegating to explain_concept."""
    # Default details if none provided to emphasize introductory level
    intro_details = details or "Provide an introductory explanation suitable for beginners."

    return await invoke(
        explain_concept_tool,
        ctx=ctx.context,  # Pass raw TutorContext to invoke
        topic=topic,
        details=intro_details,
        concept=concept,
        **unused_kwargs,
    ) 