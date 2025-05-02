from ai_tutor.skills import skill
from ai_tutor.api_models import ExplanationResponse
from ai_tutor.skills.explain_concept import explain_concept
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext

@skill(name_override="concept_overview")
async def concept_overview(
    ctx: RunContextWrapper[TutorContext],
    topic: str | None = None,
    details: str | None = None,
    concept: str | None = None,
    **unused_kwargs,
) -> ExplanationResponse:
    """Alias skill for concept overview, delegating to explain_concept."""
    return await explain_concept(
        ctx,
        topic=topic,
        details=details,
        concept=concept,
        **unused_kwargs
    ) 