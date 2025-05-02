from ai_tutor.skills import skill
from ai_tutor.api_models import ExplanationResponse
# Import invoke helper
from ai_tutor.utils.tool_helpers import invoke
# Import the tool object itself
from ai_tutor.skills.explain_concept import explain_concept as explain_concept_tool
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
    """Alias skill for providing overviews, delegating via invoke."""
    # Use invoke to call the explain_concept TOOL
    # The invoke helper handles context wrapping based on target signature
    return await invoke(
        explain_concept_tool, # Pass the FunctionTool object
        ctx=ctx.context,      # Pass the raw TutorContext to invoke
        topic=topic,
        details="Provide a brief overview. Keep it concise.", # Override detail level
        concept=concept,
        **unused_kwargs
    ) 