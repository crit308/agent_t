from ai_tutor.skills import skill
from ai_tutor.api_models import ExplanationResponse
# Import invoke helper
from ai_tutor.utils.tool_helpers import invoke
# Import the tool object itself (or its name)
from ai_tutor.skills.explain_concept import explain_concept as explain_concept_tool 
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext

@skill(name_override="explain_process")
async def explain_process(
    ctx: RunContextWrapper[TutorContext],
    topic: str | None = None,
    details: str | None = None,
    concept: str | None = None,
    **unused_kwargs,
) -> ExplanationResponse:
    """Alias skill for explaining processes, delegating via invoke."""
    # Use invoke to call the explain_concept TOOL
    # The invoke helper handles context wrapping based on target signature
    return await invoke(
        explain_concept_tool, # Pass the FunctionTool object
        ctx=ctx.context,      # Pass the raw TutorContext to invoke
        topic=topic,
        details=details,
        concept=concept,
        **unused_kwargs
    ) 