from ai_tutor.skills import skill
from ai_tutor.api_models import QuestionResponse
# Import invoke helper
from ai_tutor.utils.tool_helpers import invoke
# Import the tool object itself
from ai_tutor.skills.create_quiz import create_quiz as create_quiz_tool 
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext

@skill(name_override="diagnostic_question")
async def diagnostic_question(
    ctx: RunContextWrapper[TutorContext],
    topic: str | None = None,
    question_type: str = "multiple_choice",
    difficulty: str = "Easy", # Default to easy for diagnostic
    **unused_kwargs,
) -> QuestionResponse:
    """Skill to ask a simple diagnostic question, delegating via invoke."""
    # Use invoke to call the create_quiz TOOL
    # The invoke helper handles context wrapping based on target signature
    return await invoke(
        create_quiz_tool,   # Pass the FunctionTool object
        ctx=ctx.context,    # Pass the raw TutorContext to invoke
        topic=topic,
        question_type=question_type,
        difficulty=difficulty,
        **unused_kwargs
    ) 