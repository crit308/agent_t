from ai_tutor.skills import skill
from ai_tutor.skills.create_quiz import create_quiz
from ai_tutor.api_models import QuestionResponse
from agents.run_context import RunContextWrapper
from ai_tutor.context import TutorContext

@skill(name_override="diagnostic_question")
async def diagnostic_question(
    ctx: RunContextWrapper[TutorContext],
    topic: str | None = None,
    instructions: str | None = None,
    **unused_kwargs,
) -> QuestionResponse:
    """Alias skill for diagnostic questions, delegating to create_quiz."""
    # Use provided instructions or default to a diagnostic prompt
    instr = instructions or "Please generate a diagnostic multiple-choice question to assess the user's initial understanding of the topic."
    # Delegate to create_quiz under the hood
    return await create_quiz(ctx, topic or "", instr) 