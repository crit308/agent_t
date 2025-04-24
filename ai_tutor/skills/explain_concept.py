from ai_tutor.core.llm import LLMClient
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.skills import skill

@skill(cost="medium")
async def explain_concept(ctx: RunContextWrapper[TutorContext], topic: str, explanation_details: str) -> str:
    """Skill that uses LLMClient to explain a concept."""
    llm = LLMClient()
    # System prompt for explanation
    system_msg = {"role": "system", "content": "You are an AI tutor. Provide a clear, detailed explanation."}
    user_msg = {"role": "user", "content": f"{explanation_details} on topic '{topic}'."}
    return await llm.chat([system_msg, user_msg]) 