from agents import function_tool
from ai_tutor.core.llm import LLMClient
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
import logging

logger = logging.getLogger(__name__)

@function_tool
async def explain_concept(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    details: str
) -> str:
    """Skill that uses LLMClient to explain a concept."""
    logger.info(f"[Skill explain_concept] Explaining topic='{topic}', details='{details}'")
    llm = LLMClient()
    # System prompt for explanation
    system_msg = {"role": "system", "content": "You are an AI tutor. Provide a clear, detailed explanation of the requested concept segment."}
    # Use the passed arguments in the prompt
    user_msg = {"role": "user", "content": f"{details} on topic '{topic}'."}
    # Optional: Add logic here to fetch relevant info from KB using ctx if needed
    # kb_text = await invoke(read_knowledge_base, ctx) ... include in prompt ...
    explanation = await llm.chat([system_msg, user_msg])
    logger.info(f"[Skill explain_concept] LLM Explanation received: {explanation[:100]}...")
    return explanation 