# from agents import function_tool # No longer used
from ai_tutor.skills import skill # Import correct decorator
# from ai_tutor.utils.agent_callers import call_teacher_agent # No longer used
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.core.llm import LLMClient # Import LLMClient
import logging

logger = logging.getLogger(__name__)

@skill
async def remediate_concept(ctx: RunContextWrapper[TutorContext], topic: str, remediation_details: str) -> str:
    """Skill that uses LLMClient to generate remediation text for a specific topic based on provided details."""
    llm = LLMClient()
    prompt = (
        f"The user needs help understanding the topic: '{topic}'. "
        f"Specific issues or context for remediation: {remediation_details}. "
        f"Provide a clear, targeted explanation or exercise to help the user overcome this specific difficulty. "
        f"Focus on clarifying the points mentioned in the remediation details."
    )
    system_msg = {"role": "system", "content": "You are an AI tutor providing targeted remediation help. Generate helpful text based on the user's difficulty."}
    user_msg = {"role": "user", "content": prompt}

    try:
        logger.info(f"Calling LLM for remediation on topic: {topic}")
        remediation_text = await llm.chat([system_msg, user_msg])
        logger.info(f"LLM remediation response received for topic: {topic}")
        return remediation_text
        
    except Exception as e:
        logger.error(f"Error in remediate_concept skill for topic '{topic}': {e}", exc_info=True)
        # Return an error message or re-raise?
        # Returning error message string for now, as skill is expected to return str.
        return f"Sorry, an error occurred while trying to generate remediation for '{topic}'." 