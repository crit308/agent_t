from ai_tutor.skills import skill
from ai_tutor.core.llm import LLMClient
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.api_models import ExplanationResponse
import logging
import json

logger = logging.getLogger(__name__)

@skill
async def explain_concept(
    ctx: RunContextWrapper[TutorContext],
    topic: str | None = None,
    details: str | None = None,
    concept: str | None = None,
    **unused_kwargs,
) -> ExplanationResponse:
    """Skill that uses LLMClient to explain a concept and optionally generate whiteboard actions."""
    logger.info(f"[Skill explain_concept] Explaining topic='{topic}', details='{details}'")
    # Allow 'concept' alias for topic
    if topic is None and concept is not None:
        topic = concept

    llm = LLMClient()
    # System prompt for explanation and optional whiteboard actions
    system_msg = {
        "role": "system",
        "content": (
            "You are an AI tutor. Provide a clear, detailed explanation of the requested concept segment. "
            "If a simple diagram or visual would help, also return a 'whiteboard_actions' key as a list of CanvasObjectSpec objects. "
            "Respond ONLY with a JSON object: { 'explanation_text': ..., 'whiteboard_actions': [ ... ] (optional) }. "
            "If no drawing is needed, omit the 'whiteboard_actions' key."
        )
    }
    user_msg = {"role": "user", "content": f"{details} on topic '{topic}'."}
    llm_response = await llm.chat([system_msg, user_msg])
    logger.info(f"[Skill explain_concept] LLM raw response: {llm_response[:200]}...")
    # Parse the LLM response as JSON
    if isinstance(llm_response, str):
        # Try to extract JSON
        start = llm_response.find('{')
        end = llm_response.rfind('}')
        if start != -1 and end != -1:
            llm_response = llm_response[start:end+1]
        parsed = json.loads(llm_response)
    elif isinstance(llm_response, dict):
        parsed = llm_response
    else:
        raise ValueError("LLM did not return a valid JSON object.")
    explanation_text = parsed.get("explanation_text")
    whiteboard_actions = parsed.get("whiteboard_actions")
    # Build ExplanationResponse payload
    payload = ExplanationResponse(
        explanation_text=explanation_text,
        explanation_title=None,
        related_objectives=None,
        lesson_content=None
    )
    # Attach whiteboard_actions as a non-model attribute for the executor to pick up
    setattr(payload, 'whiteboard_actions', whiteboard_actions)
    return payload 