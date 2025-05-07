from ai_tutor.skills import skill
from ai_tutor.core.llm import LLMClient
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.api_models import ExplanationResponse
import logging
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError, root_validator
from ai_tutor.exceptions import ToolInputError

logger = logging.getLogger(__name__)

class ExplainConceptArgs(BaseModel):
    topic: Optional[str] = None
    details: Optional[str] = None
    concept: Optional[str] = None

    @root_validator(pre=True)
    def handle_concept_alias_and_ensure_topic(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        topic, concept = values.get('topic'), values.get('concept')
        
        if topic is None and concept is not None:
            values['topic'] = concept
        elif topic is None and concept is None:
            raise ValueError("Either 'topic' or 'concept' must be provided.")
        
        if values.get('topic') == "":
            raise ValueError("'topic' or 'concept' cannot be an empty string.")
            
        return values

    @validator('topic', 'concept', 'details')
    def check_not_just_whitespace(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Field cannot be only whitespace.")
        return v

@skill
async def explain_concept(
    ctx: RunContextWrapper[TutorContext],
    **kwargs,
) -> ExplanationResponse:
    """Skill that uses LLMClient to explain a concept and optionally generate whiteboard actions."""
    try:
        args = ExplainConceptArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for explain_concept: {e}")

    topic_to_explain = args.topic 
    details_for_explanation = args.details if args.details else ""

    logger.info(f"[Skill explain_concept] Explaining topic='{topic_to_explain}', details='{details_for_explanation}'")

    llm = LLMClient()
    system_msg = {
        "role": "system",
        "content": (
            "You are an AI tutor. Provide a clear, detailed explanation of the requested concept segment. "
            "If a simple diagram or visual would help, also return a 'whiteboard_actions' key as a list of CanvasObjectSpec objects. "
            "Respond ONLY with a JSON object: { 'explanation_text': ..., 'whiteboard_actions': [ ... ] (optional) }. "
            "If no drawing is needed, omit the 'whiteboard_actions' key."
        )
    }
    user_msg = {"role": "user", "content": f"{details_for_explanation} on topic '{topic_to_explain}'."}
    
    messages_for_llm: List[Dict[str, str]] = [system_msg, user_msg]
    llm_response_content = await llm.chat(messages_for_llm)
    
    logger.info(f"[Skill explain_concept] LLM raw response: {llm_response_content[:200]}...")
    
    parsed_response: Dict[str, Any]
    if isinstance(llm_response_content, str):
        try:
            parsed_response = json.loads(llm_response_content)
        except json.JSONDecodeError:
            start_index = llm_response_content.find('{')
            end_index = llm_response_content.rfind('}')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                json_str_candidate = llm_response_content[start_index : end_index + 1]
                try:
                    parsed_response = json.loads(json_str_candidate)
                except json.JSONDecodeError as e_inner:
                    logger.error(f"Failed to parse extracted JSON: {json_str_candidate}. Error: {e_inner}")
                    raise ValueError("LLM did not return a valid JSON object after extraction attempt.") from e_inner
            else:
                logger.error(f"Could not find JSON object delimiters in LLM response: {llm_response_content}")
                raise ValueError("LLM did not return a valid JSON object and delimiters not found.")
    elif isinstance(llm_response_content, dict):
        parsed_response = llm_response_content 
    else:
        logger.error(f"Unexpected response type from LLM: {type(llm_response_content)}")
        raise ValueError("LLM response was not a string or a dictionary.")

    explanation_text = parsed_response.get("explanation_text")
    if not explanation_text or not isinstance(explanation_text, str) or not explanation_text.strip():
        logger.error(f"LLM response missing or invalid 'explanation_text': {parsed_response}")
        raise ValueError("LLM response must include a non-empty 'explanation_text' string.")

    whiteboard_actions = parsed_response.get("whiteboard_actions")
    if whiteboard_actions is not None and not isinstance(whiteboard_actions, list):
        logger.warning(f"LLM response 'whiteboard_actions' is not a list: {whiteboard_actions}. Ignoring.")
        whiteboard_actions = None

    payload = ExplanationResponse(
        explanation_text=explanation_text,
        explanation_title=None,
        related_objectives=None,
        lesson_content=None
    )
    setattr(payload, 'whiteboard_actions', whiteboard_actions)
    return payload 