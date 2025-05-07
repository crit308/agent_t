from ai_tutor.skills import skill
from ai_tutor.core.llm import LLMClient
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.api_models import ExplanationResponse
import logging
import json
from typing import Optional, Dict, Any, List
from pydantic import field_validator, model_validator, BaseModel, Field, ValidationError
from ai_tutor.exceptions import ToolInputError, ExecutorError

logger = logging.getLogger(__name__)

class ExplainConceptArgs(BaseModel):
    topic: Optional[str] = Field(None, description="The main topic or concept to explain.")
    details: Optional[str] = Field(None, description="Specific details or sub-topics to focus on within the main topic.")
    concept: Optional[str] = Field(None, description="Alias for topic, for backward compatibility or alternative naming.")

    @model_validator(mode='before')
    @classmethod
    def check_topic_or_concept_provided_and_alias(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            # Pydantic usually passes a dict here from **kwargs, but good to be safe
            # or raise an error if this scenario is not expected
            logger.warning(f"ExplainConceptArgs received non-dict data for validation: {type(data)}")
            return data # Or raise TypeError("Expected a dictionary for validation data")

        topic = data.get('topic')
        concept_alias = data.get('concept') # Renamed for clarity within this validator

        # Handle aliasing: If topic is missing but concept_alias is present, use concept_alias as topic
        if topic is None and concept_alias is not None:
            data['topic'] = concept_alias
            topic = concept_alias # Update local variable for subsequent checks
            logger.debug(f"Aliased 'concept': \"{concept_alias}\" to 'topic'.")
        
        # Check if at least one is provided (after potential aliasing)
        if topic is None: # No need to check concept_alias again as it would have been aliased to topic
            raise ValueError('Either "topic" or "concept" (acting as an alias for topic) must be provided.')

        # Check if the final topic (after aliasing) is empty or just whitespace
        # Ensure topic is a string before stripping, as it could be None if the above check failed (though ValueError should prevent that)
        if topic is not None and not str(topic).strip():
             raise ValueError('"topic" (or "concept" if used as alias) cannot be empty or just whitespace.')
        
        # Remove the original 'concept' field after aliasing, if it's still there and different from topic,
        # to avoid confusion, unless it's explicitly needed downstream.
        # For now, we'll let it pass through if not aliased, Pydantic will handle it based on model fields.
        # If 'concept' was aliased, 'topic' now holds its value.
        # If 'concept' was provided AND 'topic' was provided, 'topic' takes precedence as per Pydantic model.

        return data

    @field_validator('topic', 'concept', 'details', mode='before')
    @classmethod
    def check_not_just_whitespace(cls, value: Optional[str], field) -> Optional[str]: # Added 'field' for context in error
        if isinstance(value, str) and not value.strip():
            # Use field.name to make the error message more specific
            raise ValueError(f"Field '{field.name}' cannot be empty or just whitespace if provided.")
        if value is not None and not isinstance(value, str):
            # Ensure that if a value is provided, it's a string
            raise TypeError(f"Field '{field.name}' must be a string if provided.")
        return value

    # Optional: Add model_config if needed, e.g., for extra='forbid'
    # model_config = ConfigDict(extra='forbid')

@skill
async def explain_concept(
    ctx: Any, # Assuming RunContextWrapper or similar. Replace Any with specific type if available.
    **kwargs,
) -> ExplanationResponse:
    """Skill that uses LLMClient to explain a concept and optionally generate whiteboard actions."""
    logger.info(f"[Skill explain_concept] Received raw args: {kwargs}")
    try:
        # Pydantic V2 automatically handles alias for 'concept' to 'topic' if defined in Field
        # but our model_validator handles it more explicitly for the logic needed.
        validated_args = ExplainConceptArgs(**kwargs)
        logger.info(f"[Skill explain_concept] Validated args: topic='{validated_args.topic}', details='{validated_args.details}'")
    except ValidationError as e:
        logger.error(f"[Skill explain_concept] Input validation failed: {e}")
        # Provide a more user-friendly error message that can be shown to the LLM or client
        error_messages = [err['msg'] for err in e.errors()]
        raise ToolInputError(f"Invalid arguments for explain_concept: {'; '.join(error_messages)}") from e

    topic_to_explain = validated_args.topic # This will be correctly populated by the model_validator
    details_for_prompt = validated_args.details or "Provide a general explanation of the topic."

    logger.info(f"[Skill explain_concept] Explaining topic='{topic_to_explain}', details='{details_for_prompt}'")

    llm = LLMClient()
    system_msg = {
        "role": "system",
        "content": (
            "You are an AI tutor. Provide a clear, detailed explanation of the requested concept segment. "
            "If a simple diagram or visual would help, also return a 'whiteboard_actions' key as a list of valid CanvasObjectSpec-like objects. "
            "Respond ONLY with a JSON object with the following schema: { \"explanation_text\": \"string\", \"whiteboard_actions\": [object] (optional) }. "
            "If no drawing is needed, omit the 'whiteboard_actions' key entirely or set it to null or an empty list. "
            "The 'explanation_text' MUST be a non-empty string. "
            "If 'whiteboard_actions' is provided, it MUST be a list of objects."
        )
    }
    user_msg = {"role": "user", "content": f"Explain: {details_for_prompt} (related to the broader topic of '{topic_to_explain}')."}
    
    messages_for_llm: List[Dict[str, str]] = [system_msg, user_msg]
    
    try:
        # Assuming llm.chat can handle Pydantic models directly or returns a dict/str
        llm_response = await llm.chat(messages_for_llm, response_format={"type": "json_object"}) # Request JSON object
        logger.debug(f"[Skill explain_concept] LLM raw response object: {llm_response}")

        parsed: Optional[Dict[str, Any]] = None
        if isinstance(llm_response, str): # Should be less common with response_format={"type": "json_object"}
            try:
                parsed = json.loads(llm_response)
            except json.JSONDecodeError:
                logger.warning("[Skill explain_concept] LLM responded with string but it wasn't valid JSON despite json_object mode. Attempting regex extraction.")
                match = re.search(r"\{.*\}", llm_response, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        logger.info("[Skill explain_concept] Successfully parsed JSON extracted via regex.")
                    except json.JSONDecodeError as e_inner:
                        logger.error(f"[Skill explain_concept] Failed to parse regex-extracted JSON: {match.group(0)}. Error: {e_inner}")
                        raise ExecutorError(f"LLM response was not valid JSON even after regex extraction: {e_inner}") from e_inner
                else:
                    logger.error("[Skill explain_concept] LLM string response was not JSON and no JSON object found via regex.")
                    raise ExecutorError("LLM response was not valid JSON and no object found via regex.")
        elif isinstance(llm_response, dict): # Expected path with response_format={"type": "json_object"}
            parsed = llm_response
        else:
            logger.error(f"[Skill explain_concept] Unexpected LLM response type: {type(llm_response)}. Expected dict or str.")
            raise ExecutorError(f"Unexpected LLM response type: {type(llm_response)}")

        if not parsed: # Should be caught by earlier checks, but as a safeguard
            raise ExecutorError("LLM did not return a parsable JSON response object.")

        explanation_text = parsed.get("explanation_text")
        whiteboard_actions = parsed.get("whiteboard_actions") # Can be None, empty list, or list of dicts

        if not explanation_text or not isinstance(explanation_text, str) or not explanation_text.strip():
            logger.error(f"[Skill explain_concept] LLM response missing valid 'explanation_text'. Received: {explanation_text}")
            raise ExecutorError("LLM response is missing a valid 'explanation_text' or it's empty.")
        
        if whiteboard_actions is not None and not isinstance(whiteboard_actions, list):
            logger.warning(f"[Skill explain_concept] LLM response provided 'whiteboard_actions' but it was not a list. Type: {type(whiteboard_actions)}. Value: {whiteboard_actions}. Ignoring.")
            whiteboard_actions = None # Treat as no actions if format is incorrect
        
        # Further validation for whiteboard_actions if needed (e.g., structure of objects in the list)
        # For now, we assume if it's a list, it's in the expected format for the frontend.

        logger.info(f"[Skill explain_concept] Successfully parsed LLM response for '{topic_to_explain}'. Explanation length: {len(explanation_text)}, Actions: {'Yes' if whiteboard_actions else 'No'}")

        payload = ExplanationResponse(
            explanation_text=explanation_text.strip(),
            explanation_title=f"Explanation: {topic_to_explain}",
            # related_objectives=None, # Fill if applicable
            # lesson_content=None, # Fill if applicable
        )
        
        # Storing whiteboard_actions directly in the Pydantic model if it has such a field,
        # or passing them separately if the consuming code expects that.
        # ExplanationResponse model does not have whiteboard_actions field.
        # We can attach it as an ad-hoc attribute if the invoker knows to look for it,
        # or modify ExplanationResponse to include it.
        # For now, using setattr as per original logic.
        if whiteboard_actions: # Only set if there are actions
            setattr(payload, 'whiteboard_actions', whiteboard_actions)
            logger.debug(f"Attached whiteboard_actions to ExplanationResponse payload: {whiteboard_actions}")

        return payload

    except ToolInputError: # Re-raise ToolInputError if it was from arg validation
        raise
    except ExecutorError as e: # Catch specific ExecutorErrors from LLM processing
         logger.error(f"[Skill explain_concept] ExecutorError for topic '{topic_to_explain}': {e}", exc_info=True)
         # Re-raise to be handled by the agent/executor framework
         raise
    except Exception as e:
         # Catch any other unexpected errors during the skill execution
         logger.error(f"[Skill explain_concept] Unexpected error for topic '{topic_to_explain}': {e}", exc_info=True)
         # Wrap in ExecutorError for consistent error handling upstream
         raise ExecutorError(f"An unexpected error occurred in 'explain_concept' skill: {e}") from e

# Ensure other parts of the file, if any, are compatible.
# For instance, if RunContextWrapper expects a specific TutorContext type:
# from ai_tutor.context import TutorContext
# async def explain_concept(ctx: RunContextWrapper[TutorContext], ... )
# However, the provided snippet uses `Any` for `ctx`, so we'll stick to that unless more info is given. 