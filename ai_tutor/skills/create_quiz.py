# from agents import function_tool # No longer used
from ai_tutor.skills import skill # Import correct decorator
# from ai_tutor.utils.agent_callers import call_quiz_creator_agent # No longer used
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.agents.models import QuizQuestion # Import the required Pydantic model
from ai_tutor.api_models import QuestionResponse
from ai_tutor.core.llm import LLMClient
import json
import logging
from typing import Optional, Dict, Any, List # Added List, Dict, Any
from pydantic import BaseModel, Field, ValidationError, validator # Added BaseModel, Field, validator
from ai_tutor.exceptions import ExecutorError, ToolInputError # Added ToolInputError

logger = logging.getLogger(__name__)

# Get the schema once
QUIZ_QUESTION_SCHEMA_JSON = json.dumps(QuizQuestion.model_json_schema(), indent=2)

# System prompt for the LLM that generates the quiz question content
# (This is distinct from the skill's own arguments)
QUIZ_GENERATION_SYSTEM_PROMPT = f"""You are an AI assistant specialized in creating educational quiz questions.
Your task is to generate a single quiz question based on the provided topic and instructions.

The output MUST be a single, valid JSON object with two keys:
1.  'quiz_question': A JSON object conforming exactly to the QuizQuestion schema detailed below.
2.  'whiteboard_actions': An OPTIONAL list of CanvasObjectSpec objects to visually present the Multiple Choice Question (MCQ).

QuizQuestion Schema (for the 'quiz_question' key):
```json
{QUIZ_QUESTION_SCHEMA_JSON}
```

Guidelines for 'quiz_question':
- Ensure 'question' is a non-empty string.
- Ensure 'options' is a list of at least 2 strings (typically 4 for multiple choice).
- Ensure 'correct_answer_index' is a 0-based integer corresponding to an option in the 'options' list.
- Ensure 'explanation' is a non-empty string.
- 'difficulty' should typically be 'Easy', 'Medium', or 'Hard'.
- 'related_section' should be the topic name you were given.

Guidelines for 'whiteboard_actions' when generating an MCQ:
- If you provide 'whiteboard_actions' for an MCQ, it MUST be a list of individual canvas object specifications.
- **DO NOT** use a single object with `"kind":"radio"` and an "options" array (e.g., `{{"kind":"radio", "id": "any_id", "options": ["A", "B"]}}`). This format is deprecated.
- The list of objects MUST represent the MCQ as follows:
    1.  **Question Text Object**:
        *   A single object with `"kind": "text"`.
        *   It must have a unique `"id"` string (e.g., "mcq-[question_id]-text").
        *   It must contain the question string in a `"text"` field.
        *   Include `"metadata": {{"role": "question", "question_id": "[question_id]"}}`, where `[question_id]` is a unique identifier for this entire question (e.g., a short UUID).
    2.  **Option Objects (for each option in 'quiz_question.options')**:
        *   For each option, provide two objects: a selectable circle and a text label.
        *   Let `[question_id]` be the same unique identifier used for the question text object, and `[option_idx]` be the 0-based index of the option.
        *   **Option Selector Object (Circle)**:
            *   `"kind": "circle"`
            *   `"id": f"mcq-[question_id]-opt-[option_idx]-radio"`
            *   `"metadata": {{"role": "option_selector", "question_id": "[question_id]", "option_id": [option_idx]}}`
            *   Include properties for position (e.g., x, y) and appearance (e.g., radius, fill, stroke).
        *   **Option Label Object (Text)**:
            *   `"kind": "text"`
            *   `"id": f"mcq-[question_id]-opt-[option_idx]-text"`
            *   `"text": "[Option text, e.g., 'A. Evaporation']"`
            *   `"metadata": {{"role": "option_label", "question_id": "[question_id]", "option_id": [option_idx]}}`
            *   Include properties for position (e.g., x, y) and appearance (e.g., fontSize, fill).
- Ensure all `[question_id]` parts within the IDs and metadata for a single MCQ are identical and unique for that MCQ.
- If no drawing is needed, omit the 'whiteboard_actions' key or provide an empty list.

Respond ONLY with the single JSON object containing 'quiz_question' and optionally 'whiteboard_actions'.
"""

class CreateQuizArgs(BaseModel):
    topic: str = Field(..., min_length=1, description="The topic for the quiz question.")
    instructions: Optional[str] = Field(default=None, description="Specific instructions for generating the quiz question.")
    # num_questions: int = Field(default=1, gt=0, description="Number of questions to generate.") # Future: for generating multiple questions
    # question_type: Literal["multiple_choice", "free_response"] = Field(default="multiple_choice") # Future: to specify question type

    @validator('topic', 'instructions')
    def check_not_just_whitespace(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Field cannot be only whitespace if provided.")
        return v

@skill
async def create_quiz(ctx: RunContextWrapper[TutorContext], **kwargs) -> QuestionResponse:
    """Skill that uses LLMClient to generate a single QuizQuestion and optionally whiteboard actions for the given topic."""
    try:
        args = CreateQuizArgs(**kwargs)
    except ValidationError as e:
        raise ToolInputError(f"Invalid arguments for create_quiz: {e}")

    topic = args.topic
    instructions = args.instructions if args.instructions else "Ensure the question is clear and relevant to the topic."

    logger.info(f"[Skill create_quiz] Generating question for topic='{topic}', instructions='{instructions}'")
    llm = LLMClient()

    system_msg = {"role": "system", "content": QUIZ_GENERATION_SYSTEM_PROMPT}
    user_msg = {"role": "user", "content": f"Generate a quiz question about the topic '{topic}'. Specific instructions: {instructions}"}

    try:
        logger.info("create_quiz: Calling LLM to generate quiz content...")
        messages_for_llm: List[Dict[str, str]] = [system_msg, user_msg] # type: ignore
        llm_response_content = await llm.chat(messages_for_llm)
        logger.info(f"create_quiz: LLM raw response for quiz content: {llm_response_content}")
        
        parsed_llm_output: Dict[str, Any]
        if isinstance(llm_response_content, str):
            try:
                parsed_llm_output = json.loads(llm_response_content)
            except json.JSONDecodeError:
                start_idx = llm_response_content.find('{')
                end_idx = llm_response_content.rfind('}')
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    json_candidate = llm_response_content[start_idx : end_idx + 1]
                    try:
                        parsed_llm_output = json.loads(json_candidate)
                    except json.JSONDecodeError as e_inner:
                        logger.error(f"Failed to parse extracted JSON from LLM response: {json_candidate}. Error: {e_inner}")
                        raise ExecutorError("LLM response for quiz creation was not valid JSON after extraction.") from e_inner
                else:
                    logger.error(f"Could not find JSON delimiters in LLM response for quiz: {llm_response_content}")
                    raise ExecutorError("LLM response for quiz creation was not valid JSON and delimiters not found.")
        elif isinstance(llm_response_content, dict):
            parsed_llm_output = llm_response_content
        else:
            logger.error(f"Unexpected LLM response type for quiz creation: {type(llm_response_content)}")
            raise ExecutorError("LLM response for quiz creation was not a string or dictionary.")

        quiz_data = parsed_llm_output.get("quiz_question")
        whiteboard_actions = parsed_llm_output.get("whiteboard_actions")

        if not quiz_data or not isinstance(quiz_data, dict):
            logger.error(f"LLM response missing 'quiz_question' key or it's not a dict: {parsed_llm_output}")
            raise ExecutorError("LLM did not return a valid 'quiz_question' object in the response.")
        
        try:
            quiz_question = QuizQuestion.model_validate(quiz_data)
        except ValidationError as e_val:
            logger.error(f"[Skill create_quiz] Failed to validate quiz_data against QuizQuestion model: {e_val}. Data: {quiz_data}")
            raise ExecutorError(f"LLM-generated quiz_question did not match schema: {e_val}") from e_val
        
        payload = QuestionResponse(
            question_type="multiple_choice", # Assuming multiple_choice for now as per prompt
            question_data=quiz_question,
            context_summary=None
        )
        setattr(payload, 'whiteboard_actions', whiteboard_actions)
        return payload
        
    except ExecutorError: # Re-raise ExecutorErrors to be caught by dispatch if needed
        raise
    except Exception as e:
        logger.error(f"[Skill create_quiz] Unexpected error during LLM call or processing for topic '{topic}': {e}", exc_info=True)
        # Wrap unexpected errors in ExecutorError for consistent handling by dispatcher
        raise ExecutorError(f"Unexpected failure in create_quiz skill: {e}") from e

    # This block might be unreachable if all paths raise, but included for completeness
    # except Exception as e:
    #     logger.error(f"Error in create_quiz skill for topic '{topic}': {e}", exc_info=True)
    #     raise ExecutorError(f"Unexpected error in create_quiz skill: {e}") from e 