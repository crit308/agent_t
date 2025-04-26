# from agents import function_tool # No longer used
from ai_tutor.skills import skill # Import correct decorator
# from ai_tutor.utils.agent_callers import call_quiz_creator_agent # No longer used
from ai_tutor.context import TutorContext
from agents.run_context import RunContextWrapper
from ai_tutor.agents.models import QuizQuestion # Import the required Pydantic model
from ai_tutor.core.llm import LLMClient
import json
import logging
from pydantic import ValidationError
from ai_tutor.exceptions import ExecutorError # Correct import

logger = logging.getLogger(__name__)

# Get the schema once
QUIZ_QUESTION_SCHEMA_JSON = json.dumps(QuizQuestion.model_json_schema(), indent=2)

QUIZ_QUESTION_SYSTEM_PROMPT = '''
You are an AI assistant specialized in creating educational quiz questions.
Your task is to generate a single multiple-choice question based on the provided topic and instructions.

The output MUST be a single, valid JSON object conforming exactly to the following Pydantic model schema:

```json
{
  "question": "string (The question text)",
  "options": [
    "string (Option 1)",
    "string (Option 2)",
    "string (Option 3)",
    "string (Option 4)"
  ],
  "correct_answer_index": "integer (0-based index of the correct option)",
  "explanation": "string (Brief explanation why the answer is correct)",
  "difficulty": "string (Enum: 'Easy', 'Medium', 'Hard')",
  "related_section": "string (The topic name provided)"
}
```

Guidelines:
- Ensure there are exactly 4 options unless specified otherwise.
- The `correct_answer_index` must correspond to the correct option in the `options` list.
- The `related_section` field should contain the topic name you were given.
- Do NOT include any text before or after the JSON object.
- Ensure the JSON is well-formed and all values are of the correct type.
'''

@skill
async def create_quiz(ctx: RunContextWrapper[TutorContext], topic: str, instructions: str) -> QuizQuestion:
    """Skill that uses LLMClient to generate a single QuizQuestion Pydantic object for the given topic."""
    logger.info(f"create_quiz skill started for topic: {topic}") # Log Start
    logger.info(f"[Skill create_quiz] Generating question for topic='{topic}', instructions='{instructions}'")
    llm = LLMClient()

    system_msg = {"role": "system", "content": QUIZ_QUESTION_SYSTEM_PROMPT}
    user_msg = {"role": "user", "content": f"Generate a multiple-choice question about the topic '{topic}'. Specific instructions: {instructions}"}

    try:
        logger.info("create_quiz: Calling LLM...") # Log Before LLM call
        response_json_str = await llm.chat(
            messages=[system_msg, user_msg],
            response_format={"type": "json_object"} # Request JSON output explicitly
        )
        # Log After LLM call (use response_json_str directly as it might be dict or str)
        logger.info(f"create_quiz: LLM raw response: {response_json_str}") 
        # logger.debug(f"[Skill create_quiz] LLM raw response: {response_json_str}") # Keep debug or remove duplicate

        quiz_data = None # Initialize quiz_data
        # Parse the JSON string
        try:
            if isinstance(response_json_str, dict): # If already parsed (depends on LLMClient)
                quiz_data = response_json_str
                logger.info(f"create_quiz: LLM response was already a dict: {quiz_data}")
            else:
                # Clean up potential markdown fences
                cleaned = response_json_str.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                quiz_data = json.loads(cleaned)
                logger.info(f"create_quiz: Parsed JSON: {quiz_data}") # Log After JSON parsing
                # logger.info(f"[Skill create_quiz] Parsed quiz_data: {quiz_data}") # Keep this or remove duplicate
        except json.JSONDecodeError as e:
            logger.error(f"create_quiz: Caught Exception TYPE: {type(e).__name__} - {e}. Response: {response_json_str}", exc_info=True) # Log Exception
            logger.error(f"[Skill create_quiz] Failed to parse JSON response: {e}. Response: {response_json_str}")
            raise ExecutorError(f"LLM returned invalid JSON: {e}") from e # Re-raise with custom type

        # Validate against the Pydantic model
        try:
            quiz_question = QuizQuestion.model_validate(quiz_data)
            # Log After Pydantic validation
            logger.info(f"create_quiz: Validated QuizQuestion: {quiz_question.model_dump_json(indent=2)}") 
            logger.info(f"[Skill create_quiz] Successfully generated and validated QuizQuestion for topic '{topic}'")
            return quiz_question
        except ValidationError as e:
            logger.error(f"create_quiz: Caught Exception TYPE: {type(e).__name__} - {e}. Data: {quiz_data}", exc_info=True) # Log Exception
            logger.error(f"[Skill create_quiz] Failed to validate JSON against QuizQuestion model: {e}. Data: {quiz_data}")
            raise ExecutorError(f"LLM response did not match QuizQuestion schema: {e}") from e # Re-raise with custom type

    except Exception as e:
        logger.error(f"create_quiz: Caught Exception TYPE: {type(e).__name__} - {e}", exc_info=True) # Log Exception
        logger.error(f"[Skill create_quiz] Error during LLM call or processing: {e}", exc_info=True)
        raise ExecutorError(f"Failed to create quiz question during LLM call or processing: {e}") from e # Re-raise with custom type

    # This block might be unreachable if all paths raise, but included for completeness
    # except Exception as e:
    #     logger.error(f"Error in create_quiz skill for topic '{topic}': {e}", exc_info=True)
    #     raise ExecutorError(f"Unexpected error in create_quiz skill: {e}") from e 