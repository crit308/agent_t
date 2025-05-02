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
async def create_quiz(ctx: RunContextWrapper[TutorContext], topic: str, instructions: str) -> QuestionResponse:
    """Skill that uses LLMClient to generate a single QuizQuestion and optionally whiteboard actions for the given topic."""
    logger.info(f"[Skill create_quiz] Generating question for topic='{topic}', instructions='{instructions}'")
    llm = LLMClient()

    system_msg = {
        "role": "system",
        "content": (
            "You are an AI assistant specialized in creating educational quiz questions. "
            "Your task is to generate a single multiple-choice question based on the provided topic and instructions. "
            "The output MUST be a single, valid JSON object with two keys: "
            "'quiz_question': (a QuizQuestion object as described below), and, optionally, 'whiteboard_actions': (a list of CanvasObjectSpec objects to visually present the MCQ using draw_text, draw_shape, etc). "
            "If no drawing is needed, omit the 'whiteboard_actions' key. "
            "QuizQuestion schema: {"
            "  'question': str, 'options': List[str], 'correct_answer_index': int, 'explanation': str, 'difficulty': str, 'related_section': str "
            "} "
            "Respond ONLY with the JSON object."
        )
    }
    user_msg = {"role": "user", "content": f"Generate a multiple-choice question about the topic '{topic}'. Specific instructions: {instructions}"}

    try:
        logger.info("create_quiz: Calling LLM...")
        llm_response = await llm.chat([system_msg, user_msg])
        logger.info(f"create_quiz: LLM raw response: {llm_response}")
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
        quiz_data = parsed.get("quiz_question")
        whiteboard_actions = parsed.get("whiteboard_actions")
        if not quiz_data:
            raise ExecutorError("LLM did not return a 'quiz_question' key in the response.")
        # Validate against the Pydantic model
        try:
            quiz_question = QuizQuestion.model_validate(quiz_data)
        except ValidationError as e:
            logger.error(f"[Skill create_quiz] Failed to validate JSON against QuizQuestion model: {e}. Data: {quiz_data}")
            raise ExecutorError(f"LLM response did not match QuizQuestion schema: {e}") from e
        # Build QuestionResponse payload
        payload = QuestionResponse(
            question_type="multiple_choice",
            question_data=quiz_question,
            context_summary=None
        )
        # Attach whiteboard_actions as a non-model attribute for the executor to pick up
        setattr(payload, 'whiteboard_actions', whiteboard_actions)
        return payload
    except Exception as e:
        logger.error(f"[Skill create_quiz] Error during LLM call or processing: {e}", exc_info=True)
        raise ExecutorError(f"Failed to create quiz question during LLM call or processing: {e}") from e

    # This block might be unreachable if all paths raise, but included for completeness
    # except Exception as e:
    #     logger.error(f"Error in create_quiz skill for topic '{topic}': {e}", exc_info=True)
    #     raise ExecutorError(f"Unexpected error in create_quiz skill: {e}") from e 