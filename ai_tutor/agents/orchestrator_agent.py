import os
from typing import List, Optional

from agents import Agent, Runner, RunConfig, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.run_context import RunContextWrapper

from ai_tutor.context import TutorContext # Use the enhanced context
# Import tool functions (assuming they will exist in a separate file)
from ai_tutor.tools.orchestrator_tools import (
    call_teacher_explain,
    call_quiz_creator_mini,
    call_quiz_teacher_evaluate,
    call_planner_get_next_topic,
    update_user_model,
    get_user_model_status,
    # Add other tools as needed
)

# Import models needed for type hints if tools return them
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem

def create_orchestrator_agent(api_key: str = None) -> Agent[TutorContext]:
    """Creates the Orchestrator Agent for the AI Tutor."""

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    provider: ModelProvider = OpenAIProvider()
    # Use a capable model for orchestration logic
    base_model = provider.get_model("gpt-4o") # Using gpt-4o or similar recommended

    orchestrator_tools = [
        call_teacher_explain,
        call_quiz_creator_mini,
        call_quiz_teacher_evaluate,
        call_planner_get_next_topic,
        update_user_model,
        get_user_model_status,
    ]

    orchestrator_agent = Agent[TutorContext]( # Specify context type
        name="TutorOrchestrator",
        instructions=prompt_with_handoff_instructions("""
        You are the conductor of an AI tutoring session. Your primary goal is to help the user learn the material effectively by guiding them through a lesson plan.

        CONTEXT:
        - You have access to the overall LessonPlan via context.
        - You can read and update the UserModelState (tracking concept mastery, current topic) via context using tools.
        - You know the user's last input/action.

        CORE WORKFLOW:
        1.  **Assess State:** Check the user's last input and the current `UserModelState` (use `get_user_model_status`).
        2.  **Determine Next Step:** Consult the `LessonPlan` (use `call_planner_get_next_topic`) to identify the next logical topic or section. If the user struggled previously (check user model), consider revisiting or providing different activities for the *current* topic.
        3.  **Select Action:** Based on the assessment and next step, choose the best pedagogical action:
            *   **Explain:** If moving to a new topic or re-explaining, use `call_teacher_explain`.
            *   **Quiz:** If checking understanding after an explanation, use `call_quiz_creator_mini`.
            *   **Feedback:** If the user just answered a quiz, evaluate it (using evaluation tool - currently placeholder) and provide feedback.
            *   **Summarize:** (Future Tool) Ask the user to summarize.
            *   **Question:** (Future Tool) Prompt the user if they have questions.
        4.  **Execute Action:** Call the appropriate tool to perform the action.
        5.  **Update State:** Use `update_user_model` to record the outcome of the interaction (e.g., user answered correctly/incorrectly, topic explained).
        6.  **Formulate Response:** Your final output for this turn should be the content to present directly to the user (e.g., the explanation text from the teacher, the quiz question, or feedback on their answer).

        PRINCIPLES:
        - **Be Adaptive:** Adjust the plan based on user performance recorded in the `UserModelState`. If mastery is high, move faster. If struggling, provide more support or different explanations.
        - **Be Interactive:** Prefer shorter cycles of explanation followed by interaction (quiz, summary) over long lectures.
        - **Be Clear:** Your final output should be ready for the user interface to display.
        """),
        tools=orchestrator_tools,
        model=base_model,
        # Output type is dynamic (string explanation, QuizQuestion, string feedback),
        # so leave as None/str and handle based on the action taken.
        # Or define a complex Union type if strict output is desired.
        output_type=None,
    )
    return orchestrator_agent 