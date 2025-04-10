import os
# import openai
import json
from typing import Any

# Use ADK imports
from google.adk.agents import LLMAgent
# Keep OpenAIProvider import only if still used directly, otherwise remove
# from agents.models.openai_provider import OpenAIProvider
# from agents.run_context import RunContextWrapper # Remove if context not needed
# from agents.extensions.handoff_prompt import prompt_with_handoff_instructions # Remove handoff

from ai_tutor.agents.models import LessonContent, Quiz, QuizCreationResult, QuizQuestion
# from ai_tutor.agents.utils import RoundingModelWrapper # Remove if not used with ADK model


def quiz_to_teacher_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Process handoff data from quiz creator to quiz teacher."""
    print("DEBUG: Entering quiz_to_teacher_handoff_filter (Quiz Creator -> Quiz Teacher)")
    
    try:
        processed_data = process_handoff_data(handoff_data)
        print("DEBUG: Exiting quiz_to_teacher_handoff_filter")
        return processed_data
    except Exception as e:
        print(f"ERROR in quiz_to_teacher_handoff_filter: {e}")
        return handoff_data  # Fallback


# This function now defines the AGENT used AS A TOOL
def create_quiz_creator_agent(api_key: str = None):
    """Create a basic quiz creator agent (now as an ADK LLMAgent)."""

    # ADK typically uses Application Default Credentials (ADC) or GOOGLE_APPLICATION_CREDENTIALS
    # Direct API key setting might not be needed or handled differently.
    # Check ADK documentation for preferred authentication methods.
    # if api_key:
    #     os.environ["OPENAI_API_KEY"] = api_key # Example - May not apply to ADK
    # api_key = os.environ.get("OPENAI_API_KEY")
    # if not api_key:
    #     print("WARNING: API Key not set for quiz creator agent! Check ADC/Env Vars.")
    # else:
    #     print(f"Using API Key from environment for quiz creator agent (Check if needed for ADK)")

    # Use ADK models
    model_name = "gemini-1.5-flash" # Or other ADK supported model

    # Create the quiz creator agent as an ADK LLMAgent
    quiz_creator_agent = LLMAgent(
        name="QuizCreatorToolAgent", # Name reflects tool usage
        instructions="""
        You are an expert educational assessment designer. Your task is to create quiz questions based on the specific instructions provided in the prompt (e.g., topic, number of questions, difficulty).

        Guidelines for creating effective quiz questions:
        1. Create a mix of easy, medium, and hard questions that cover all key concepts from the lesson
        2. Ensure questions are clear, unambiguous, and test understanding rather than just memorization
        3. For multiple-choice questions, create plausible distractors that represent common misconceptions
        4. Include detailed explanations for the correct answers that reinforce learning
        5. Distribute questions across all sections of the lesson to ensure comprehensive coverage
        6. Target approximately 2-3 questions per lesson section

        CRITICAL REQUIREMENTS:
        1. Follow the instructions in the prompt regarding the number of questions requested.
        2. Each multiple-choice question MUST have exactly 4 options unless specified otherwise.
        3. Set an appropriate passing score (typically 70% of total points)
        4. Ensure total_points equals the number of questions
        5. Set a reasonable estimated_completion_time_minutes (typically 1-2 minutes per question)

        FORMAT REQUIREMENTS FOR YOUR OUTPUT:
        - Your final output MUST be a single, valid `QuizCreationResult` JSON object.
        - If successful, set `status` to "created".
        - If you created a single question, put the `QuizQuestion` object in the `question` field.
        - If you created multiple questions, put the full `Quiz` object in the `quiz` field.
        - If failed, set `status` to "failed" and provide details.

        Focus ONLY on generating the requested question(s) based on the input instructions.
        """,
        # Specify output schema for ADK agent
        output_schema=QuizCreationResult,
        # Define input schema if this agent is called as a tool with structured input
        # Example: Define a Pydantic model like `QuizCreationRequest` if needed
        # input_schema=QuizCreationRequest,
        model=model_name, # Pass model name string
    )

    return quiz_creator_agent


# Remove create_quiz_creator_agent_with_teacher_handoff and generate_quiz
# as quiz creation is now handled by calling this agent as a tool. 