import os
import openai
import json
from typing import Any

from agents import Agent, Runner, handoff, HandoffInputData, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from ai_tutor.agents.models import LessonContent, Quiz, QuizCreationResult, QuizQuestion
from ai_tutor.agents.utils import process_handoff_data, RoundingModelWrapper


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


def create_quiz_creator_agent(api_key: str = None):
    """Create a basic quiz creator agent without handoff capability."""
    
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Ensure OPENAI_API_KEY is set in the environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set for quiz creator agent!")
    else:
        print(f"Using OPENAI_API_KEY from environment for quiz creator agent")
    
    # Instantiate the base model provider and get the base model
    provider: ModelProvider = OpenAIProvider()
    base_model = provider.get_model("gpt-4o")
    
    # Create the quiz creator agent
    quiz_creator_agent = Agent[Any](
        name="Quiz Creator",
        instructions="""
        You are an expert educational assessment designer specialized in creating effective quizzes.
        Your task is to create quiz questions based on the specific instructions provided in the prompt (e.g., topic, number of questions, difficulty).
        
        Guidelines for creating effective quiz questions:
        1. Create a mix of easy, medium, and hard questions that cover all key concepts from the lesson
        2. Ensure questions are clear, unambiguous, and test understanding rather than just memorization
        3. For multiple-choice questions, create plausible distractors that represent common misconceptions
        4. Include detailed explanations for the correct answers that reinforce learning
        5. Distribute questions across all sections of the lesson to ensure comprehensive coverage
        6. Target approximately 2-3 questions per lesson section
        
        CRITICAL REQUIREMENTS:
        1. Follow the instructions in the prompt regarding the number of questions. If asked for ONE question, create only ONE.
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

        Do NOT ask follow-up questions or manage conversation flow. Just create the requested quiz/question(s).
        """,
        output_type=QuizCreationResult,
        model=base_model,
    )
    
    return quiz_creator_agent


def create_quiz_creator_agent_with_teacher_handoff(api_key: str = None):
    """Create a quiz creator agent with handoff capability to the quiz teacher."""
    
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Ensure OPENAI_API_KEY is set in the environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set for quiz creator agent!")
    else:
        print(f"Using OPENAI_API_KEY from environment for quiz creator agent")
    
    # Define an on_handoff function for when quiz creator hands off to quiz teacher
    async def on_handoff_to_quiz_teacher(ctx: RunContextWrapper[any], quiz: Quiz) -> None:
        print(f"Handoff triggered from quiz creator to quiz teacher: {quiz.title}")
        print(f"Quiz has {len(quiz.questions)} questions")
    
    # Instantiate the base model provider and get the base model
    provider: ModelProvider = OpenAIProvider()
    base_model = provider.get_model("gpt-4o")
    
    # Create the quiz creator agent
    quiz_creator_agent = Agent[Any](
        name="Quiz Creator",
        instructions=prompt_with_handoff_instructions("""
        You are an expert educational assessment designer specialized in creating effective quizzes.
        Your task is to create quiz questions based on the specific instructions provided in the prompt (e.g., topic, number of questions, difficulty).
        
        Guidelines for creating effective quiz questions:
        1. Create a mix of easy, medium, and hard questions that cover all key concepts from the lesson
        2. Ensure questions are clear, unambiguous, and test understanding rather than just memorization
        3. For multiple-choice questions, create plausible distractors that represent common misconceptions
        4. Include detailed explanations for the correct answers that reinforce learning
        5. Distribute questions across all sections of the lesson to ensure comprehensive coverage
        6. Target approximately 2-3 questions per lesson section
        
        CRITICAL REQUIREMENTS:
        1. Follow the instructions in the prompt regarding the number of questions. If asked for ONE question, create only ONE.
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

        Do NOT ask follow-up questions or manage conversation flow. Just create the requested quiz/question(s).
        """),
        handoffs=[
            # Remove handoff - Orchestrator calls Quiz Teacher tool directly if needed
        ],
        output_type=QuizCreationResult,
        model=base_model,
    )
    
    return quiz_creator_agent


async def generate_quiz(lesson_content: LessonContent, api_key: str = None, enable_teacher_handoff: bool = False, context=None) -> Quiz:
    """Generate a quiz based on the simplified lesson content."""
    # Create the quiz creator agent
    if enable_teacher_handoff:
        agent = create_quiz_creator_agent_with_teacher_handoff(api_key)
        print("Created quiz creator agent with teacher handoff capability")
    else:
        agent = create_quiz_creator_agent(api_key)
        print("Created quiz creator agent without teacher handoff capability")

    # --- SIMPLIFIED PROMPT FORMATTING ---
    lesson_content_str = f"""
    LESSON CONTENT:

    Title: {lesson_content.title}

    Text:
    {lesson_content.text}
    --- End of Text ---

    INSTRUCTIONS:
    Based on the lesson text provided above (titled '{lesson_content.title}'), create a comprehensive quiz that tests understanding of the key concepts discussed.
    Create approximately 5-10 questions covering the most important information.
    Distribute questions across the topics mentioned in the text.
    Assign a relevant 'related_section' title (e.g., based on headings you infer from the text, or just use the main lesson title if unsure).

    Your output should ONLY be a valid Quiz object structure.
    """
    # --- End SIMPLIFIED PROMPT FORMATTING ---

    from agents import RunConfig # Keep RunConfig import
    run_config = None
    if context and hasattr(context, 'session_id'):
        run_config = RunConfig(
            workflow_name="AI Tutor - Quiz Creation",
            group_id=context.session_id
        )

    result = await Runner.run(
        agent,
        lesson_content_str,
        run_config=run_config,
        context=context
    )

    try:
        quiz = result.final_output_as(Quiz)
        # --- Add validation for related_section ---
        if quiz and quiz.questions:
             for q in quiz.questions:
                 if not q.related_section:
                     q.related_section = lesson_content.title # Default if missing
        # --- End validation ---
        return quiz
    except Exception as e:
        print(f"Error parsing quiz output: {e}")
        # Minimal quiz fallback
        return Quiz(
            title=f"Quiz on {lesson_content.title}",
            description="Quiz based on the lesson content.",
            lesson_title=lesson_content.title,
            questions=[],
            passing_score=0,
            total_points=0,
            estimated_completion_time_minutes=0
        ) 