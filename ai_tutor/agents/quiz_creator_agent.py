import os
import openai
import json

from agents import Agent, Runner, handoff, HandoffInputData
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import LessonContent, Quiz
from ai_tutor.agents.utils import round_search_result_scores
from ai_tutor.agents.quiz_teacher_agent import create_quiz_teacher_agent


def quiz_to_teacher_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Process handoff data from quiz creator to quiz teacher."""
    print("Applying handoff filter to pass quiz to teacher agent")
    
    # Apply score rounding to avoid precision errors
    print(f"HandoffInputData type: {type(handoff_data)}")
    
    try:
        # Use a very conservative max_decimal_places value
        handoff_data = round_search_result_scores(handoff_data, max_decimal_places=5)
        print("Applied score rounding to handoff data")
        
        # Extra validation - try to serialize to JSON and back
        if hasattr(handoff_data, 'data') and handoff_data.data:
            try:
                json_str = json.dumps(str(handoff_data.data))
                json.loads(json_str)
                print("Validated handoff data can be serialized to JSON")
            except Exception as json_err:
                print(f"Warning: JSON validation failed: {json_err}")
        
        return handoff_data
    except Exception as e:
        print(f"Error in handoff filter: {e}")
        print("Returning original handoff data without processing")
        return handoff_data


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
    
    # Create the quiz creator agent
    quiz_creator_agent = Agent(
        name="Quiz Creator",
        instructions="""
        You are an expert educational assessment designer specialized in creating effective quizzes.
        Your task is to create a comprehensive quiz based on the lesson content provided to you.
        
        Guidelines for creating effective quiz questions:
        1. Create a mix of easy, medium, and hard questions that cover all key concepts from the lesson
        2. Ensure questions are clear, unambiguous, and test understanding rather than just memorization
        3. For multiple-choice questions, create plausible distractors that represent common misconceptions
        4. Include detailed explanations for the correct answers that reinforce learning
        5. Distribute questions across all sections of the lesson to ensure comprehensive coverage
        6. Target approximately 2-3 questions per lesson section
        
        Format your response as a structured Quiz object.
        
        YOUR OUTPUT MUST BE ONLY A VALID QUIZ OBJECT.
        """,
        output_type=Quiz,
        model="o3-mini",
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
    
    # Create the quiz teacher agent to hand off to
    quiz_teacher_agent = create_quiz_teacher_agent(api_key)
    
    # Define an on_handoff function for when quiz creator hands off to quiz teacher
    def on_handoff_to_quiz_teacher(ctx: RunContextWrapper[any], quiz: Quiz) -> None:
        print(f"Handoff triggered from quiz creator to quiz teacher: {quiz.title}")
        print(f"Quiz has {len(quiz.questions)} questions")
    
    # Create the quiz creator agent
    quiz_creator_agent = Agent(
        name="Quiz Creator",
        instructions="""
        You are an expert educational assessment designer specialized in creating effective quizzes.
        Your task is to create a comprehensive quiz based on the lesson content provided to you.
        
        Guidelines for creating effective quiz questions:
        1. Create a mix of easy, medium, and hard questions that cover all key concepts from the lesson
        2. Ensure questions are clear, unambiguous, and test understanding rather than just memorization
        3. For multiple-choice questions, create plausible distractors that represent common misconceptions
        4. Include detailed explanations for the correct answers that reinforce learning
        5. Distribute questions across all sections of the lesson to ensure comprehensive coverage
        6. Target approximately 2-3 questions per lesson section
        
        Format your response as a structured Quiz object.
        
        YOUR OUTPUT MUST BE ONLY A VALID QUIZ OBJECT.
        
        After generating the quiz, use the transfer_to_quiz_teacher tool to hand off to the Quiz Teacher agent
        which will evaluate user responses and provide feedback.
        """,
        handoffs=[
            handoff(
                agent=quiz_teacher_agent,
                on_handoff=on_handoff_to_quiz_teacher,
                input_type=Quiz,
                input_filter=quiz_to_teacher_handoff_filter,
                tool_description_override="Transfer to the Quiz Teacher agent who will evaluate user responses and provide feedback. Provide the complete Quiz object as input."
            )
        ],
        output_type=Quiz,
        model="o3-mini",
    )
    
    return quiz_creator_agent


async def generate_quiz(lesson_content: LessonContent, api_key: str = None, enable_teacher_handoff: bool = False) -> Quiz:
    """Generate a quiz based on the provided lesson content.
    
    Args:
        lesson_content: The lesson content to base the quiz on
        api_key: Optional OpenAI API key
        enable_teacher_handoff: Whether to enable handoff to the quiz teacher agent
        
    Returns:
        Quiz object
    """
    # Create the quiz creator agent
    if enable_teacher_handoff:
        agent = create_quiz_creator_agent_with_teacher_handoff(api_key)
        print("Created quiz creator agent with teacher handoff capability")
    else:
        agent = create_quiz_creator_agent(api_key)
        print("Created quiz creator agent without teacher handoff capability")
    
    # Check if the lesson content has sections
    if not hasattr(lesson_content, 'sections') or len(lesson_content.sections) == 0:
        print("Warning: Lesson content has no sections. Creating a minimal quiz.")
        # Create a minimal quiz with just the title and description
        return Quiz(
            title=f"Quiz on {lesson_content.title}",
            description="This is a quiz based on the lesson content.",
            lesson_title=lesson_content.title,
            questions=[],
            passing_score=0,
            total_points=0,
            estimated_completion_time_minutes=0
        )
    
    # Format the lesson content as a string for the quiz creator agent
    lesson_content_str = f"""
    LESSON CONTENT:
    
    Title: {lesson_content.title}
    Introduction: {lesson_content.introduction}
    
    Sections:
    """
    
    for i, section in enumerate(lesson_content.sections):
        lesson_content_str += f"""
        Section {i+1}: {section.title}
        Introduction: {section.introduction}
        
        Key Concepts:
        """
        
        for j, explanation in enumerate(section.explanations):
            lesson_content_str += f"""
            Concept {j+1}: {explanation.topic}
            Explanation: {explanation.explanation}
            Examples: {', '.join(explanation.examples)}
            """
        
        lesson_content_str += f"""
        Summary: {section.summary}
        """
    
    lesson_content_str += f"""
    Conclusion: {lesson_content.conclusion}
    
    INSTRUCTIONS:
    Based on this lesson content, create a comprehensive quiz that tests understanding of the key concepts.
    Create approximately 2-3 questions per section, with a mix of difficulty levels.
    Ensure your questions cover the most important concepts from each section.
    
    Your output should ONLY be a valid Quiz object with the following structure:
    - title: The quiz title
    - description: Brief description of the quiz
    - lesson_title: Title of the lesson this quiz is based on
    - questions: Array of QuizQuestion objects
    - passing_score: Minimum points to pass
    - total_points: Total possible points
    - estimated_completion_time_minutes: Estimated time to complete
    """
    
    # Run the quiz creator agent with the lesson content
    result = await Runner.run(agent, lesson_content_str)
    
    # Return the quiz
    try:
        quiz = result.final_output_as(Quiz)
        return quiz
    except Exception as e:
        print(f"Error parsing quiz output: {e}")
        # Return a minimal quiz if parsing fails
        return Quiz(
            title=f"Quiz on {lesson_content.title}",
            description="This is a quiz based on the lesson content.",
            lesson_title=lesson_content.title,
            questions=[],
            passing_score=0,
            total_points=0,
            estimated_completion_time_minutes=0
        ) 