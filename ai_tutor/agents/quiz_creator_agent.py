import os
import openai

from agents import Agent, Runner

from ai_tutor.agents.models import LessonContent, Quiz


def create_quiz_creator_agent(api_key: str = None):
    """Create a quiz creator agent."""
    
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
        """,
        output_type=Quiz,
        model="gpt-4o",
    )
    
    return quiz_creator_agent


async def generate_quiz(lesson_content: LessonContent, api_key: str = None) -> Quiz:
    """Generate a quiz based on the provided lesson content."""
    # Create the quiz creator agent
    agent = create_quiz_creator_agent(api_key)
    
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
    """
    
    # Run the quiz creator agent with the lesson content
    result = await Runner.run(agent, lesson_content_str)
    
    # Return the quiz
    return result.final_output_as(Quiz) 