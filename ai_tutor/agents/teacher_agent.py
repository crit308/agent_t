import os
import openai

from agents import Agent, FileSearchTool, Runner, handoff, HandoffInputData
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import LessonPlan, LessonSection, LessonContent, SectionContent, ExplanationContent, Exercise
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
from ai_tutor.agents.utils import round_search_result_scores


def lesson_content_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Filter function for handoff from teacher to quiz creator agent.
    
    This ensures the quiz creator agent receives the lesson content.
    """
    print("Applying handoff filter to pass lesson content to quiz creator agent")
    print(f"HandoffInputData type: {type(handoff_data)}")
    
    try:
        # Apply the round_search_result_scores function with 15 decimal places max
        processed_data = round_search_result_scores(handoff_data, max_decimal_places=15)
        print("Applied score rounding to handoff data")
        return processed_data
    except Exception as e:
        print(f"Error in handoff filter: {e}")
        # Return the original data if there was an error
        return handoff_data


def create_teacher_agent(vector_store_id: str, api_key: str = None):
    """Create a teacher agent with access to the provided vector store."""
    
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Ensure OPENAI_API_KEY is set in the environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set for teacher agent!")
    else:
        print(f"Using OPENAI_API_KEY from environment for teacher agent")
    
    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=5,
        include_search_results=True,
    )
    
    print(f"Created FileSearchTool for teacher agent using vector store: {vector_store_id}")
    
    # Create the quiz creator agent to hand off to after teaching is complete
    quiz_creator_agent = create_quiz_creator_agent(api_key)
    
    # Define an on_handoff function to prepare the lesson content for the quiz creator agent
    def on_handoff_to_quiz_creator(ctx: RunContextWrapper[any], lesson_content: LessonContent) -> None:
        print(f"Handoff triggered with lesson content: {lesson_content.title}")
        print(f"Lesson content has {len(lesson_content.sections)} sections")
        # We don't need to do anything here, as the lesson content will be accessible to the quiz creator agent
    
    # Create the teacher agent with access to the file search tool and handoff to quiz creator
    teacher_agent = Agent(
        name="Lesson Teacher",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are an expert teacher. Your task is to create comprehensive lesson content 
        based on the lesson plan provided to you.
        
        You will receive a detailed lesson plan with sections and learning objectives.
        Your job is to create engaging, informative content for each section, including:
        
        1. Clear explanations of key concepts
        2. Illustrative examples
        3. Practical exercises with solutions
        4. Summaries of key points
        
        Your explanations should be thorough but accessible. Use examples that clarify 
        complex ideas. Design exercises that reinforce learning and test understanding.
        
        Use the file_search tool to find relevant information from the uploaded 
        documents to include in your content.
        
        IMPORTANT PROCESS INSTRUCTIONS:
        1. If you were handed off to from the Planner agent, you'll receive a LessonPlan object.
           Use that lesson plan to guide your content creation.
        2. Search for information about each concept in the plan using file_search
        3. After gathering all needed information, create a complete LessonContent object
        4. YOU MUST FIRST output a complete structured LessonContent object before proceeding.
        5. ONLY AFTER generating the complete lesson content, hand off to the Quiz Creator agent who
           will create the quiz based on your lesson content.
        
        FORMAT INSTRUCTIONS:
        - Your output MUST be a valid JSON-serializable LessonContent object
        - The LessonContent should include a title, introduction, sections, conclusion, and next_steps
        - Each section should have a title, introduction, explanations, exercises, and summary
        
        DO NOT:
        - Do not reference any tools or future steps in your output
        - Do not return incomplete content
        - Do not include text outside the LessonContent object
        - Do not attempt to hand off if you were called directly (not via handoff from Planner)
        
        SEQUENCE OF OPERATIONS:
        1. Create and output a complete LessonContent
        2. ONLY THEN use the transfer_to_quiz_creator tool IF you were handed off from Planner
        
        When using transfer_to_quiz_creator, you should pass the entire LessonContent object 
        that you've just generated. For example:
        
        transfer_to_quiz_creator(your_lesson_content_object)
        
        CRITICAL: 
        - ONLY output the complete LessonContent object as your final result
        - ONLY use transfer_to_quiz_creator if you were handed off from Planner
        - If you were called directly (not via handoff), just return the LessonContent
        """,
        tools=[file_search_tool],
        handoffs=[
            handoff(
                agent=quiz_creator_agent,
                on_handoff=on_handoff_to_quiz_creator,
                input_type=LessonContent,
                input_filter=lesson_content_handoff_filter,
                tool_description_override="Transfer to the Quiz Creator agent who will generate a comprehensive quiz based on your lesson content. Provide the complete LessonContent object as input."
            )
        ],
        output_type=LessonContent,
        model="o3-mini",
    )
    
    return teacher_agent


async def generate_lesson_content(lesson_plan: LessonPlan, vector_store_id: str, api_key: str = None) -> LessonContent:
    """Generate the full lesson content based on the provided lesson plan."""
    
    # Create the teacher agent
    agent = create_teacher_agent(vector_store_id, api_key)
    
    # Format the lesson plan as a string for the teacher agent
    lesson_plan_str = f"""
    LESSON PLAN:
    
    Title: {lesson_plan.title}
    Description: {lesson_plan.description}
    Target Audience: {lesson_plan.target_audience}
    Prerequisites: {', '.join(lesson_plan.prerequisites)}
    Total Estimated Duration: {lesson_plan.total_estimated_duration_minutes} minutes
    
    Sections:
    """
    
    for i, section in enumerate(lesson_plan.sections):
        lesson_plan_str += f"""
        Section {i+1}: {section.title}
        Estimated Duration: {section.estimated_duration_minutes} minutes
        
        Learning Objectives:
        """
        
        for j, objective in enumerate(section.objectives):
            lesson_plan_str += f"""
            Objective {j+1}: {objective.description}
            Priority: {objective.priority}
            """
        
        if hasattr(section, 'concepts_to_cover') and section.concepts_to_cover:
            lesson_plan_str += f"""
            Key Concepts: {', '.join(section.concepts_to_cover)}
            """
    
    # Instruction to use FileSearchTool
    lesson_plan_str += f"""
    IMPORTANT INSTRUCTIONS:
    
    1. For EACH section and concept, you MUST search the vector store using the file_search tool
       to gather accurate information before creating content.
       
    2. Use search queries that are directly related to the key concepts. For example, if a key
       concept is "Supervised Learning", search for "Supervised Learning" specifically.
       
    3. Create content ONLY after you have searched for and found information about each topic.
       
    4. If you cannot find information on a specific concept, note this in your explanation 
       but still provide basic information about the concept.
    
    5. YOUR OUTPUT MUST BE A VALID LESSONCONTENT OBJECT with the following structure:
       - title: String (lesson title)
       - introduction: String (general introduction to the lesson)
       - sections: Array of SectionContent objects, each with:
         - title: String (section title)
         - introduction: String (introduction to this section)
         - explanations: Array of ExplanationContent objects
         - exercises: Array of Exercise objects
         - summary: String (summary of key points)
       - conclusion: String (overall lesson conclusion)
       - next_steps: Array of strings (suggested next steps)
       
    6. IMPORTANT: Make sure to include at least one section with explanations and exercises.
       
    7. DO NOT attempt to use any handoff tools or create a quiz.
    
    8. YOUR OUTPUT MUST BE ONLY THE COMPLETE LESSONCONTENT OBJECT.
    
    NOTE: This function is being called directly (not via handoff). You're receiving a formatted
    lesson plan to use for content creation.
    """
    
    # Run the teacher agent with the lesson plan to get the LessonContent
    result = await Runner.run(agent, lesson_plan_str)
    
    # Get the lesson content
    try:
        lesson_content = result.final_output_as(LessonContent)
        print("Successfully generated LessonContent")
        
        # Verify we have valid sections
        if not lesson_content.sections or len(lesson_content.sections) == 0:
            raise ValueError("Generated LessonContent has no sections")
            
        return lesson_content
    except Exception as e:
        print(f"Error extracting LessonContent: {e}")
        # If we got a Quiz instead or another error, create a minimal LessonContent object
        return LessonContent(
            title=lesson_plan.title,
            introduction=f"Content for {lesson_plan.title}.",
            sections=[
                SectionContent(
                    title="Introduction to " + lesson_plan.title,
                    introduction="This is an introduction to the topic.",
                    explanations=[
                        ExplanationContent(
                            topic="Basic Concepts",
                            explanation="This section covers the fundamental concepts of the topic.",
                            examples=["Example 1", "Example 2"]
                        )
                    ],
                    exercises=[
                        Exercise(
                            question="What is the main purpose of this lesson?",
                            difficulty_level="Easy",
                            answer="To understand the basic concepts.",
                            explanation="This question tests your understanding of the lesson objectives."
                        )
                    ],
                    summary="This section introduced the basic concepts of the topic."
                )
            ],
            conclusion="Thank you for completing this lesson.",
            next_steps=["Explore more advanced topics", "Practice with exercises"]
        ) 