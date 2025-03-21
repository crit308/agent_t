import os
import openai

from agents import Agent, FileSearchTool, Runner, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from ai_tutor.agents.planner_agent import LessonPlan, LessonSection
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
from ai_tutor.agents.models import LessonContent


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
    
    # Create the quiz creator agent for handoff
    quiz_creator_agent = create_quiz_creator_agent(api_key)
    
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
        
        Format your response as a structured LessonContent object.
        
        IMPORTANT: You MUST use the file_search tool to search for every key concept before
        creating content. For each section, search for relevant information in the documents.
        Without this step, you cannot create accurate content.
        
        AFTER CREATING THE LESSON CONTENT: You should use the transfer_to_quiz_creator tool to 
        hand off the lesson content to the Quiz Creator agent, which will create a quiz based 
        on the lesson content you've created.
        """,
        tools=[file_search_tool],
        handoffs=[handoff(quiz_creator_agent)],
        output_type=LessonContent,
        model="gpt-4o",
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
    
    Prerequisites:
    {', '.join(lesson_plan.prerequisites)}
    
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
            {j+1}. {objective.title} (Priority: {objective.priority})
               {objective.description}
            """
            
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
    
    5. After you have created the lesson content, use the transfer_to_quiz_creator tool
       to hand off the completed lesson content to the Quiz Creator agent. This will
       automatically create a quiz based on your lesson.
    
    I will evaluate your response based on how well you use the file_search tool to find 
    and incorporate information from the uploaded documents.
    """
    
    # Run the teacher agent with the lesson plan
    result = await Runner.run(agent, lesson_plan_str)
    
    # Return the lesson content
    return result.final_output_as(LessonContent) 