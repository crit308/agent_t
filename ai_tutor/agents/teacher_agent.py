from pydantic import BaseModel, Field
from typing import List
import os
import openai

from agents import Agent, FileSearchTool, Runner

from ai_tutor.agents.planner_agent import LessonPlan, LessonSection


class ExplanationContent(BaseModel):
    """Content explaining a concept or topic."""
    topic: str = Field(description="The topic being explained")
    explanation: str = Field(description="A clear, detailed explanation of the topic")
    examples: List[str] = Field(description="Examples that illustrate the topic")


class Exercise(BaseModel):
    """An exercise for the student to complete."""
    question: str = Field(description="The exercise question or prompt")
    difficulty_level: str = Field(description="Easy, Medium, or Hard")
    answer: str = Field(description="The answer or solution to the exercise")
    explanation: str = Field(description="Explanation of how to solve the exercise")


class SectionContent(BaseModel):
    """The full content for a section of the lesson."""
    title: str = Field(description="The title of the section")
    introduction: str = Field(description="Introduction to the section")
    explanations: List[ExplanationContent] = Field(description="Explanations of key concepts")
    exercises: List[Exercise] = Field(description="Exercises for practice")
    summary: str = Field(description="A summary of key points from the section")


class LessonContent(BaseModel):
    """The complete lesson content created by the teacher agent."""
    title: str = Field(description="The title of the lesson")
    introduction: str = Field(description="Introduction to the overall lesson")
    sections: List[SectionContent] = Field(description="Content for each section of the lesson")
    conclusion: str = Field(description="Conclusion summarizing the lesson")
    next_steps: List[str] = Field(description="Suggested next steps for continued learning")


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
    
    # Create the teacher agent with access to the file search tool
    teacher_agent = Agent(
        name="Lesson Teacher",
        instructions="""
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
        """,
        tools=[file_search_tool],
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
    
    I will evaluate your response based on how well you use the file_search tool to find 
    and incorporate information from the uploaded documents.
    """
    
    # Run the teacher agent with the lesson plan
    result = await Runner.run(agent, lesson_plan_str)
    
    # Return the lesson content
    return result.final_output_as(LessonContent) 