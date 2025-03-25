from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import openai
import json

from agents import Agent, FileSearchTool, function_tool, handoff, HandoffInputData
from agents.run_context import RunContextWrapper

from ai_tutor.agents.teacher_agent import create_teacher_agent
from ai_tutor.agents.models import LearningObjective, LessonSection, LessonPlan
from ai_tutor.agents.utils import round_search_result_scores


def lesson_plan_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Filter function for handoff from planner to teacher agent.
    
    This ensures the teacher agent receives the lesson plan.
    """
    print("Applying handoff filter to pass lesson plan to teacher agent")
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


def create_planner_agent(vector_store_id: str, api_key: str = None):
    """Create a planner agent with access to the provided vector store."""
    
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Ensure OPENAI_API_KEY is set in the environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set!")
    else:
        print(f"Using OPENAI_API_KEY from environment for planner agent")
    
    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=5,
        include_search_results=True,
    )
    
    print(f"Created FileSearchTool for vector store: {vector_store_id}")
    
    # Create the teacher agent to hand off to after planning is complete
    teacher_agent = create_teacher_agent(vector_store_id, api_key)
    
    # Define an on_handoff function to prepare the lesson plan for the teacher agent
    def on_handoff_to_teacher(ctx: RunContextWrapper[Any], lesson_plan: LessonPlan) -> None:
        print(f"Handoff triggered with lesson plan: {lesson_plan.title}")
        print(f"Lesson plan has {len(lesson_plan.sections)} sections")
        # We don't need to do anything here, as the lesson plan will be accessible to the teacher agent
    
    # Create the planner agent with access to the file search tool and handoff to teacher agent
    planner_agent = Agent(
        name="Lesson Planner",
        instructions="""
        You are an expert curriculum designer and educator. Your task is to create a 
        well-structured lesson plan based on the documents that have been uploaded.
        
        Follow these steps IN ORDER:
        1. First, analyze the uploaded documents to understand their content and structure.
           Use the file_search tool to search through the documents.
        2. Identify the key topics, concepts, and skills that should be taught.
        3. Create a coherent lesson plan that organizes the content into logical sections.
        4. For each section, define clear learning objectives, key concepts, and estimate 
           how long it would take to teach.
        5. Consider the appropriate sequence for teaching the material.
        6. Consider who the target audience might be and any prerequisites they should know.
        7. YOU MUST FIRST output a complete structured LessonPlan object before proceeding.
        8. ONLY AFTER generating the complete lesson plan, hand off to the Teacher agent who
           will create the detailed lesson content based on your plan.
        
        Your lesson plan should be comprehensive but focused on the most important aspects 
        of the material. Break complex topics into manageable sections.
        
        IMPORTANT: You MUST use the file_search tool to find information in the uploaded documents.
        Start by searching for general terms like "introduction", "overview", or specific topics
        mentioned in the documents.
        
        CRITICAL: DO NOT hand off to the teacher agent before you have created and output
        a complete LessonPlan object. The teacher agent needs your lesson plan to create content.
        
        SEQUENCE OF OPERATIONS:
        1. Search through documents
        2. Create and output a complete LessonPlan
        3. ONLY THEN use the transfer_to_lesson_teacher tool
        
        When using transfer_to_lesson_teacher, you should pass the entire LessonPlan object 
        that you've just generated. For example:
        
        transfer_to_lesson_teacher(your_lesson_plan_object)
        """,
        tools=[file_search_tool],
        handoffs=[
            handoff(
                agent=teacher_agent,
                on_handoff=on_handoff_to_teacher,
                input_type=LessonPlan,
                input_filter=lesson_plan_handoff_filter,
                tool_description_override="Transfer to the Lesson Teacher agent who will generate detailed lesson content based on your lesson plan. Provide the complete LessonPlan object as input."
            )
        ],
        output_type=LessonPlan,
        model="o3-mini",
    )
    
    return planner_agent 