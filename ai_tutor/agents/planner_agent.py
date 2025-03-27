from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import openai
import json
# import re # No longer needed if not used elsewhere

from agents import Agent, FileSearchTool, function_tool, handoff, HandoffInputData, ModelProvider
from agents.models.openai_provider import OpenAIProvider # Assuming you use this
from agents.run_context import RunContextWrapper

from ai_tutor.agents.teacher_agent import create_teacher_agent
from ai_tutor.agents.models import LearningObjective, LessonSection, LessonPlan
from ai_tutor.agents.utils import process_handoff_data, RoundingModelWrapper # Import the wrapper
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


def lesson_plan_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Process handoff data from lesson planner to lesson teacher."""
    print("DEBUG: Entering lesson_plan_handoff_filter (Planner -> Teacher)")
    
    try:
        processed_data = process_handoff_data(handoff_data)
        print("DEBUG: Exiting lesson_plan_handoff_filter")
        return processed_data
    except Exception as e:
        print(f"ERROR in lesson_plan_handoff_filter: {e}")
        return handoff_data  # Fallback


def create_planner_agent(vector_store_id: str):
    """Create a planner agent with access to the provided vector store."""
    
    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=5,
        include_search_results=True,
    )
    
    print(f"Created FileSearchTool for vector store: {vector_store_id}")
    
    # Create the teacher agent to hand off to after planning is complete
    teacher_agent = create_teacher_agent(vector_store_id)
    
    # Define an on_handoff function to prepare the lesson plan for the teacher agent
    async def on_handoff_to_teacher(ctx: RunContextWrapper[Any], lesson_plan: LessonPlan) -> None:
        print(f"Planner handing off lesson plan: {lesson_plan.title}")
        print(f"Lesson plan has {len(lesson_plan.sections)} sections")
        # We don't need to do anything here, as the lesson plan will be accessible to the teacher agent

    # Instantiate the base model provider and get the base model
    provider: ModelProvider = OpenAIProvider() # Or your specific provider
    base_model = provider.get_model("o3-mini") # Or the desired model name
    
    # Create the planner agent with access to the file search tool and handoff to teacher agent
    planner_agent = Agent(
        name="Lesson Planner",
        instructions=prompt_with_handoff_instructions("""
        You are an expert curriculum designer. Your task is to create a 
        well-structured lesson plan based on the documents that have been uploaded.
        
        Follow these steps IN ORDER:
        1. First, analyze the uploaded documents to understand their content and structure.
           Use the file_search tool to search through the documents. Perform multiple searches if necessary.
        2. Identify the key topics, concepts, and skills that should be taught.
        3. Create a coherent lesson plan that organizes the content into logical sections.
        4. For each section, define clear learning objectives, key concepts, and estimate 
           how long it would take to teach.
        5. Consider the appropriate sequence for teaching the material.
        6. Consider who the target audience might be and any prerequisites they should know.
        7. YOU MUST FIRST output a complete structured LessonPlan object before proceeding.
           **Your final response before the handoff MUST ONLY be the valid LessonPlan JSON object.**
        8. IMMEDIATELY AFTER outputting the LessonPlan object, YOU MUST use the 
           `transfer_to_lesson_teacher` tool to hand off to the Teacher agent who
           will create the detailed lesson content based on your plan.
        
        IMPORTANT: You MUST use the file_search tool to find information in the uploaded documents.
        Start by searching for general terms like "introduction", "overview", or specific topics
        mentioned in the documents.
        
        CRITICAL WORKFLOW INSTRUCTIONS:
        1. Search through documents using file_search tool
        2. Create and output a complete LessonPlan object
           **(Ensure ONLY the LessonPlan JSON is the output before the tool call)**
        3. IMMEDIATELY use the `transfer_to_lesson_teacher` tool with the generated lesson plan as input
        
        The handoff to the teacher agent is REQUIRED and MUST happen immediately after
        you generate the lesson plan. DO NOT skip this step under any circumstances.
        Ensure you pass the complete LessonPlan object to the `transfer_to_lesson_teacher` tool.
        
        Call transfer_to_lesson_teacher with the generated LessonPlan object.
        """),
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
        model=RoundingModelWrapper(base_model), # Wrap the base model
    )
    
    return planner_agent 