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


@function_tool
def read_knowledge_base(ctx: RunContextWrapper[Any]) -> str:
    """Read the Knowledge Base file that was created by the Document Analyzer agent.
    This file contains analysis of all the documents in the vector store, including key concepts,
    terms, and other important information extracted from the documents.
    
    Returns:
        The complete content of the Knowledge Base file.
    """
    try:
        with open("Knowledge Base", "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with fallback encoding if UTF-8 fails
        with open("Knowledge Base", "r", encoding="latin-1") as f:
            return f.read()
    except FileNotFoundError:
        return "Knowledge Base file not found. The Document Analyzer may not have completed yet."


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
        
        REQUIRED STRICT WORKFLOW - YOU MUST FOLLOW THESE STEPS IN THIS EXACT ORDER:
        
        STEP 1: KNOWLEDGE BASE ANALYSIS
        - First, use the read_knowledge_base tool to get the document analysis created by the Document Analyzer agent.
        - Thoroughly review this analysis, which provides key concepts, terminology, and document structure.
        - YOU MUST COMPLETE THIS STEP BEFORE PROCEEDING. DO NOT SKIP THIS STEP UNDER ANY CIRCUMSTANCES.
        
        STEP 2: DETAILED DOCUMENT SEARCH
        - After reviewing the Knowledge Base, use the file_search tool to search through the documents.
        - Search for specific topics identified in the Knowledge Base and any other relevant information.
        - Perform multiple searches to gather comprehensive information.
        - YOU MUST COMPLETE THIS STEP BEFORE PROCEEDING TO CREATING THE LESSON PLAN.
        
        STEP 3: LESSON PLAN CREATION
        - Based on the Knowledge Base analysis and document searches, create a comprehensive lesson plan.
        - Your lesson plan must include:
          * Clear learning objectives for each section
          * Logical sequence of sections
          * Key concepts to cover in each section
          * Estimated duration for each section
          * Target audience considerations
          * Any prerequisites
        - Output the lesson plan as a complete structured LessonPlan object.
        - YOUR FINAL OUTPUT BEFORE THE HANDOFF MUST ONLY BE THE VALID LessonPlan JSON OBJECT.
        
        STEP 4: TEACHER HANDOFF
        - IMMEDIATELY after outputting the LessonPlan object, use the `transfer_to_lesson_teacher` tool.
        - Pass the complete LessonPlan object to this tool.
        - This handoff is REQUIRED and MUST happen after you generate the lesson plan.
        
        CRITICAL REMINDERS:
        1. YOU MUST READ THE KNOWLEDGE BASE FIRST - this is a mandatory requirement.
        2. YOU MUST THEN USE FILE SEARCH before creating the lesson plan.
        3. The sequence of steps must be strictly followed: Knowledge Base → File Search → Lesson Plan → Handoff.
        4. Do not skip any steps or change the order of operations.
        
        Call transfer_to_lesson_teacher with the generated LessonPlan object after completing steps 1-3.
        """),
        tools=[read_knowledge_base, file_search_tool],
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