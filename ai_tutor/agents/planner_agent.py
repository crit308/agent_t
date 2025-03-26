from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import openai
import json

from agents import Agent, FileSearchTool, function_tool, handoff, HandoffInputData
from agents.run_context import RunContextWrapper

from ai_tutor.agents.teacher_agent import create_teacher_agent
from ai_tutor.agents.models import LearningObjective, LessonSection, LessonPlan
from ai_tutor.agents.utils import round_search_result_scores, fix_search_result_scores


def lesson_plan_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Process handoff data from lesson planner to lesson teacher."""
    print("Applying handoff filter to pass lesson plan to teacher agent")
    
    # Apply score rounding to avoid precision errors
    print(f"HandoffInputData type: {type(handoff_data)}")
    
    try:
        # Import fix_search_result_scores function for direct precision control
        from ai_tutor.agents.utils import fix_search_result_scores
        
        # First apply the new direct fix for all data structures in handoff_data
        # This will recursively process all nested dictionaries and arrays
        handoff_data = fix_search_result_scores(handoff_data, max_decimal_places=8)
        print("Applied direct fix to all data structures in handoff_data")
        
        # For compatibility, also apply the regular score rounding
        from ai_tutor.agents.utils import round_search_result_scores
        handoff_data = round_search_result_scores(handoff_data, max_decimal_places=5)
        print("Applied score rounding to handoff data")
        
        # Extra validation - try to serialize to JSON and back
        if hasattr(handoff_data, 'data') and handoff_data.data:
            try:
                import json
                from ai_tutor.agents.models import PrecisionControlEncoder
                
                # Use our custom encoder to enforce precision limits
                json_str = json.dumps(handoff_data.data, cls=PrecisionControlEncoder, max_decimals=8)
                json.loads(json_str)
                print("Validated handoff data can be serialized to JSON")
            except Exception as json_err:
                print(f"Warning: JSON validation failed: {json_err}")
        
        return handoff_data
    except Exception as e:
        print(f"Error in handoff filter: {e}")
        print("Returning original handoff data without processing")
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
        8. IMMEDIATELY AFTER generating the complete lesson plan, YOU MUST hand off to the Teacher agent who
           will create the detailed lesson content based on your plan.
        
        Your lesson plan should be comprehensive but focused on the most important aspects 
        of the material. Break complex topics into manageable sections.
        
        IMPORTANT: You MUST use the file_search tool to find information in the uploaded documents.
        Start by searching for general terms like "introduction", "overview", or specific topics
        mentioned in the documents.
        
        CRITICAL WORKFLOW INSTRUCTIONS:
        1. Search through documents using file_search tool
        2. Create and output a complete LessonPlan object
        3. IMMEDIATELY use the transfer_to_lesson_teacher tool with your lesson plan as input
        
        The handoff to the teacher agent is REQUIRED and MUST happen immediately after
        you generate the lesson plan. DO NOT skip this step under any circumstances.
        
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