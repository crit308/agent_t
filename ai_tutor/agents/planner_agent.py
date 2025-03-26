from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import openai
import json

from agents import Agent, FileSearchTool, function_tool, handoff, HandoffInputData
from agents.run_context import RunContextWrapper

from ai_tutor.agents.teacher_agent import create_teacher_agent
from ai_tutor.agents.models import LearningObjective, LessonSection, LessonPlan
from ai_tutor.agents.utils import round_search_result_scores, fix_search_result_scores, limit_decimal_places, process_handoff_data


def lesson_plan_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Process handoff data from lesson planner to lesson teacher."""
    print("Applying handoff filter to pass lesson plan to teacher agent")
    
    # Apply score rounding to avoid precision errors
    print(f"HandoffInputData type: {type(handoff_data)}")
    
    try:
        # Direct aggressive processing of search result scores
        if hasattr(handoff_data, 'input_history') and isinstance(handoff_data.input_history, tuple):
            # Convert to list for easy modification
            input_history_list = list(handoff_data.input_history)
            
            # Go through each item looking for file search results
            for i, item in enumerate(input_history_list):
                if isinstance(item, dict):
                    # Fix all search result scores precisely
                    if 'type' in item and item['type'] in ('file_search_call', 'file_search_results'):
                        if 'results' in item and isinstance(item['results'], list):
                            for result in item['results']:
                                if isinstance(result, dict) and 'score' in result:
                                    # Use multiple techniques to ensure precision is limited
                                    score = float(result['score'])
                                    # Round to 15 places
                                    score = round(score, 15)
                                    # Format to string with exactly 15 places then back
                                    score = float(f"{score:.15f}")
                                    # Double check decimal places
                                    str_val = str(score)
                                    if '.' in str_val and len(str_val.split('.')[1]) > 15:
                                        int_part = str_val.split('.')[0]
                                        decimal_part = str_val.split('.')[1][:15]
                                        score = float(f"{int_part}.{decimal_part}")
                                    result['score'] = score
                                    print(f"Aggressive decimal limiting on score: {score}")
            
            # Update handoff_data with fixed input_history
            input_history = tuple(input_history_list)
        else:
            # Make sure it's still a tuple even if input_history isn't one
            input_history = handoff_data.input_history if handoff_data.input_history is not None else ()
        
        # Get pre_handoff_items and new_items, ensuring they're tuples
        pre_handoff_items = tuple(handoff_data.pre_handoff_items) if hasattr(handoff_data, 'pre_handoff_items') and handoff_data.pre_handoff_items is not None else ()
        new_items = tuple(handoff_data.new_items) if hasattr(handoff_data, 'new_items') and handoff_data.new_items is not None else ()
        
        # Apply comprehensive processing via utils
        try:
            processed_data = process_handoff_data(handoff_data)
            print("Successfully processed handoff data with precision limits")
            return processed_data
        except Exception as process_err:
            print(f"Error in comprehensive processing: {process_err}, falling back to direct approach")
            # If that fails, try to create a new HandoffInputData with just our input_history fix
            try:
                return HandoffInputData(
                    input_history=input_history,
                    pre_handoff_items=pre_handoff_items,  # Always a tuple
                    new_items=new_items  # Always a tuple
                )
            except Exception as handoff_err:
                print(f"Error creating new HandoffInputData: {handoff_err}")
                # Create minimal empty HandoffInputData as last resort
                try:
                    return HandoffInputData(
                        input_history=() if not input_history else input_history,
                        pre_handoff_items=(),
                        new_items=()
                    )
                except Exception as final_err:
                    print(f"Final error creating HandoffInputData: {final_err}")
                    # Last resort, return the original
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