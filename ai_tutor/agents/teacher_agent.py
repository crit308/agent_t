import os
import openai
import json

from agents import Agent, FileSearchTool, Runner, handoff, HandoffInputData, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import LessonPlan, LessonSection, LessonContent, SectionContent, ExplanationContent, Exercise
from typing import List, Callable, Optional, Any, Dict
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
from ai_tutor.agents.utils import process_handoff_data


def lesson_content_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Process handoff data from lesson teacher to quiz creator."""
    print("Applying handoff filter to pass lesson content to quiz creator agent")
    
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


def create_teacher_agent(vector_store_id: str, api_key: str = None):
    """Create a lesson content generation agent with handoff capability to Quiz Creator.
    
    Args:
        vector_store_id: The vector store ID to use for file search
        api_key: OpenAI API key
        
    Returns:
        Teacher agent with handoff to Quiz Creator
    """
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Create the quiz creator agent to hand off to
    quiz_creator_agent = create_quiz_creator_agent(api_key)
    
    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=5,
        include_search_results=True,
    )
    
    # Define an on_handoff function for when teacher hands off to quiz creator
    async def on_handoff_to_quiz_creator(ctx: RunContextWrapper[any], lesson_content: LessonContent) -> None:
        print(f"Handoff triggered from teacher to quiz creator with lesson: {lesson_content.title}")
        print(f"Lesson has {len(lesson_content.sections)} sections")
        
        # Debug serialization
        try:
            # Test if lesson content can be serialized
            json_str = json.dumps(lesson_content.model_dump())
            print(f"Serialized lesson content: {len(json_str)} characters")
        except Exception as e:
            print(f"Warning: Serialization test failed: {e}")
    
    # Create the teacher agent
    teacher_agent = Agent(
        name="Lesson Teacher",
        instructions=prompt_with_handoff_instructions("""
        You are an expert educational content creator specializing in developing comprehensive lesson content.
        Your task is to create detailed lesson content based on the lesson plan provided to you.
        
        Guidelines for creating effective lesson content:
        1. Thoroughly research each section using the file_search tool before creating content
        2. Create clear explanations with relevant examples for each concept
        3. Include practical exercises that reinforce understanding
        4. Ensure content flows logically from basic to more advanced concepts
        5. Summarize key points at the end of each section
        6. Provide a comprehensive conclusion and suggest next steps for further learning
        
        Use the file_search tool to search for relevant information in the uploaded documents.
        
        CRITICAL WORKFLOW INSTRUCTIONS:
        1. FIRST create and output a complete LessonContent object
        2. IMMEDIATELY AFTER THAT use the transfer_to_quiz_creator tool to hand off to the Quiz Creator agent
        
        Workflow:
        1. Research content using file_search tool
        2. Create and output a complete LessonContent object
        3. Call transfer_to_quiz_creator(your_lesson_content) to hand off to the Quiz Creator
        
        YOUR OUTPUT MUST BE ONLY A VALID LESSON CONTENT OBJECT FOLLOWED BY THE HANDOFF.
        
        After outputting your LessonContent object, you MUST use the transfer_to_quiz_creator handoff tool.
        """),
        tools=[file_search_tool],
        handoffs=[
            handoff(
                agent=quiz_creator_agent,
                on_handoff=on_handoff_to_quiz_creator,
                input_type=LessonContent,
                input_filter=lesson_content_handoff_filter,
                tool_description_override="Transfer to the Quiz Creator agent who will create a quiz based on your lesson content. Provide the complete LessonContent object as input."
            )
        ],
        output_type=LessonContent,
        model="o3-mini",
    )
    
    return teacher_agent


def create_teacher_agent_without_handoffs(vector_store_id: str, api_key: str = None):
    """Create a lesson content generation agent WITHOUT handoff capability to Quiz Creator.
    
    This version should be used when a Quiz has already been created via handoff chain,
    to avoid creating duplicate workflows.
    
    Args:
        vector_store_id: The vector store ID to use for file search
        api_key: OpenAI API key
        
    Returns:
        Teacher agent without handoff capabilities
    """
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=5,
        include_search_results=True,
    )
    
    # Create the teacher agent without handoffs
    teacher_agent = Agent(
        name="Lesson Teacher",
        instructions="""
        You are an expert educational content creator specializing in developing comprehensive lesson content.
        Your task is to create detailed lesson content based on the lesson plan provided to you.
        
        Guidelines for creating effective lesson content:
        1. Thoroughly research each section using the file_search tool before creating content
        2. Create clear explanations with relevant examples for each concept
        3. Include practical exercises that reinforce understanding
        4. Ensure content flows logically from basic to more advanced concepts
        5. Summarize key points at the end of each section
        6. Provide a comprehensive conclusion and suggest next steps for further learning
        
        Use the file_search tool to search for relevant information in the uploaded documents.
        
        YOUR OUTPUT MUST BE ONLY A VALID LESSON CONTENT OBJECT.
        """,
        tools=[file_search_tool],
        # No handoffs in this version
        output_type=LessonContent,
        model="o3-mini",
    )
    
    return teacher_agent


async def generate_lesson_content(teacher_agent: Agent, lesson_plan: LessonPlan, context=None) -> LessonContent:
    """Generate the full lesson content based on the provided lesson plan."""
    
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
       
    7. DO NOT attempt to use any handoff tools or create a quiz. Since you're being called directly 
       and not via a handoff from the planner agent, DO NOT try to hand off to the Quiz Creator.
       Just return the LessonContent object directly.
    
    8. YOUR OUTPUT MUST BE ONLY THE COMPLETE LESSONCONTENT OBJECT.
    
    NOTE: This function is being called directly (not via handoff). You're receiving a formatted
    lesson plan to use for content creation. DO NOT use any handoff tools even if they are available.
    """
    
    # Setup RunConfig for tracing
    from agents import RunConfig
    
    run_config = None
    if context and hasattr(context, 'session_id'):
        run_config = RunConfig(
            workflow_name="AI Tutor - Content Creation",
            group_id=context.session_id # Link traces within the same session
        )
    
    # Run the teacher agent with the lesson plan to get the LessonContent
    result = await Runner.run(
        teacher_agent, 
        lesson_plan_str,
        run_config=run_config,
        context=context
    )
    
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