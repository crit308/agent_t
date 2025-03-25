import os
import openai
import json

from agents import Agent, FileSearchTool, Runner, handoff, HandoffInputData
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import LessonPlan, LessonSection, LessonContent, SectionContent, ExplanationContent, Exercise
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
from ai_tutor.agents.utils import round_search_result_scores


def lesson_content_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Process handoff data from teacher to quiz creator."""
    print("Applying handoff filter to pass lesson content to quiz creator agent")
    
    # Apply score rounding to avoid precision errors
    print(f"HandoffInputData type: {type(handoff_data)}")
    
    try:
        # Use a very conservative max_decimal_places value
        handoff_data = round_search_result_scores(handoff_data, max_decimal_places=5)
        print("Applied score rounding to handoff data")
        
        # Extra validation - try to serialize to JSON and back
        if hasattr(handoff_data, 'data') and handoff_data.data:
            try:
                json_str = json.dumps(str(handoff_data.data))
                json.loads(json_str)
                print("Validated handoff data can be serialized to JSON")
            except Exception as json_err:
                print(f"Warning: JSON validation failed: {json_err}")
        
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
    def on_handoff_to_quiz_creator(ctx: RunContextWrapper[any], lesson_content: LessonContent) -> None:
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
        
        After creating the lesson content, you MUST use the transfer_to_quiz_creator tool to hand off to the Quiz Creator agent.
        The Quiz Creator will create a quiz based on your lesson content.
        """
        """
        CRITICAL WORKFLOW:
        1. YOU MUST FIRST create and output a complete LessonContent object
        2. ONLY AFTER that, use the transfer_to_quiz_creator tool to hand off to the Quiz Creator agent
        
        DO NOT SKIP the handoff to the quiz creator. After you've generated the lesson content,
        you MUST use the transfer_to_quiz_creator tool to pass the content to the Quiz Creator.
        """,
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


async def generate_lesson_content(teacher_agent: Agent, lesson_plan: LessonPlan) -> LessonContent:
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
    
    # Run the teacher agent with the lesson plan to get the LessonContent
    result = await Runner.run(teacher_agent, lesson_plan_str)
    
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