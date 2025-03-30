import os
import openai
import json

from agents import Agent, FileSearchTool, Runner, handoff, HandoffInputData, function_tool, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import LessonPlan, LessonSection, LessonContent
from typing import List, Callable, Optional, Any, Dict
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
from ai_tutor.agents.utils import process_handoff_data, RoundingModelWrapper


def lesson_content_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Process handoff data from lesson teacher to quiz creator."""
    print("Applying handoff filter to pass lesson content to quiz creator agent")
    
    try:
        processed_data = process_handoff_data(handoff_data)
        print("Successfully processed handoff data with precision limits")
        return processed_data
    except Exception as e:
        print(f"Error in handoff filter: {e}")
        print("Returning original handoff data without processing")
        return handoff_data


def create_teacher_agent(vector_store_id: str, api_key: str = None):
    """Create a simplified lesson content generation agent."""
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
        # Debug serialization if needed
        # try:
        #     json_str = json.dumps(lesson_content.model_dump())
        #     print(f"Serialized lesson content: {len(json_str)} characters")
        # except Exception as e:
        #     print(f"Warning: Serialization test failed: {e}")
    
    # Instantiate the base model provider and get the base model
    provider: ModelProvider = OpenAIProvider()
    base_model = provider.get_model("o3-mini")
    
    # Create the teacher agent
    teacher_agent = Agent(
        name="Lesson Teacher",
        instructions=prompt_with_handoff_instructions("""
        You are an expert educational content creator. Your task is to create lesson content based on the lesson plan provided.

        GUIDELINES:
        1. Use the file_search tool to research the topics outlined in the lesson plan.
        2. Synthesize the information into a single, coherent body of text for the lesson.
        3. The text should cover the key concepts and objectives from the lesson plan.
        4. Structure the text logically, perhaps using Markdown for headings or lists if appropriate within the single text block.
        5. Ensure the content is clear, accurate, and engaging for the target audience.

        CRITICAL OUTPUT FORMAT:
        - Your output MUST be a JSON object with exactly two fields: "title" (string) and "text" (string).
        - The "title" should be the overall lesson title.
        - The "text" field should contain the complete, synthesized lesson content as a single string.

        WORKFLOW:
        1. Research content using file_search tool based on the lesson plan.
        2. Create and output the complete LessonContent JSON object { "title": "...", "text": "..." }.
        3. IMMEDIATELY AFTER outputting the JSON, use the transfer_to_quiz_creator tool to hand off the LessonContent object.

        YOUR FINAL OUTPUT MUST BE ONLY THE VALID LessonContent JSON OBJECT, FOLLOWED BY THE HANDOFF TOOL CALL.
        """),
        tools=[file_search_tool],
        handoffs=[
            handoff(
                agent=quiz_creator_agent,
                on_handoff=on_handoff_to_quiz_creator,
                input_type=LessonContent,
                input_filter=lesson_content_handoff_filter,
                tool_description_override="Transfer to the Quiz Creator agent who will create a quiz based on your lesson content. Provide the complete LessonContent object { \"title\": \"...\", \"text\": \"...\" } as input."
            )
        ],
        output_type=LessonContent,
        model=RoundingModelWrapper(base_model),
    )
    
    return teacher_agent


def create_teacher_agent_without_handoffs(vector_store_id: str, api_key: str = None):
    """Create a simplified lesson content generation agent WITHOUT handoff capability."""
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=5,
        include_search_results=True,
    )
    
    # Instantiate the base model provider and get the base model
    provider: ModelProvider = OpenAIProvider()
    base_model = provider.get_model("o3-mini")
    
    # Create the teacher agent without handoffs
    teacher_agent = Agent(
        name="Lesson Teacher",
        instructions="""
        You are an expert educational content creator. Your task is to create lesson content based on the lesson plan provided.

        GUIDELINES:
        1. Use the file_search tool to research the topics outlined in the lesson plan.
        2. Synthesize the information into a single, coherent body of text for the lesson.
        3. The text should cover the key concepts and objectives from the lesson plan.
        4. Structure the text logically, perhaps using Markdown for headings or lists if appropriate within the single text block.
        5. Ensure the content is clear, accurate, and engaging for the target audience.

        CRITICAL OUTPUT FORMAT:
        - Your output MUST be a JSON object with exactly two fields: "title" (string) and "text" (string).
        - The "title" should be the overall lesson title.
        - The "text" field should contain the complete, synthesized lesson content as a single string.

        YOUR OUTPUT MUST BE ONLY THE VALID LessonContent JSON OBJECT { "title": "...", "text": "..." }.
        DO NOT USE ANY HANDOFF TOOLS.
        """,
        tools=[file_search_tool],
        # No handoffs in this version
        output_type=LessonContent,
        model=RoundingModelWrapper(base_model),
    )
    
    return teacher_agent


async def generate_lesson_content(teacher_agent: Agent, lesson_plan: LessonPlan, context=None) -> LessonContent:
    """Generate the simplified lesson content based on the provided lesson plan."""

    # Format the lesson plan as a string for the teacher agent
    lesson_plan_str = f"""
    LESSON PLAN:

    Title: {lesson_plan.title}
    Description: {lesson_plan.description}
    Target Audience: {lesson_plan.target_audience}
    Prerequisites: {', '.join(lesson_plan.prerequisites)}
    Total Estimated Duration: {lesson_plan.total_estimated_duration_minutes} minutes

    Key Topics/Objectives from Sections:
    """
    for i, section in enumerate(lesson_plan.sections):
        lesson_plan_str += f"\nSection {i+1}: {section.title}\n"
        lesson_plan_str += f"  Objectives: {', '.join([obj.description for obj in section.objectives])}\n"
        if hasattr(section, 'concepts_to_cover') and section.concepts_to_cover:
            lesson_plan_str += f"  Concepts: {', '.join(section.concepts_to_cover)}\n"

    lesson_plan_str += f"""

    IMPORTANT INSTRUCTIONS:
    1. Use the file_search tool to research the topics above.
    2. Synthesize this information into a single, coherent 'text' field.
    3. Generate a suitable 'title' for the lesson.
    4. YOUR OUTPUT MUST BE ONLY A VALID JSON OBJECT: {{ "title": "...", "text": "..." }}.
    5. DO NOT attempt to use any handoff tools.
    """
    
    # Setup RunConfig for tracing
    from agents import RunConfig
    
    run_config = None
    if context and hasattr(context, 'session_id'):
        run_config = RunConfig(
            workflow_name="AI Tutor - Content Creation",
            group_id=context.session_id
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
        print("Successfully generated simplified LessonContent")
        # Basic validation
        if not lesson_content.title or not lesson_content.text:
             raise ValueError("Generated LessonContent is missing title or text")
        return lesson_content
    except Exception as e:
        print(f"Error extracting simplified LessonContent: {e}")
        # --- SIMPLIFIED FALLBACK ---
        return LessonContent(
            title=lesson_plan.title,
            text=f"Error generating content for {lesson_plan.title}. Please check logs. Basic concepts should be covered here based on the plan description: {lesson_plan.description}."
        )
        # --- End SIMPLIFIED FALLBACK --- 