from __future__ import annotations

import os
import openai
import json

from agents import Agent, FileSearchTool, Runner, handoff, HandoffInputData, function_tool, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import LessonPlan, LessonSection, LessonContent
from typing import List, Callable, Optional, Any, Dict, TYPE_CHECKING
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
from ai_tutor.agents.utils import process_handoff_data, RoundingModelWrapper

if TYPE_CHECKING:
    from ai_tutor.context import TutorContext


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


async def generate_lesson_content(
    teacher_agent: Agent['TutorContext'],  # Use forward reference string
    lesson_plan: LessonPlan,
    topic_to_explain: Optional[str] = None,  # Added parameter
    context: Optional['TutorContext'] = None  # Use forward reference string
) -> LessonContent:
    """Generate simplified lesson content. If topic_to_explain is provided,
       focuses the content generation on that specific topic using the lesson plan
       and analysis from context for background. Otherwise, generates broader content."""

    if not context:
         raise ValueError("TutorContext is required for generating lesson content.")

    # --- Build the prompt ---
    lesson_plan_str = f"""
    LESSON PLAN CONTEXT (Use this for structure and objectives):

    Title: {lesson_plan.title}
    Description: {lesson_plan.description}
    Target Audience: {lesson_plan.target_audience}
    Prerequisites: {', '.join(lesson_plan.prerequisites)}
    Total Estimated Duration: {lesson_plan.total_estimated_duration_minutes} minutes
    Sections Overview:"""
    for i, section in enumerate(lesson_plan.sections):
        lesson_plan_str += f"\nSection {i+1}: {section.title}\n"
        lesson_plan_str += f"  Objectives: {', '.join([obj.description for obj in section.objectives])}\n"
        lesson_plan_str += f"  Duration: {section.estimated_duration_minutes} minutes\n"
        if section.concepts_to_cover:
            lesson_plan_str += f"  Key Concepts: {', '.join(section.concepts_to_cover)}\n"
    lesson_plan_str += "\n--- End of Lesson Plan Context ---\n"

    # Add document analysis context if available
    analysis_str = ""
    if context.analysis_result:
        analysis_str = f"""
    DOCUMENT ANALYSIS CONTEXT (Use this for key concepts and terms):
    Key Concepts: {', '.join(context.analysis_result.key_concepts)}
    Key Terms: {', '.join(context.analysis_result.key_terms.keys())}
    --- End of Analysis Context ---
    """

    # Combine contexts
    full_context = f"""
    {lesson_plan_str}
    {analysis_str}
    """

    # --- Build the core prompt ---
    # This will be appended to the context

    # Modify prompt based on whether a specific topic is requested
    if topic_to_explain:
        prompt_core = f"Focus entirely on explaining the specific topic: '{topic_to_explain}'. Use the Lesson Plan context for objectives and structure, and the Document Analysis context for key terms and concepts related to this topic. Synthesize a clear and concise explanation for this single topic into the 'text' field using file_search if necessary for details. Title should reflect the topic."
    else:
        # Original behavior (though less likely to be used by orchestrator)
        prompt_core = "Based on the full Lesson Plan and Document Analysis context provided above, generate the complete lesson content, synthesizing all relevant information into the 'text' field. Use the Lesson Plan title for the 'title' field."

    final_prompt = f"{full_context}\n\nTASK:\n{prompt_core}"
    
    # Setup RunConfig for tracing
    from agents import RunConfig
    
    run_config = None
    if context:
        run_config = RunConfig(
            workflow_name="AI Tutor - Content Creation",
            group_id=context.session_id
        )
    
    # Run the teacher agent with the lesson plan to get the LessonContent
    result = await Runner.run(
        teacher_agent,
        final_prompt, # Use the constructed prompt
        run_config=run_config,
        context=context
    )
    
    # Get the lesson content
    try:
        lesson_content = result.final_output_as(LessonContent)
        return lesson_content
    except Exception as e:
        print(f"Error getting lesson content: {e}")
        raise 