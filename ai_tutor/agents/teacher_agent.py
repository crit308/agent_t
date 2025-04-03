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


def create_teacher_agent(api_key: str = None) -> Agent['TutorContext']:
    """Creates the Teacher Agent for the AI Tutor."""

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    provider: ModelProvider = OpenAIProvider()
    base_model = provider.get_model("gpt-4o-2024-08-06")  # Using exact model version

    # Create the teacher agent
    teacher_agent = Agent(
        name="Lesson Teacher",
        instructions="""
        You are an expert educational content creator. Your task is to create lesson content for a *specific topic segment* based on the provided lesson plan context and document analysis context.

        GUIDELINES:
        1. You will be given a specific `topic` and possibly a `segment_index` to explain.
        2. Use the provided `Lesson Plan Context` and `Document Analysis Context` for background information and structure.
        3. Use the `file_search` tool to find specific details or examples related *only* to the assigned `topic` and `segment_index` if the provided context is insufficient.
        4. Synthesize the information into a clear, concise, and engaging explanation for the assigned segment. Structure the text logically. Use Markdown for formatting if helpful (headings, lists).
        5. Aim for segments that are digestible, perhaps 1-3 paragraphs long, unless the topic is inherently complex.
        6. If `segment_index` is 0, provide an introduction to the `topic`. If it's a later index, assume prior segments have been covered.

        CRITICAL OUTPUT FORMAT:
        - Your output MUST be a valid `LessonContent` JSON object with three fields: "title" (string), "topic" (string), and "text" (string).
        - The "title" should be the *overall lesson title* (from the Lesson Plan context).
        - The "topic" field MUST contain the specific topic you were asked to explain.
        - The "text" field should contain the explanation *only for the assigned topic segment*.

        YOUR OUTPUT MUST BE ONLY THE VALID LessonContent JSON OBJECT { "title": "...", "text": "..." }.
        """,
        tools=[file_search_tool],
        # No handoffs in this version
        output_type=LessonContent,
        # Use a stronger model for better segmented content generation
        model=RoundingModelWrapper(provider.get_model("gpt-4o-mini")),
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
    teacher_agent: Agent['TutorContext'],
    lesson_plan: LessonPlan,
    topic_to_explain: Optional[str] = None,  # Added parameter
    context: Optional['TutorContext'] = None  # Use forward reference string
) -> LessonContent:
    """Generate lesson content using the teacher agent."""

    # Validate context
    if not context:
         raise ValueError("TutorContext is required for generating lesson content.")

    # --- Build the prompt ---
    lesson_plan_context_str = f"""
    LESSON PLAN CONTEXT (Use this for structure and objectives):

    Title: {lesson_plan.title}
    Description: {lesson_plan.description}
    Target Audience: {lesson_plan.target_audience}
    Prerequisites: {', '.join(lesson_plan.prerequisites)}
    # Total Estimated Duration: {lesson_plan.total_estimated_duration_minutes} minutes # Comment out - less relevant for single topic
    Sections Overview:"""
    for i, section in enumerate(lesson_plan.sections):
        lesson_plan_str += f"\nSection {i+1}: {section.title}\n"
        lesson_plan_str += f"  Objectives: {', '.join(obj.title for obj in section.objectives)}\n"
        lesson_plan_str += f"  Duration: {section.estimated_duration_minutes} minutes\n"
        if section.concepts_to_cover:
            lesson_plan_str += f"  Key Concepts: {', '.join(section.concepts_to_cover)}\n"
    lesson_plan_context_str += "\n--- End of Lesson Plan Context ---\n"

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
    {lesson_plan_context_str}
    {analysis_str}
    """

    # --- Build the core prompt ---
    # This will be appended to the context

    # Always require a topic now for this function
    if not topic_to_explain:
        raise ValueError("`topic_to_explain` must be provided to generate focused content.")

    prompt_core = f"Your task is to explain the specific topic: '{topic_to_explain}'. Use the provided Lesson Plan context for overall objectives and structure, and the Document Analysis context for relevant key terms and concepts. Use file_search only if necessary to find specific details or examples for '{topic_to_explain}'. Synthesize a clear and concise explanation for this topic segment into the 'text' field. Ensure the 'title' field contains the overall lesson title ('{lesson_plan.title}') and the 'topic' field contains '{topic_to_explain}'."

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