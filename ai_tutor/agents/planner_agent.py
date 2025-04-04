from __future__ import annotations
from typing import Optional, List, Dict, Any
from agents import Agent, FileSearchTool, ModelProvider, function_tool
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import LearningObjective, LessonSection, LessonPlan, QuizQuestion
from ai_tutor.agents.utils import RoundingModelWrapper
from ai_tutor.context import TutorContext
import os

# --- Define read_knowledge_base tool locally ---
@function_tool
def read_knowledge_base(ctx: RunContextWrapper[TutorContext]) -> str:
    """Reads the Knowledge Base file path stored in context and returns its content."""
    kb_path = ctx.context.knowledge_base_path
    print(f"Tool: read_knowledge_base called. Path from context: {kb_path}")
    if not kb_path or not os.path.exists(kb_path):
        return "Error: Knowledge Base file path not found in context or file does not exist."
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(f"Tool: read_knowledge_base successful. Content length: {len(content)}")
            return content
    except Exception as e:
        error_msg = f"Error reading Knowledge Base file at {kb_path}: {e}"
        print(f"Tool: {error_msg}")
        return error_msg
# -----------------------------------------------

def create_planner_agent(vector_store_id: str) -> Agent[TutorContext]:
    """Creates a planner agent that can search through files and create a lesson plan."""
    
    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=5,
        include_search_results=True,
    )

    print(f"Created FileSearchTool for vector store: {vector_store_id}")

    # Include the read_knowledge_base tool
    planner_tools = [file_search_tool, read_knowledge_base]

    # Instantiate the base model provider and get the base model
    provider: ModelProvider = OpenAIProvider()
    base_model = provider.get_model("gpt-4o-2024-08-06")

    # Create the planner agent with access to the file search tool
    planner_agent = Agent[TutorContext](
        name="Lesson Planner",
        instructions="""You are an expert curriculum designer. Your task is to create a well-structured lesson plan based on analyzed documents.

        AVAILABLE INFORMATION:
        - You have a `read_knowledge_base` tool to get the document analysis summary.
        - You have a `file_search` tool to look up specific details within the source documents.

        YOUR WORKFLOW **MUST** BE:
        1.  **Read Knowledge Base:** Call the `read_knowledge_base` tool first to get the document analysis summary.
        2.  **Analyze Summary:** Understand the key concepts, terms, and structure from the KB content.
        3.  **Use `file_search` for Details:** If the KB summary lacks specific details needed for planning (e.g., examples, specific steps), use `file_search` to find that information in the source documents.
        4.  **Create Lesson Plan:** Synthesize information from the KB analysis and any `file_search` results to create a comprehensive `LessonPlan` object.
        - Your lesson plan must include:
          * Clear learning objectives for each section
          * Logical sequence of sections
          * Appropriate time durations for each section
          * Consideration of prerequisites
          * Target audience

        STEP 4: OUTPUT
        - Output the lesson plan as a complete structured LessonPlan object.

        CRITICAL REMINDERS:
        - **You MUST call `read_knowledge_base` first.**
        - DO NOT call any handoff tools. Your only output should be the LessonPlan object.
        """,
        tools=planner_tools,
        output_type=LessonPlan,
        model=RoundingModelWrapper(base_model),
    )
    
    return planner_agent 