from __future__ import annotations
from typing import Optional, List, Dict, Any
from uuid import UUID
from supabase import Client
from agents import Agent, FileSearchTool, ModelProvider, function_tool
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import LearningObjective, LessonSection, LessonPlan, QuizQuestion
from ai_tutor.agents.utils import RoundingModelWrapper
from ai_tutor.context import TutorContext
import os

# --- Get Supabase client dependency (needed for the tool) ---
from ai_tutor.dependencies import get_supabase_client

# --- Define read_knowledge_base tool locally ---
@function_tool
async def read_knowledge_base(ctx: RunContextWrapper[TutorContext]) -> str:
    """Reads the Knowledge Base content stored in the Supabase 'folders' table associated with the current session's folder_id."""
    folder_id = ctx.context.folder_id
    user_id = ctx.context.user_id
    print(f"Tool: read_knowledge_base called. Folder ID from context: {folder_id}")

    if not folder_id:
        return "Error: Folder ID not found in context."

    try:
        supabase = await get_supabase_client()
        response = supabase.table("folders").select("knowledge_base").eq("id", str(folder_id)).eq("user_id", user_id).maybe_single().execute()

        if response.data and response.data.get("knowledge_base"):
            kb_content = response.data["knowledge_base"]
            print(f"Tool: read_knowledge_base successful from Supabase. Content length: {len(kb_content)}")
            return kb_content
        else:
            return f"Error: Knowledge Base not found for folder {folder_id} or query failed: {response.error}"
    except Exception as e:
        error_msg = f"Error reading Knowledge Base from Supabase for folder {folder_id}: {e}"
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
    provider: OpenAIProvider = OpenAIProvider()
    base_model = provider.get_model("o3-mini")

    # Create the planner agent with access to the file search tool
    planner_agent = Agent[TutorContext](
        name="Lesson Planner",
        instructions="""You are an expert curriculum designer. Your task is to create a well-structured lesson plan based on analyzed documents.

        AVAILABLE INFORMATION:
        - You have a `read_knowledge_base` tool to get the document analysis summary stored in the database.
        - You have a `file_search` tool to look up specific details within the source documents (vector store).

        YOUR WORKFLOW **MUST** BE:
        1.  **Read Knowledge Base:** Call the `read_knowledge_base` tool first to get the document analysis summary.
        2.  **Analyze Summary:** Understand the key concepts, terms, and structure from the KB content.
        3.  **Use `file_search` for Details:** If the KB summary lacks specific details needed for planning (e.g., examples, specific steps), use `file_search` to find that information in the source documents.
        4.  **Create Lesson Plan:** Synthesize information from the KB analysis and any `file_search` results to create a comprehensive `LessonPlan` object.
        - For each `LessonSection`, you MUST include:
          * Clear learning objectives for each section
          * Logical sequence of sections
          * Appropriate time durations for each section
          * Consideration of prerequisites
          * Target audience
          * `prerequisites`: A list of concept/section titles that must be understood *before* this section. Leave empty if none.
          * `is_optional`: A boolean indicating if the section covers core material (False) or is supplementary/advanced (True). Infer this based on the content's nature (e.g., introductory sections are rarely optional).

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