from __future__ import annotations
from typing import Optional, List, Dict, Any
from uuid import UUID
from supabase import Client
from google.adk.agents import LlmAgent
from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.retrieval import FilesRetrieval
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import FocusObjective
from ai_tutor.agents.utils import RoundingModelWrapper
from ai_tutor.context import TutorContext
import os

# --- Get Supabase client dependency (needed for the tool) ---
from ai_tutor.dependencies import get_supabase_client

# --- Define read_knowledge_base tool locally ---
@FunctionTool
async def read_knowledge_base(ctx: RunContextWrapper[TutorContext]) -> str:
    """Reads the Knowledge Base content stored in the Supabase 'folders' table associated with the current session's folder_id."""
    folder_id = ctx.context.folder_id
    user_id = ctx.context.user_id
    print(f"Tool: read_knowledge_base called. Folder ID from context: {folder_id}")

    if not folder_id:
        return "Error: Folder ID not found in context."

    # --- ADD CHECK HERE ---
    # Check if analysis result with text is already in context from SessionManager loading
    if ctx.context.analysis_result and ctx.context.analysis_result.analysis_text:
        print(f"Tool: read_knowledge_base - Found analysis text in context. Returning cached text.")
        return ctx.context.analysis_result.analysis_text

    try:
        supabase = await get_supabase_client()
        response = supabase.table("folders").select("knowledge_base").eq("id", str(folder_id)).eq("user_id", user_id).maybe_single().execute()

        if response.data and response.data.get("knowledge_base"):
            kb_content = response.data["knowledge_base"]
            print(f"Tool: read_knowledge_base successful from Supabase. Content length: {len(kb_content)}")
            # Store it back into context in case it wasn't there (though SessionManager should handle this on load)
            if not ctx.context.analysis_result:
                 # Assuming AnalysisResult model exists and can be instantiated like this
                 from ai_tutor.agents.analyzer_agent import AnalysisResult
                 ctx.context.analysis_result = AnalysisResult(analysis_text=kb_content, vector_store_id=ctx.context.vector_store_id or "", key_concepts=[], key_terms={}, file_names=[])
            elif not ctx.context.analysis_result.analysis_text:
                 ctx.context.analysis_result.analysis_text = kb_content
            return kb_content
        else:
            return f"Error: Knowledge Base not found for folder {folder_id} or query failed: {response.error}"
    except Exception as e:
        error_msg = f"Error reading Knowledge Base from Supabase for folder {folder_id}: {e}"
        print(f"Tool: {error_msg}")
        return error_msg
# -----------------------------------------------

def create_planner_agent(vector_store_id: str) -> Agent[TutorContext]:
    """Creates a planner agent that determines the next learning focus."""
    
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
    base_model = provider.get_model("gpt-4o")

    # Create the planner agent specifying context type generically and output_type via parameter
    planner_agent = LlmAgent(
        name="FocusPlanner",
        instructions="""You are an expert learning strategist. Your task is to determine the user's **next learning focus** based on the analyzed documents and potentially their current progress (provided in the prompt context).

        AVAILABLE INFORMATION:
        - You have a `read_knowledge_base` tool to get the document analysis summary stored in the database.
        - You have a `file_search` tool to look up specific details within the source documents (vector store).
        - The prompt may contain information about the user's current state (`UserModelState` summary).

        YOUR WORKFLOW **MUST** BE:
        1.  **Read Knowledge Base ONCE:** Call the `read_knowledge_base` tool *exactly one time* at the beginning to get the document analysis summary (key concepts, terms, etc.).
        2.  **Confirm KB Received & Analyze Summary:** Once you have the Knowledge Base summary from the tool, **DO NOT call `read_knowledge_base` again**. Analyze the KB and any provided user state summary.
        3.  **Identify Next Focus:** Determine the single most important topic or concept the user should learn next. Consider prerequisites implied by the KB structure and the user's current state (e.g., last completed topic, identified struggles).
        4.  **Define Learning Goal:** Formulate a clear, specific learning goal for this focus topic.
        5.  **Use `file_search` Sparingly:** If needed to clarify the goal or identify crucial related concepts for the chosen focus topic, use `file_search`.

        OUTPUT:
        - Your output **MUST** be a single, valid JSON object matching the `FocusObjective` schema.
        - The `FocusObjective` object MUST contain:
            * `topic`: The main topic to focus on.
            * `learning_goal`: The specific objective for this topic.
            * `priority`: An estimated priority (1-5).
            * `relevant_concepts`: Key concepts from the KB related to this topic.
            * `suggested_approach`: (Optional) A hint for the Orchestrator.

        CRITICAL REMINDERS:
        - **You MUST call `read_knowledge_base` only ONCE at the very start.**
        - Your only output MUST be a single `FocusObjective` object.
        """,
        tools=planner_tools,
        output_schema=FocusObjective,
        model=base_model,
    )
    
    return planner_agent 