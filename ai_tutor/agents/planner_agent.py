from __future__ import annotations
from typing import Optional, List, Dict, Any
from uuid import UUID
from supabase import Client
from agents import Agent, FileSearchTool, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import FocusObjective, PlannerOutput
from ai_tutor.agents.utils import RoundingModelWrapper
from ai_tutor.context import TutorContext
import os

# --- Get Supabase client dependency (needed for the tool) ---
from ai_tutor.dependencies import get_supabase_client

from functools import lru_cache, wraps
import asyncio
from ai_tutor.utils.decorators import function_tool_logged

# Global cache for concept graph edges and last updated timestamp
_dag_cache = {
    "edges": None,
    "updated_at": None
}
_dag_cache_lock = asyncio.Lock()

async def _get_concept_graph_edges(supabase):
    """Fetch edges and updated_at from Supabase, with cache."""
    async with _dag_cache_lock:
        # Check updated_at timestamp
        updated_resp = supabase.table("concept_graph").select("updated_at").order("updated_at", desc=True).limit(1).execute()
        updated_at = None
        if updated_resp.data:
            updated_at = updated_resp.data[0].get("updated_at")
        if _dag_cache["edges"] is not None and _dag_cache["updated_at"] == updated_at:
            return _dag_cache["edges"]
        # Fetch all edges
        response = supabase.table("concept_graph").select("prereq, concept, updated_at").execute()
        edges = response.data or []
        _dag_cache["edges"] = edges
        _dag_cache["updated_at"] = updated_at
        return edges

# --- Define read_knowledge_base tool locally ---
@function_tool_logged()
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
            # Knowledge Base missing or query failed
            return f"Error: Knowledge Base not found for folder {folder_id}."
    except Exception as e:
        error_msg = f"Error reading Knowledge Base from Supabase for folder {folder_id}: {e}"
        print(f"Tool: {error_msg}")
        return error_msg

@function_tool_logged()
async def dag_query(ctx: RunContextWrapper[TutorContext], mastered: list[str]) -> list[str]:
    """Returns next learnable concepts based on the concept_graph table and user's mastered concepts."""
    supabase = await get_supabase_client()
    # Fetch all prerequisite relationships between concepts, using cache
    edges = await _get_concept_graph_edges(supabase)
    # Build prerequisite map: concept -> list of prereqs
    prereq_map: Dict[str, List[str]] = {}
    for e in edges:
        prereq_map.setdefault(e["concept"], []).append(e["prereq"])
    # Identify next learnable concepts: not yet mastered and all prereqs satisfied
    candidates = [c for c, prereqs in prereq_map.items() if c not in mastered and all(p in mastered for p in prereqs)]
    return candidates

def create_planner_agent(vector_store_id: str) -> Agent[TutorContext]:
    """Creates a planner agent that can search through files and create a lesson plan."""
    
    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=5,
        include_search_results=True,
    )

    print(f"Created FileSearchTool for vector store: {vector_store_id}")

    # Only include tools the planner should use (avoid file_search by default)
    planner_tools = [read_knowledge_base, dag_query]

    # Instantiate the base model provider and get the base model
    provider: OpenAIProvider = OpenAIProvider()
    base_model = provider.get_model("gpt-4o")

    # Create the planner agent specifying context type generically and output_type via parameter
    planner_agent = Agent[TutorContext](
        name="Focus Planner",
        instructions="""You are an expert learning strategist. Your task is to determine the user's **next learning focus** based on the analyzed documents and their current progress (provided in the prompt context).

        AVAILABLE INFORMATION:
        - You have a `read_knowledge_base` tool to retrieve the document analysis summary stored in the database.
        - You have a `file_search` tool to look up specific details within the source documents (vector store).
        - You have a `dag_query` tool that accepts a list of mastered concepts and returns the next learnable concepts from the concept graph.
        - The prompt may contain information about the user's current state (`UserModelState` summary), including a list of mastered concepts.

        YOUR WORKFLOW **MUST** BE:
        1.  **Read Knowledge Base ONCE:** Call the `read_knowledge_base` tool *exactly one time* at the beginning to obtain the document analysis summary (key concepts, terms, etc.).
        2.  **Obtain Candidate Concepts:** Call the `dag_query` tool with the list of mastered concepts from the user model state to retrieve next learnable concepts based on prerequisites.
        3.  **Analyze KB and Candidates:** Once you have the Knowledge Base summary and candidate concepts, **DO NOT** call `read_knowledge_base` again. Analyze the KB, candidate list, and any provided user state summary.
        4.  **Identify Next Focus:** Determine the single most important topic or concept the user should learn next, selecting from candidate concepts and considering prerequisites and user progress.
        5.  **Define Learning Goal:** Formulate a clear, specific learning goal for this focus topic.
        6.  **Use `file_search` Sparingly:** If needed to clarify the goal or identify crucial related concepts for the chosen focus topic, use `file_search`.

        OUTPUT:
        - Your output **MUST** be a single, valid JSON object matching the `PlannerOutput` schema. Do NOT add any other text before or after the JSON object.
        - The `PlannerOutput` object MUST contain:
            * `objective`: The FocusObjective (see schema).
            * `next_action`: An ActionSpec object specifying the next agent, params, success_criteria, and max_steps.
        
        EXAMPLE OUTPUT (JSONC):
        {
          "objective": {
            "topic": "Limits",
            "learning_goal": "Understand the concept of limits in calculus.",
            "priority": 5,
            "relevant_concepts": ["limit definition", "epsilon-delta"],
            "suggested_approach": "Needs examples",
            "target_mastery": 0.8,
            "initial_difficulty": "medium"
          },
          "next_action": {
            "agent": "teacher",
            "params": {"difficulty": "medium", "section_title": "Limits"},
            "success_criteria": "The student can explain the definition of a limit and solve a basic limit problem.",
            "max_steps": 3
          }
        }

        CRITICAL REMINDERS:
        - **You MUST call `read_knowledge_base` only ONCE at the very start.**
        - Your only output MUST be a single `PlannerOutput` object. Do NOT create a full `LessonPlan`.
        """,
        tools=planner_tools,
        model=base_model,
    )
    
    return planner_agent 