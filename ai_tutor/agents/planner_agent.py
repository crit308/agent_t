from __future__ import annotations
from typing import Optional, List, Dict, Any
from uuid import UUID
from supabase import Client
from agents.run_context import RunContextWrapper
from ai_tutor.core.llm import LLMClient

from ai_tutor.context import TutorContext
import os

# --- Get Supabase client dependency (needed for the tool) ---
from ai_tutor.dependencies import get_supabase_client

from functools import lru_cache, wraps
import asyncio
from ai_tutor.utils.decorators import function_tool_logged

from ai_tutor.core.schema import PlannerOutput

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

async def run_planner(ctx: TutorContext) -> PlannerOutput:
    """Direct planner that invokes meta-skills and uses LLMClient to decide the next focus objective."""
    # Wrap context for tool invocation
    wrapper = RunContextWrapper(ctx)
    # 1. Retrieve knowledge base
    kb_text = await read_knowledge_base(wrapper)
    # 2. Determine mastered concepts
    mastered = [t for t, s in ctx.user_model_state.concepts.items() if s.mastery > 0.8 and s.confidence >= 5]
    # 3. Query DAG for next learnable concepts
    next_concepts = await dag_query(wrapper, mastered)
    # 4. Call LLM
    llm = LLMClient()
    system_msg = {
        "role": "system",
        "content": (
            "You are the Focus Planner. You have two tools: read_knowledge_base and dag_query. "
            "Use the knowledge base content and concept relationships to pick the single most important next learning objective. "
            "Respond ONLY with a JSON object matching the PlannerOutput schema (objectives: list of {topic, learning_goal, target_mastery, priority})."
        )
    }
    messages = [
        system_msg,
        {"role": "user", "content": f"Knowledge base:\n{kb_text}"},
        {"role": "user", "content": f"Mastered concepts: {mastered}"},
        {"role": "user", "content": f"Next learnable concepts: {next_concepts}"}
    ]
    response_text = await llm.chat(messages)
    output = PlannerOutput.parse_raw(response_text)
    # Store chosen objective in context
    ctx.current_focus_objective = output.objectives[0]
    return output 