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
from ai_tutor.skills import skill

from ai_tutor.core.schema import PlannerOutput

import logging
import traceback
import json

# Global cache for concept graph edges and last updated timestamp
_dag_cache = {
    "edges": None,
    "updated_at": None
}
_dag_cache_lock = asyncio.Lock()

async def _get_concept_graph_edges(supabase):
    """Fetch concept prerequisite edges from Supabase without relying on updated_at column."""
    response = supabase.table("concept_graph").select("prereq, concept").execute()
    return response.data or []

logger = logging.getLogger(__name__)

# --- Define read_knowledge_base tool locally ---
@skill(cost="low")
async def read_knowledge_base(ctx: RunContextWrapper[TutorContext]) -> str:
    """Reads the Knowledge Base content stored in the Supabase 'folders' table associated with the current session's folder_id."""
    folder_id = ctx.context.folder_id
    user_id = ctx.context.user_id
    logger.info(f"Tool: read_knowledge_base called for folder {folder_id}")

    if not folder_id:
        logger.warning("Tool: read_knowledge_base - Folder ID not found in context.")
        return "Error: Folder ID not found in context."

    # --- ADD CHECK HERE ---
    # Check if analysis result with text is already in context from SessionManager loading
    if ctx.context.analysis_result and ctx.context.analysis_result.analysis_text:
        logger.info(f"Tool: read_knowledge_base - Found analysis text in context. Returning cached text.")
        return ctx.context.analysis_result.analysis_text

    try:
        logger.info(f"Tool: read_knowledge_base - Querying Supabase for folder {folder_id}")
        supabase = await get_supabase_client()
        response = supabase.table("folders").select("knowledge_base").eq("id", str(folder_id)).eq("user_id", user_id).maybe_single().execute()
        logger.info(f"Tool: read_knowledge_base - Supabase query completed for folder {folder_id}")
        if response.data and response.data.get("knowledge_base"):
            kb_content = response.data["knowledge_base"]
            logger.info(f"Tool: read_knowledge_base successful from Supabase. Content length: {len(kb_content)}")
            # Store it back into context in case it wasn't there (though SessionManager should handle this on load)
            if not ctx.context.analysis_result:
                 # Assuming AnalysisResult model exists and can be instantiated like this
                 from ai_tutor.agents.analyzer_agent import AnalysisResult
                 ctx.context.analysis_result = AnalysisResult(analysis_text=kb_content, vector_store_id=ctx.context.vector_store_id or "", key_concepts=[], key_terms={}, file_names=[])
            elif not ctx.context.analysis_result.analysis_text:
                 ctx.context.analysis_result.analysis_text = kb_content
            return kb_content
        else:
            logger.warning(f"Tool: read_knowledge_base - Knowledge Base not found for folder {folder_id} or query failed.")
            return f"Error: Knowledge Base not found for folder {folder_id}."
    except Exception as e:
        error_msg = f"Error reading Knowledge Base from Supabase for folder {folder_id}: {e}"
        logger.error(f"Tool: read_knowledge_base - {error_msg}\n{traceback.format_exc()}", exc_info=True)
        return error_msg

@skill(cost="low")
async def dag_query(ctx: RunContextWrapper[TutorContext], mastered: list[str]) -> list[str]:
    """Returns next learnable concepts based on the concept_graph table and user's mastered concepts."""
    logger.info(f"Tool: dag_query called. Mastered concepts: {mastered}")
    supabase = await get_supabase_client()
    # Fetch all prerequisite relationships between concepts, using cache
    edges = await _get_concept_graph_edges(supabase)
    logger.info(f"Tool: dag_query - Supabase query completed. Found {len(edges)} edges.")
    # Build prerequisite map: concept -> list of prereqs
    prereq_map: Dict[str, List[str]] = {}
    for e in edges:
        prereq_map.setdefault(e["concept"], []).append(e["prereq"])
    # Identify next learnable concepts: not yet mastered and all prereqs satisfied
    candidates = [c for c, prereqs in prereq_map.items() if c not in mastered and all(p in mastered for p in prereqs)]
    logger.info(f"Tool: dag_query - Calculated candidates: {candidates}")
    return candidates

async def run_planner(ctx: TutorContext) -> PlannerOutput:
    """Direct planner that invokes meta-skills and uses LLMClient to decide the next focus objective."""
    logger.info(f"run_planner started for session {ctx.session_id}")
    wrapper = RunContextWrapper(ctx)
    kb_text = None
    next_concepts = None
    mastered = []

    try:
        # 1. Retrieve knowledge base
        logger.info(f"run_planner: Calling read_knowledge_base for session {ctx.session_id}")
        try:
            kb_text = await read_knowledge_base(wrapper)
            logger.info(f"run_planner: read_knowledge_base returned for session {ctx.session_id}. Length: {len(kb_text) if kb_text else 'None'}")
            if kb_text and "Error:" in kb_text: # Check for errors returned by the tool
                 logger.error(f"run_planner: Error from read_knowledge_base: {kb_text}")
                 # Raise specific error to be caught by outer block
                 raise ValueError(f"Failed to read knowledge base: {kb_text}")
        except Exception as tool_e:
            logger.error(f"run_planner: Exception calling read_knowledge_base for session {ctx.session_id}: {tool_e}\n{traceback.format_exc()}", exc_info=True)
            raise # Re-raise to outer block

        # 2. Determine mastered concepts
        try:
            mastered = [t for t, s in ctx.user_model_state.concepts.items() if s.mastery > 0.8 and s.confidence >= 5]
            logger.info(f"run_planner: Determined mastered concepts for session {ctx.session_id}: {mastered}")
        except Exception as mastery_e:
             logger.error(f"run_planner: Error determining mastered concepts for session {ctx.session_id}: {mastery_e}", exc_info=True)
             # Continue, maybe with empty mastered list?

        # 3. Query DAG for next learnable concepts
        logger.info(f"run_planner: Calling dag_query for session {ctx.session_id}")
        try:
            next_concepts = await dag_query(wrapper, mastered)
            logger.info(f"run_planner: dag_query returned for session {ctx.session_id}: {next_concepts}")
        except Exception as tool_e:
            logger.error(f"run_planner: Exception calling dag_query for session {ctx.session_id}: {tool_e}\n{traceback.format_exc()}", exc_info=True)
            raise # Re-raise to outer block

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
            {"role": "user", "content": f"Knowledge base:\n{kb_text or 'Not Available'}"}, # Handle None
            {"role": "user", "content": f"Mastered concepts: {mastered}"},
            {"role": "user", "content": f"Next learnable concepts: {next_concepts or 'None Available'}"} # Handle None
        ]
        logger.info(f"run_planner: Calling LLM for session {ctx.session_id}")
        try:
            response_text = await llm.chat(messages)
            logger.info(f"run_planner: LLM call completed for session {ctx.session_id}")
        except Exception as llm_e:
             logger.error(f"run_planner: LLM call failed for session {ctx.session_id}: {llm_e}", exc_info=True)
             raise # Re-raise to outer block

        # Clean up model response: strip markdown fences and extract JSON
        import re
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[^\n]*\n", "", cleaned)
        if cleaned.endswith("```"):
            cleaned = re.sub(r"\n```$", "", cleaned)
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]
        else:
            cleaned = "{}" # Ensure valid JSON if extraction fails

        # Load cleaned JSON into dict and coerce types for validation
        try:
            logger.info(f"run_planner: Attempting to parse LLM response JSON for {ctx.session_id}")
            data = json.loads(cleaned)
            logger.info(f"run_planner: JSON parsed successfully for {ctx.session_id}: {data}")

            # Handle fallback and type coercion logic
            if not data.get("objectives"):
                logger.warning(f"run_planner: LLM response missing 'objectives'. Applying fallback for session {ctx.session_id}")
                fallback_topic = next_concepts[0] if next_concepts else "General Review"
                data["objectives"] = [
                    {
                        "topic": fallback_topic,
                        "learning_goal": f"Understand the concept of {fallback_topic}",
                        "target_mastery": 0.8,
                        "priority": 3,
                    }
                ]
            else:
                logger.info(f"run_planner: Processing {len(data.get('objectives',[]))} objectives from LLM for session {ctx.session_id}")
                for obj in data.get("objectives", []):
                    tm = obj.get("target_mastery")
                    if isinstance(tm, str):
                        try: obj["target_mastery"] = float(tm)
                        except ValueError: obj["target_mastery"] = 0.8 # Default fallback
                    pr = obj.get("priority")
                    if isinstance(pr, str):
                        try: obj["priority"] = int(pr)
                        except ValueError: obj["priority"] = 3 # Default fallback

            # Validate and create PlannerOutput model
            logger.info(f"run_planner: Attempting to validate PlannerOutput for {ctx.session_id}. Data: {data}")
            output = PlannerOutput.model_validate(data)
            logger.info(f"run_planner: PlannerOutput validated successfully for {ctx.session_id}")

        except json.JSONDecodeError as json_e:
            logger.error(f"run_planner: Failed to parse LLM JSON response for session {ctx.session_id}: {json_e}. Response: {cleaned}", exc_info=True)
            raise ValueError("Failed to parse planner output from LLM") from json_e
        except Exception as val_e: # Catch Pydantic validation errors etc.
             logger.error(f"run_planner: Failed to validate PlannerOutput for session {ctx.session_id}: {val_e}. Data: {data}", exc_info=True)
             raise ValueError("Failed to validate planner output") from val_e

        # Store chosen objective in context
        if output.objectives:
             ctx.current_focus_objective = output.objectives[0]
             logger.info(f"run_planner: Stored focus objective '{ctx.current_focus_objective.topic}' in context for session {ctx.session_id}")
        else:
             logger.warning(f"run_planner: Planner output had no objectives for session {ctx.session_id}")

        logger.info(f"run_planner finished successfully for session {ctx.session_id}")
        return output

    except Exception as outer_e:
         # Catch any exception missed by inner blocks
         logger.critical(f"Unhandled exception within run_planner for session {ctx.session_id}: {type(outer_e).__name__}: {outer_e}\n{traceback.format_exc()}", exc_info=True)
         # Re-raise to be caught by the endpoint handler
         raise 