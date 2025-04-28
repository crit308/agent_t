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
from ai_tutor.utils.tool_helpers import invoke # Import the invoke helper

from ai_tutor.core.schema import PlannerOutput

import logging
import traceback
import json
from ai_tutor.agents.models import FocusObjective

# Global cache for concept graph edges and last updated timestamp
_dag_cache = {
    "edges": None,
    "updated_at": None
}
_dag_cache_lock = asyncio.Lock()

async def _get_concept_graph_edges(supabase):
    """Fetch concept prerequisite edges from Supabase without relying on updated_at column."""
    response = supabase.table("concept_graph").select("prereq, concept").execute()
    if response and hasattr(response, 'data') and response.data:
        return response.data
    return []

logger = logging.getLogger(__name__)

# --- Define read_knowledge_base tool locally ---
@skill(cost="low") # This now creates a FunctionTool object named read_knowledge_base
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

@skill(cost="low") # This now creates a FunctionTool object named dag_query
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

async def determine_session_focus(ctx: TutorContext) -> FocusObjective:
    """Analyzes the KB and user state to determine the primary FocusObjective for the session."""
    logger.info(f"determine_session_focus started for session {ctx.session_id}")
    kb_text: Optional[str] = None
    next_concepts: Optional[List[str]] = None
    mastered: List[str] = []
    user_model_state_summary: str = "No user model state available."

    try:
        # 1. Retrieve knowledge base using invoke
        logger.info(f"determine_session_focus: Calling read_knowledge_base via invoke for session {ctx.session_id}")
        try:
            # --- FIX: Use invoke --- 
            # Pass the FunctionTool object and context
            kb_text = await invoke(read_knowledge_base, ctx=ctx)
            # --- END FIX ---
            logger.info(f"determine_session_focus: read_knowledge_base returned. Length: {len(kb_text) if kb_text else 'None'}")
            if kb_text and "Error:" in kb_text: # Check for errors returned by the tool
                 logger.error(f"determine_session_focus: Error from read_knowledge_base: {kb_text}")
                 raise ValueError(f"Failed to read knowledge base: {kb_text}")
        except Exception as tool_e:
            logger.error(f"determine_session_focus: Exception calling read_knowledge_base for session {ctx.session_id}: {tool_e}\n{traceback.format_exc()}", exc_info=True)
            raise # Re-raise to outer block

        # 2. Determine mastered concepts and summarize user model state
        try:
            if ctx.user_model_state and ctx.user_model_state.concepts:
                 mastered = [t for t, s in ctx.user_model_state.concepts.items() if s.mastery > 0.8 and s.confidence >= 5] # Assuming 0.8 mastery and 5 confidence means mastered
                 # Create a summary string for the prompt
                 state_items = []
                 for topic, state in ctx.user_model_state.concepts.items():
                     state_items.append(f"- {topic}: Mastery={state.mastery:.2f}, Confidence={state.confidence}, Attempts={state.attempts}")
                 if state_items:
                     user_model_state_summary = "Current user concept understanding:\n" + "\n".join(state_items)
                 else:
                     user_model_state_summary = "User has no tracked concepts yet."

            logger.info(f"determine_session_focus: Determined mastered concepts for session {ctx.session_id}: {mastered}")
            logger.info(f"determine_session_focus: User model state summary for prompt: {user_model_state_summary}")
        except Exception as mastery_e:
             logger.error(f"determine_session_focus: Error processing user model state for session {ctx.session_id}: {mastery_e}", exc_info=True)
             # Continue, maybe with empty mastered list and default summary

        # 3. Query DAG for next learnable concepts (using invoke)
        logger.info(f"determine_session_focus: Calling dag_query via invoke for session {ctx.session_id}")
        try:
            # --- FIX: Use invoke --- 
            # Pass the FunctionTool object, context, and mastered list as kwargs
            next_concepts = await invoke(dag_query, ctx=ctx, mastered=mastered)
            # --- END FIX ---
            logger.info(f"determine_session_focus: dag_query returned: {next_concepts}")
        except Exception as tool_e:
            logger.error(f"determine_session_focus: Exception calling dag_query for session {ctx.session_id}: {tool_e}\n{traceback.format_exc()}", exc_info=True)
            # Don't raise, planner can still function without DAG info
            next_concepts = None # Indicate that DAG info is unavailable

        # 4. Call LLM
        llm = LLMClient()
        system_msg = {
            "role": "system",
            "content": (
                "You are the Focus Planner agent. Your task is to analyze the provided Knowledge Base text (potentially truncated to show recent history), the user's current concept understanding (User Model State), and potential next concepts based on prerequisites (if available). "
                "Based on this analysis, select the single most important FocusObjective for the current tutoring session. Consider the importance of topics, prerequisites, and the user's progress. "
                "Output ONLY a single, valid JSON object conforming exactly to the FocusObjective Pydantic model schema. Ensure 'topic', 'learning_goal', 'priority' (integer 1-5), and 'target_mastery' fields are ALWAYS included. Do not add any commentary before or after the JSON object."
                "\n\nFocusObjective Schema:\n"
                "{\n"
                "  'topic': str,              // The primary topic or concept to focus on.\n"
                "  'learning_goal': str,      // A specific, measurable goal (e.g., 'Understand local vs global scope').\n"
                "  'priority': int,           // Priority 1-5 (5=highest). MANDATORY FIELD.\n"
                "  'relevant_concepts': List[str], // Optional list of related concepts from the KB.\n"
                "  'suggested_approach': Optional[str], // Optional hint (e.g., 'Needs examples').\n"
                "  'target_mastery': float,   // Target mastery level (e.g., 0.8). MANDATORY FIELD.\n"
                "  'initial_difficulty': Optional[str] // Optional initial difficulty (e.g., 'Medium').\n"
                "}"
            )
        }

        # --- KB Truncation Logic (Task E-2) --- #
        KB_INPUT_LIMIT_BYTES = 8000 # Approx 8kB ~ 2k tokens
        kb_prompt_content: str
        if kb_text and len(kb_text.encode('utf-8')) > KB_INPUT_LIMIT_BYTES:
             # Get the last N bytes (tail)
             kb_tail_bytes = kb_text.encode('utf-8')[-KB_INPUT_LIMIT_BYTES:]
             # Decode carefully, ignoring errors if split mid-character
             kb_snippet = kb_tail_bytes.decode('utf-8', errors='ignore')
             kb_prompt_content = f"... (Beginning of Knowledge Base truncated)\n\n{kb_snippet}"
             logger.warning(f"Knowledge base text for session {ctx.session_id} truncated to last ~{KB_INPUT_LIMIT_BYTES} bytes for planner prompt.")
        elif kb_text:
             kb_prompt_content = kb_text
        else:
             kb_prompt_content = "Knowledge Base is empty or unavailable."
        # --- End KB Truncation Logic ---

        dag_info = f"Suggested next learnable concepts based on prerequisites: {next_concepts}" if next_concepts is not None else "Prerequisite information (DAG) is not available for planning."

        messages = [
            system_msg,
            {"role": "user", "content": f"Knowledge Base Content (Recent History):\n{kb_prompt_content}"},
            {"role": "user", "content": user_model_state_summary},
            {"role": "user", "content": dag_info},
            {"role": "user", "content": "Select the single best FocusObjective for this session based on all available information. Respond only with the JSON object."}
        ]
        logger.info(f"determine_session_focus: Calling LLM for session {ctx.session_id}")
        try:
            response_text = await llm.chat(messages) # Removed response_format
            logger.info(f"determine_session_focus: LLM call completed for session {ctx.session_id}")
        except Exception as llm_e:
             logger.error(f"determine_session_focus: LLM call failed for session {ctx.session_id}: {llm_e}", exc_info=True)
             raise ValueError("LLM call failed during focus planning.") from llm_e

        # 5. Parse and Validate LLM Response
        try:
            logger.info(f"determine_session_focus: Attempting to parse LLM response JSON for {ctx.session_id}")
            # LLMClient with json_object mode should return parsed dict directly or raise error
            if isinstance(response_text, str): # Fallback if json_object mode didn't work as expected
                 # Clean up model response: strip markdown fences and extract JSON
                 import re
                 cleaned = response_text.strip()
                 if cleaned.startswith("```"):
                     cleaned = re.sub(r"^```[^\\n]*\\n", "", cleaned)
                 if cleaned.endswith("```"):
                     cleaned = re.sub(r"\\n```$", "", cleaned)
                 start = cleaned.find('{')
                 end = cleaned.rfind('}')
                 if start != -1 and end != -1:
                     json_str = cleaned[start:end+1]
                 else:
                    logger.error(f"determine_session_focus: Could not extract JSON object from LLM string response: {response_text}")
                    raise ValueError("LLM did not return a valid JSON object string.")
                 data = json.loads(json_str)
            elif isinstance(response_text, dict):
                data = response_text # Assume it's already parsed JSON
            else:
                 logger.error(f"determine_session_focus: Unexpected LLM response type: {type(response_text)}")
                 raise ValueError("Unexpected LLM response type during focus planning.")

            logger.info(f"determine_session_focus: JSON parsed/obtained successfully for {ctx.session_id}: {data}")

            # Validate and create FocusObjective model
            logger.info(f"determine_session_focus: Attempting to validate FocusObjective for {ctx.session_id}. Data: {data}")
            # Ensure target_mastery is present and a float before validation if needed
            if 'target_mastery' not in data or not isinstance(data.get('target_mastery'), (float, int)):
                 logger.warning(f"determine_session_focus: 'target_mastery' missing or not a number in LLM response for {ctx.session_id}. Setting default 0.8. Data: {data}")
                 data['target_mastery'] = 0.8 # Add default if missing, as it's required by spec

            # Ensure priority is present and an int before validation if needed
            if 'priority' not in data or not isinstance(data.get('priority'), int):
                 logger.warning(f"determine_session_focus: 'priority' missing or not an int in LLM response for {ctx.session_id}. Setting default 3. Data: {data}")
                 data['priority'] = 3 # Add default if missing

            focus_objective = FocusObjective.model_validate(data)
            logger.info(f"determine_session_focus: FocusObjective validated successfully for {ctx.session_id}")

        except json.JSONDecodeError as json_e:
            logger.error(f"determine_session_focus: Failed to parse LLM JSON response for session {ctx.session_id}: {json_e}. Response: {response_text}", exc_info=True)
            raise ValueError("Failed to parse planner output from LLM") from json_e
        except Exception as val_e: # Catch Pydantic validation errors etc.
             logger.error(f"determine_session_focus: Failed to validate FocusObjective for session {ctx.session_id}: {val_e}. Data: {data}", exc_info=True)
             raise ValueError("Failed to validate planner output") from val_e

        # 6. Store chosen objective in context
        ctx.current_focus_objective = focus_objective
        logger.info(f"determine_session_focus: Stored focus objective '{ctx.current_focus_objective.topic}' in context for session {ctx.session_id}")

        logger.info(f"determine_session_focus finished successfully for session {ctx.session_id}")
        return focus_objective

    except Exception as outer_e:
         # Catch any exception missed by inner blocks
         logger.critical(f"Unhandled exception within determine_session_focus for session {ctx.session_id}: {type(outer_e).__name__}: {outer_e}\n{traceback.format_exc()}", exc_info=True)
         # Consider returning a default/fallback FocusObjective or re-raise
         raise 