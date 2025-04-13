from __future__ import annotations
from typing import Optional, List, Dict, Any
from uuid import UUID
from supabase import Client
from google.adk import Agent
from google.adk.tools import BaseTool, FunctionTool, ToolContext
from google.adk.runners import Runner, RunConfig
from pydantic import BaseModel, Field

from ai_tutor.agents.models import FocusObjective
from ai_tutor.context import TutorContext
import os
import logging

# --- Get Supabase client dependency (needed for the tool) ---
from ai_tutor.dependencies import get_supabase_client

# --- Define read_knowledge_base tool locally ---
@FunctionTool
async def read_knowledge_base(tool_context: ToolContext) -> str:
    """Reads the Knowledge Base content stored in the Supabase 'folders' table associated with the current session's folder_id."""
    folder_id = tool_context.context.folder_id
    user_id = tool_context.context.user_id
    print(f"Tool: read_knowledge_base called. Folder ID from context: {folder_id}")

    if not folder_id:
        return "Error: Folder ID not found in context."

    # --- ADD CHECK HERE ---
    # Check if analysis result with text is already in context from SessionManager loading
    if tool_context.context.analysis_result and tool_context.context.analysis_result.analysis_text:
        print(f"Tool: read_knowledge_base - Found analysis text in context. Returning cached text.")
        return tool_context.context.analysis_result.analysis_text

    try:
        supabase = await get_supabase_client()
        response = supabase.table("folders").select("knowledge_base").eq("id", str(folder_id)).eq("user_id", user_id).maybe_single().execute()

        if response.data and response.data.get("knowledge_base"):
            kb_content = response.data["knowledge_base"]
            print(f"Tool: read_knowledge_base successful from Supabase. Content length: {len(kb_content)}")
            # Store it back into context in case it wasn't there (though SessionManager should handle this on load)
            if not tool_context.context.analysis_result:
                 # Assuming AnalysisResult model exists and can be instantiated like this
                 from ai_tutor.agents.analyzer_agent import AnalysisResult
                 tool_context.context.analysis_result = AnalysisResult(analysis_text=kb_content, vector_store_id=tool_context.context.vector_store_id or "", key_concepts=[], key_terms={}, file_names=[])
            elif not tool_context.context.analysis_result.analysis_text:
                 tool_context.context.analysis_result.analysis_text = kb_content
            return kb_content
        else:
            return f"Error: Knowledge Base not found for folder {folder_id} or query failed: {response.error}"
    except Exception as e:
        error_msg = f"Error reading Knowledge Base from Supabase for folder {folder_id}: {e}"
        print(f"Tool: {error_msg}")
        return error_msg
# -----------------------------------------------

class FocusObjective(BaseModel):
    """Represents the next learning focus objective."""
    topic: str
    description: str
    prerequisites: List[str] = []
    learning_goals: List[str] = []
    estimated_time_minutes: Optional[int] = None
    difficulty_level: Optional[str] = None

def create_planner_agent() -> Agent:
    """Creates a planner agent that determines the next learning focus using ADK."""

    # Import the common tools (now only user model status)
    from ai_tutor.tools.orchestrator_tools import (
        get_user_model_status_tool  # Import the instance
    )

    planner_tools = [
        get_user_model_status_tool   # Use the tool instance
    ]

    # Use Gemini model via ADK
    model_identifier = "gemini-2.0-flash"

    # Create the planner agent focusing on identifying the next focus
    planner_agent = Agent(
        name="focus_planner",
        instruction="""You are an expert learning strategist. Your task is to determine the user's **next learning focus** based on the provided documents (via Gemini File API) and their current progress.

        WORKFLOW:
        1. **Get Current State:**
           - Call get_user_model_status tool to understand user's progress
           - Review mastery levels, confusion points, and learning pace
           - Note any mastered objectives or ongoing challenges

        2. **Analyze Content:**
           - Analyze the content of the document(s) provided directly in the prompt (via File API).
           - Extract key concepts, structure, and potential prerequisites.

        3. **Plan Next Focus:**
           - Consider prerequisites identified from the document content.
           - Consider user's current mastery levels from the tool call.
           - Consider confusion points and learning pace.
           - Identify the most appropriate next topic based on the document structure and user state.

        4. **Output Focus:**
           - Generate a FocusObjective JSON with ALL required fields:
             * topic: The specific topic to focus on (must be present in the document)
             * learning_goal: Clear, achievable goal related to the document content
             * priority: Importance (1-5)
             * relevant_concepts: Related concepts from the document
             * suggested_approach: Teaching hints based on document content, or "None" if no specific hints.

        IMPORTANT:
        - Base your analysis and planning *solely* on the provided document content and the user model status.
        - Always check user's current state before planning.
        - Respect prerequisites and learning progression suggested by the document structure.
        - Consider user's demonstrated pace and abilities.
        - Return ONLY the FocusObjective JSON containing ALL fields (topic, learning_goal, priority, relevant_concepts, suggested_approach).
        """,
        model=model_identifier,
        tools=planner_tools, # Only includes get_user_model_status_tool now
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )
    
    return planner_agent

async def plan_next_focus(context=None, supabase: Client = None) -> Optional[FocusObjective]:
    """
    Determine the next learning focus based on the knowledge base and user's current state.
    
    Args:
        context: Context object with session_id and user state
        supabase: Optional Supabase client instance
        
    Returns:
        A FocusObjective object containing the next learning focus, or None on failure.
    """
    if not context or not hasattr(context, 'session_id'):
        logger.error("plan_next_focus: No valid context provided")
        return None

    # Create the planner agent
    planner_agent = create_planner_agent()
    
    # Setup RunConfig for tracing
    run_config = RunConfig(
        workflow_name="AI Tutor - Learning Planning",
        group_id=str(context.session_id)
    )
    
    # Create prompt for the agent
    user_state = getattr(context, 'user_state', None)
    user_state_str = f"\nCurrent user state: {user_state}" if user_state else ""
    
    prompt = f"""
    Please determine the next most important topic or concept for the user to learn.
    
    First, use the `read_knowledge_base` tool to get the current document analysis.
    Then, analyze the knowledge base and determine the most appropriate next focus topic.
    {user_state_str}
    
    Consider:
    1. Topic dependencies and prerequisites
    2. Natural learning progression
    3. User's current state and progress
    4. Complexity and estimated time requirements
    
    Return a structured FocusObjective with your recommendation.
    """
    
    try:
        result = await Runner.run(
            planner_agent,
            prompt,
            run_config=run_config
        )
        
        if not result or not result.output:
            logger.error("plan_next_focus: No output from planner")
            return None
            
        # The output should already be a FocusObjective thanks to output_schema
        return result.output
        
    except Exception as e:
        logger.error(f"plan_next_focus: Error during planning: {str(e)}")
        return None 