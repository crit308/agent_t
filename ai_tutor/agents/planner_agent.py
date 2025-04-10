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
import logging
from pydantic import BaseModel
from google.adk.runners import Runner, RunConfig
from google.adk.tools import ToolContext

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

class FocusObjective(BaseModel):
    """Represents the next learning focus objective."""
    topic: str
    description: str
    prerequisites: List[str] = []
    learning_goals: List[str] = []
    estimated_time_minutes: Optional[int] = None
    difficulty_level: Optional[str] = None

def create_planner_agent() -> LlmAgent:
    """Creates a planner agent that determines the next learning focus using ADK."""

    # Import the common tools
    from ai_tutor.tools.orchestrator_tools import get_document_content

    planner_tools = [
        read_knowledge_base,
        get_document_content
    ]

    # Use Gemini model via ADK
    model_name = "gemini-1.5-pro"  # Using Pro for better planning capabilities

    # Create the planner agent
    planner_agent = LlmAgent(
        name="Learning Planner",
        instructions="""
        You are an expert learning planner. Your task is to determine the next most important topic 
        or concept for the user to learn, based on the available knowledge base and user's current state.

        AVAILABLE INFORMATION:
        - You have a `read_knowledge_base` tool to get the document analysis summary stored in the database.
        - You have a `get_document_content` tool to fetch full text if needed (provide file path from KB).
        - The prompt may contain information about the user's current state (UserModelState summary).

        YOUR WORKFLOW **MUST** BE:
        1. **Read Knowledge Base ONCE:** Call the `read_knowledge_base` tool *exactly one time* at the beginning 
           to get the document analysis summary (key concepts, terms, etc.).
        2. **Confirm KB Received & Analyze Summary:** Once you have the Knowledge Base summary from the tool, 
           **DO NOT call `read_knowledge_base` again**. Analyze the KB and any provided user state summary.
        3. **Identify Next Focus:** Determine the single most important topic or concept the user should learn next. 
           Consider prerequisites implied by the KB structure and the user's current state (e.g., last completed topic, 
           identified struggles).
        4. **Define Learning Goal:** Formulate a clear, specific learning goal for this focus topic. Use 
           `get_document_content` sparingly if the KB summary lacks sufficient detail to define the goal or identify 
           prerequisites for the chosen topic.

        OUTPUT FORMAT:
        Your output **MUST** be a valid JSON object matching the FocusObjective schema with these fields:
        - topic: The specific topic or concept to focus on next
        - description: Clear explanation of the topic and why it was chosen
        - prerequisites: List of concepts that should be understood first
        - learning_goals: List of specific learning objectives for this topic
        - estimated_time_minutes: Estimated time to learn this topic (optional)
        - difficulty_level: Suggested difficulty level (optional)

        EXAMPLE OUTPUT:
        {
            "topic": "Variables and Data Types",
            "description": "Understanding how to declare variables and work with different data types...",
            "prerequisites": ["Basic programming concepts", "Command line basics"],
            "learning_goals": [
                "Learn to declare variables using proper syntax",
                "Understand the main data types available",
                "Practice type conversion and checking"
            ],
            "estimated_time_minutes": 45,
            "difficulty_level": "beginner"
        }
        """,
        tools=planner_tools,
        output_schema=FocusObjective,
        model=model_name
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