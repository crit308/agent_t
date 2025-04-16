from __future__ import annotations
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union, AsyncGenerator
from uuid import UUID
from pydantic import BaseModel, Field
from supabase import Client
import logging

from google.adk import Agent
from google.adk.runners import Runner, RunConfig
from google.adk.tools import BaseTool, FunctionTool, ToolContext, LongRunningFunctionTool
from google.generativeai import types
from google.adk.models import Gemini

from ai_tutor.agents.planner_agent import FocusObjective
from ai_tutor.context import TutorContext
from ai_tutor.agents.models import LessonContent # Import LessonContent model

logger = logging.getLogger(__name__)

class TeachingResponse(BaseModel):
    """Structured response from the teaching agent."""
    explanation: str
    examples: List[str] = Field(default_factory=list)
    practice_exercises: List[str] = Field(default_factory=list)
    additional_resources: List[str] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)

def create_interactive_teacher_agent() -> Agent:
    """Creates an INTERACTIVE Teacher Agent using Google ADK."""

    # Define the model identifier
    model_identifier = "gemini-2.0-flash-lite" # Or other ADK supported model

    # Import tools needed by the teacher (use manual tool instances)
    from ai_tutor.tools.orchestrator_tools import (
        # Removed read_knowledge_base_tool,
        # Removed get_document_content_tool,
        update_user_model,        # Corrected import name
        reflect_on_interaction,   # For adapting based on interaction
        call_quiz_teacher_evaluate_tool, # Import instance
        quiz_creator_tool,        # CORRECT: Import the tool instance
        get_user_model_status     # To understand current mastery
    )
    from ai_tutor.tools.teacher_tools import ask_user_question_and_get_answer_tool # Keep this custom tool

    teacher_tools = [
        # Removed read_knowledge_base_tool,
        # Removed get_document_content_tool,
        ask_user_question_and_get_answer_tool,
        update_user_model,
        reflect_on_interaction,
        call_quiz_teacher_evaluate_tool,
        quiz_creator_tool  # ADDED: Add the imported tool to the list
    ]

    # Use LLMAgent, define input/output schemas
    teacher_agent = Agent(
        name="interactive_lesson_teacher",
        instruction="""You are an autonomous AI Teacher responsible for guiding a student through a specific `FocusObjective` using the provided document(s) (via Gemini File API).

        WORKFLOW:
        1. **Initial Setup:**
           - Analyze the provided document content relevant to the input `FocusObjective` (topic, learning_goal).
           - Plan micro-steps (explain concept from document, provide examples from document, check understanding).

        2. **Teaching Loop:**
           - Generate explanation text based *only* on the provided document content.
           - Create QuizQuestion JSON based *only* on the provided document content.
           - Use ask_user_question_and_get_answer tool.
           - Evaluate response when execution resumes (use call_quiz_teacher_evaluate tool).
           - Update user model:
             * Call update_user_model tool with outcome.
             * Include confusion points if detected based on the interaction.
             * Mark objectives as mastered when achieved.
           - If struggling:
             * Use reflect_on_interaction tool to analyze.
             * Adapt teaching approach (e.g., re-explain differently using document content, simplify question).
           - Continue or adjust based on progress.

        3. **Completion:**
           - When objective met:
             * Update final mastery via update_user_model tool.
             * Return TeacherTurnResult with status="objective_complete".
           - If user stuck:
             * Call reflect_on_interaction tool for analysis.
             * Update user model with struggles.
             * Return TeacherTurnResult with status="objective_failed".

        IMPORTANT:
        - Base all explanations, examples, and questions *solely* on the content of the document(s) provided in the prompt via the File API.
        - Manage your own teaching loop for the given FocusObjective.
        - Update user model after each interaction.
        - Use reflection when user struggles.
        - Return ONLY TeacherTurnResult JSON as final response.
        """,
        tools=teacher_tools,
        input_schema=FocusObjective,
        model=model_identifier,
    )
    return teacher_agent

async def teach_topic(focus: FocusObjective, context=None, supabase: Client = None) -> Optional[TeachingResponse]:
    """
    Create a teaching response for the given focus objective.
    
    Args:
        focus: FocusObjective containing the topic to teach
        context: Context object with session_id and user state
        supabase: Optional Supabase client instance
        
    Returns:
        A TeachingResponse object containing the teaching material, or None on failure.
    """
    if not focus or not context or not hasattr(context, 'session_id'):
        logger.error("teach_topic: Invalid focus objective or context")
        return None

    # Create the teacher agent - no vector_store_id needed
    teacher_agent = create_interactive_teacher_agent()
    
    # Setup RunConfig for tracing
    run_config = RunConfig(
        workflow_name="AI Tutor - Teaching",
        group_id=str(context.session_id)
    )
    
    # Create prompt for the agent
    user_state = getattr(context, 'user_state', None)
    user_state_str = f"\nCurrent user understanding: {user_state}" if user_state else ""
    
    # Format the focus objective details
    prerequisites_str = "\n- " + "\n- ".join(focus.prerequisites) if focus.prerequisites else "None"
    goals_str = "\n- " + "\n- ".join(focus.learning_goals)
    
    prompt = f"""
    Please create a comprehensive teaching response for the following topic:

    TOPIC: {focus.topic}
    DESCRIPTION: {focus.description}
    PREREQUISITES: {prerequisites_str}
    LEARNING GOALS: {goals_str}
    {user_state_str}

    First, use the `read_knowledge_base` tool to access the document analysis.
    Then, if needed, use `get_document_content` to get more detailed information about this topic.
    Note: The necessary context (like file paths and vector store IDs) will be available via ToolContext when the tools run.

    Create a structured teaching response that:
    1. Explains the topic clearly and thoroughly
    2. Provides relevant examples from the content
    3. Includes practical exercises
    4. Suggests additional resources
    5. Highlights key points
    6. Recommends next steps

    Return your response in the TeachingResponse format.
    """
    
    try:
        result = await Runner.run(
            teacher_agent,
            prompt,
            run_config=run_config,
            context=context  # Pass the context to ensure it's available in ToolContext
        )
        
        if not result or not result.output:
            logger.error("teach_topic: No output from teacher")
            return None
            
        # The output should already be a TeachingResponse thanks to output_schema
        return result.output
        
    except Exception as e:
        logger.error(f"teach_topic: Error during teaching: {str(e)}")
        return None

# Placeholder import for TeacherTurnResult if not defined elsewhere
class TeacherTurnResult(BaseModel):
    status: str = "unknown"
    summary: str = "" 