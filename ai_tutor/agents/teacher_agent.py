from __future__ import annotations
from typing import Optional, List, Dict
from uuid import UUID
from pydantic import BaseModel, Field
from supabase import Client
import logging

from google.adk.agents import LlmAgent
from google.adk.runners import Runner, RunConfig
from google.adk.tools import FunctionTool, ToolContext

from ai_tutor.agents.planner_agent import FocusObjective
from ai_tutor.context import TutorContext

logger = logging.getLogger(__name__)

class TeachingResponse(BaseModel):
    """Structured response from the teaching agent."""
    explanation: str
    examples: List[str] = Field(default_factory=list)
    practice_exercises: List[str] = Field(default_factory=list)
    additional_resources: List[str] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)

def create_teacher_agent() -> LlmAgent:
    """Creates a teacher agent that explains concepts using ADK."""

    # Use ADK models
    model_identifier = "gemini-1.5-flash" # Or other ADK supported model

    # Import tools needed by the teacher
    from ai_tutor.tools.orchestrator_tools import read_knowledge_base, get_document_content # Keep common tools
    # Import the tool function, not the decorator result if possible
    # Or ensure the tool instance is correctly exported/imported
    from ai_tutor.tools.teacher_tools import ask_user_question_and_get_answer_tool # Import the specific tool instance

    teacher_tools = [
        read_knowledge_base,
        get_document_content,
        ask_user_question_and_get_answer_tool,
        # Potentially add call_quiz_creator_agent tool if needed
    ]

    # Use LLMAgent, define input/output schemas
    teacher_agent = LlmAgent(
        name="interactive_lesson_teacher", # Use valid Python identifier
        instruction=""" # Use singular 'instruction'
        You are an autonomous AI Teacher responsible for guiding a student through a specific `FocusObjective` provided as input (topic, learning_goal).

        YOUR CONTEXT:
        - You receive the `FocusObjective` details in the initial prompt/input.
        - Maintain your progress internally through the conversation history.
        - Use `read_knowledge_base` or `get_document_content` tools to get content details for explanations. Provide the file path from the context/KB analysis when calling `get_document_content`.
        - Use the `ask_user_question_and_get_answer` tool to pause, ask the user a question (provide the full QuizQuestion JSON as args), and wait for their answer index.

        YOUR AUTONOMOUS TASK:
        1.  **Plan Micro-steps:** Based on the `FocusObjective`, plan a sequence (explain, example, check).
        2.  **Execute Loop:** Iterate through your plan:
            *   **Explain:** Generate explanation text (use content tools if needed). Provide this text directly in your response.
            *   **Check Understanding:** Generate a `QuizQuestion` JSON. Call `ask_user_question_and_get_answer` tool with this JSON as arguments. **Execution Pauses Here.**
            *   **Resume & Evaluate:** When execution resumes, the `FunctionResponse` in the history will contain the user's answer index. Evaluate if it's correct based on the question you asked previously.
            *   **Adapt:** Based on evaluation, decide the next micro-step (next explanation, re-explain, next check, etc.).
        3.  **Objective Completion:** Continue until the `learning_goal` is met or you determine the user is stuck.
        4.  **Return Final Result:** Your *very final message* in this execution run MUST be ONLY a JSON object matching the `TeacherTurnResult` schema (e.g., `{"status": "objective_complete", "summary": "Covered topic X..."}`).

        **CRITICAL:**
        - Manage your own loop.
        - Use `ask_user_question_and_get_answer` tool to get user input when needed.
        - Your FINAL response for this entire run MUST be ONLY the `TeacherTurnResult` JSON. Intermediate explanations/interactions should be plain text or tool calls.
        """,
        tools=teacher_tools, # Pass the list of ADK tools
        input_schema=FocusObjective, # Define input schema (Pydantic model)
        # output_schema=TeacherTurnResult, # REMOVE output_schema
        model=model_identifier, # Correct keyword is 'model'
        # No handoffs needed FROM the teacher in this model
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

    # Create the teacher agent
    teacher_agent = create_teacher_agent()
    
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
            run_config=run_config
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