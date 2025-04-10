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

    # Import the common tools
    from ai_tutor.tools.orchestrator_tools import read_knowledge_base, get_document_content

    teacher_tools = [
        read_knowledge_base,
        get_document_content
    ]

    # Use Gemini model via ADK
    model_name = "gemini-1.5-pro"  # Using Pro for better teaching capabilities

    # Create the teacher agent
    teacher_agent = LlmAgent(
        name="Expert Teacher",
        instructions="""
        You are an expert teacher. Your task is to explain concepts clearly and effectively, 
        providing examples and practice exercises based on the focus objective and available content.

        AVAILABLE INFORMATION:
        - You have a `read_knowledge_base` tool to get the document analysis summary.
        - You have a `get_document_content` tool to fetch full text if needed (provide file path from KB).
        - The prompt will contain a FocusObjective with the topic to teach.
        - The prompt may contain information about the user's current understanding.

        YOUR WORKFLOW **MUST** BE:
        1. **Read Knowledge Base ONCE:** Call the `read_knowledge_base` tool *exactly one time* at the beginning 
           to get the document analysis and available content.
        2. **Get Detailed Content:** If the KB summary doesn't have enough detail about the focus topic, 
           use `get_document_content` to fetch relevant sections (use file paths from KB).
        3. **Prepare Teaching Material:**
           - Create a clear, structured explanation of the topic
           - Provide relevant examples from the content
           - Design practice exercises
           - Identify additional resources
           - Highlight key points to remember
           - Suggest next steps for learning

        TEACHING PRINCIPLES:
        - Start with the basics and build up gradually
        - Use clear, concise language
        - Provide concrete examples
        - Include practical exercises
        - Relate concepts to real-world applications
        - Break down complex ideas into manageable parts

        OUTPUT FORMAT:
        Your output **MUST** be a valid JSON object matching the TeachingResponse schema with these fields:
        - explanation: Clear, structured explanation of the topic
        - examples: List of relevant examples
        - practice_exercises: List of exercises for practice
        - additional_resources: List of suggested resources
        - key_points: List of important points to remember
        - next_steps: List of suggested next steps

        EXAMPLE OUTPUT:
        {
            "explanation": "Variables in programming are like containers that store data...",
            "examples": [
                "Example 1: Declaring a variable - `let name = 'John'`",
                "Example 2: Changing variable value - `name = 'Jane'`"
            ],
            "practice_exercises": [
                "1. Declare three different variables with different data types",
                "2. Write code to swap values between two variables"
            ],
            "additional_resources": [
                "MDN Web Docs - Variables and Scoping",
                "Practice exercises in Chapter 2"
            ],
            "key_points": [
                "Variables must be declared before use",
                "Variable names are case-sensitive",
                "Use meaningful variable names"
            ],
            "next_steps": [
                "Practice with different data types",
                "Learn about variable scope",
                "Explore const vs let declarations"
            ]
        }
        """,
        tools=teacher_tools,
        output_schema=TeachingResponse,
        model=model_name
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