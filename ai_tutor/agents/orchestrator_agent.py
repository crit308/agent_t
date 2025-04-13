from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING, Union, cast

# Use ADK imports
from google.adk import Agent # Use top-level Agent alias
from google.adk.runners import Runner, RunConfig # Use ADK Runner/Config
from google.adk.tools.agent_tool import AgentTool # Correct import path

# Use TYPE_CHECKING for TutorContext import
if TYPE_CHECKING:
    from ai_tutor.context import TutorContext # Use the enhanced context
# Import the agent creation functions needed
from ai_tutor.agents.planner_agent import create_planner_agent
from ai_tutor.agents.teacher_agent import create_interactive_teacher_agent

# Import models needed for type hints if tools return them
# Also import models needed for the output Union type
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, LessonContent
from ai_tutor.api_models import (
    TutorInteractionResponse, ExplanationResponse, QuestionResponse,
    FeedbackResponse, MessageResponse, ErrorResponse
)

def create_orchestrator_agent(api_key: str = None) -> Agent:
    """Creates the Orchestrator Agent for the AI Tutor."""

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    # --- ADK Model Setup ---
    model_identifier = "gemini-2.0-flash"
    
    # --- Create Agent Instances for AgentTool ---
    planner_agent_instance: Agent = create_planner_agent()
    teacher_agent_instance: Agent = create_interactive_teacher_agent()
    
    orchestrator_tools = [
        AgentTool(planner_agent_instance),
        AgentTool(teacher_agent_instance),
    ]

    orchestrator_agent = Agent(
        name="tutor_orchestrator",
        instruction="""You are the orchestrator of an AI tutoring session, coordinating between a Planner agent that determines learning objectives and a Teacher agent that delivers instruction.

        WORKFLOW:
        1. CHECK STATE:
           - Access session state via ToolContext
           - Look for current_focus_objective
           - If missing -> Get from Planner
           - If exists -> Delegate to Teacher

        2. GET OBJECTIVE (when needed):
           - Call call_planner_agent tool
           - Store returned FocusObjective in state
           - End turn

        3. TEACH OBJECTIVE:
           - Call call_teacher_agent with objective details
           - Teacher handles all user interaction autonomously
           - Teacher manages user model updates and reflection
           - Wait for TeacherTurnResult
           - End turn

        4. PROCESS RESULTS:
           - On success: Get next objective from Planner
           - On failure: Let Teacher handle reflection and retry
           - End turn

        IMPORTANT:
        - Let specialist agents handle their domains
        - Trust Teacher to manage user state and reflection
        - End turn after delegating to allow autonomous operation

        Focus on coordinating the high-level flow between Planner and Teacher agents, letting them handle their specialized tasks independently.
        """,
        tools=orchestrator_tools,
        model=model_identifier
    )

    return orchestrator_agent 