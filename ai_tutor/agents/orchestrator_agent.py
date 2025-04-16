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
    model_identifier = "gemini-2.0-flash-lite"
    
    # --- Create Agent Instances for AgentTool ---
    planner_agent_instance: Agent = create_planner_agent()
    teacher_agent_instance: Agent = create_interactive_teacher_agent()
    
    orchestrator_tools = [
        AgentTool(planner_agent_instance),
        AgentTool(teacher_agent_instance),
    ]

    orchestrator_agent = Agent(
        name="tutor_orchestrator",
        instruction="""Your goal is to manage the tutoring session flow between a Planner and a Teacher.

        STATE:
        - Check session state for `current_focus_objective`.

        WORKFLOW:
        1. IF `current_focus_objective` is MISSING in state:
           - Call the `focus_planner` tool (no arguments).
           - It returns a `FocusObjective` JSON object.
           - Store this **entire JSON object** in the session state under the key `current_focus_objective`.
           - STOP your turn.

        2. IF `current_focus_objective` EXISTS in state:
           - Retrieve the **entire `FocusObjective` JSON object** from the state.
           - Call the `interactive_lesson_teacher` tool.
           - **CRITICAL JSON ARGUMENT STRUCTURE:** The `interactive_lesson_teacher` tool expects **ONE** argument named `focus_objective`. The **VALUE** of this argument MUST be the **complete `FocusObjective` JSON object** you retrieved from the state.
           - Example of the JSON structure you must generate for the function call arguments:
             `{ \"focus_objective\": { ... the full FocusObjective JSON object ... } }`
           - After the `interactive_lesson_teacher` tool finishes and returns its result, clear the `current_focus_objective` key from the session state.
           - STOP your turn.

        RULES:
        - Follow the workflow precisely.
        - When calling `interactive_lesson_teacher`, ensure the arguments JSON has exactly one key, `focus_objective`, and its value is the full JSON object from the state.
        - Update or clear the `current_focus_objective` state key as described.
        - Do not add conversational text, just execute the workflow step.
        """,
        tools=orchestrator_tools,
        model=model_identifier
    )

    return orchestrator_agent 