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
# Import the *tools* the orchestrator calls
# --- Import UTILITY tools ---
from ai_tutor.tools.orchestrator_tools import (
    # These are now implemented as AgentTools below
    # call_planner_agent, call_teacher_agent, call_quiz_creator_agent,
    update_user_model_tool,
    get_user_model_status_tool,
    reflect_on_interaction_tool,
)

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
    model_identifier = "gemini-2.0-flash" # Use flash for now
    
    # --- Create Agent Instances for AgentTool ---
    # Agent creation no longer needs VS ID directly - agents get context when their tools run
    planner_agent_instance: Agent = create_planner_agent() # Create without vs_id
    teacher_agent_instance: Agent = create_interactive_teacher_agent() # Create without vs_id
    
    orchestrator_tools = [
        AgentTool(planner_agent_instance), # Name is inherited from planner_agent_instance.name
        AgentTool(teacher_agent_instance), # Name is inherited from teacher_agent_instance.name
        # Keep utility tools if Orchestrator still uses them directly
        update_user_model_tool,
        get_user_model_status_tool,
        reflect_on_interaction_tool,
    ]

    orchestrator_agent = Agent(
        name="tutor_orchestrator", # Use valid Python identifier
        instruction="""
        You are the high-level conductor of an AI tutoring session. Your primary goal is to sequence learning objectives.

        CONTEXT:
        - You access the session state (including `current_focus_objective`, `UserModelState`, and other necessary IDs) via the `ToolContext`.
        - The Planner and Teacher agents you call will also receive this context when their tools run.
        - If `current_focus_objective` is missing in the state, your FIRST action MUST be to call the `call_planner_agent` tool to get the initial focus.
        - You delegate the teaching of the `current_focus_objective` to the `call_teacher_agent` tool.
        - The `call_teacher_agent` tool runs autonomously and returns a `TeacherTurnResult` (indicating objective completion status) when finished.
        - You evaluate user answers to checking questions using `call_quiz_teacher_evaluate`.
        - The `reflect_on_interaction_tool` helps analyze difficulties if the Teacher tool reports failure.
        - User's last input/action is provided in the prompt.

        **Core Responsibilities:**
        1.  **Get Objective:** If no `current_focus_objective` in state, call `call_planner_agent` tool. Store the returned `FocusObjective` in state. -> END TURN.
        2.  **Delegate Teaching:** If `current_focus_objective` exists, call the `call_teacher_agent` tool, passing the objective details (topic, goal, concepts) as arguments. -> END TURN.
        3.  **Process Teacher Result:** The `call_teacher_agent` tool will run until the objective is complete or failed. You will receive its final `TeacherTurnResult`.
        4.  **Update State:** Based on `TeacherTurnResult`, update the overall session state using `update_user_model` (e.g., mark concepts as mastered/struggled based on teacher's summary).
        5.  **Loop:**
            *   If the Teacher reported success (`objective_complete`), call the `call_planner_agent` tool again to get the *next* focus objective. Store it in state. -> END TURN.
            *   If the Planner indicates no more objectives (e.g., returns a specific signal or empty objective), the session is complete. Respond with a concluding message. -> END TURN.
            *   If the Teacher reported failure (`objective_failed`), decide how to proceed. Maybe call `reflect_on_interaction_tool` or try the `call_planner_agent` again for a different approach, or end the session. -> END TURN.

        CORE WORKFLOW:
        1. Check session state for `current_focus_objective`.
        2. If NONE -> Call `call_planner_agent` tool -> Update state -> END TURN.
        3. If EXISTS -> Call `call_teacher_agent` tool with objective details -> END TURN.
        4. When `call_teacher_agent` returns `TeacherTurnResult`: Process result -> Update state -> Go back to step 1 (to get next objective from planner).

        HANDLING USER INPUT DURING TEACHING:
        - The user interaction (asking questions, providing answers) is handled *within* the autonomous `call_teacher_agent` tool execution via its own long-running tools. You, the Orchestrator, wait for the Teacher tool to fully complete its objective before proceeding.

        PRINCIPLES:
        - **High-Level Orchestration:** Your job is to get the objective and delegate it.
        - **Delegate Autonomy:** Trust the specialist agent tools (Planner, Teacher) to manage their own internal processes.
        - **State Management:** Keep `UserModelState` updated via tools.
        - **Context Passing:** All necessary context (vector store IDs, file paths, etc.) is passed automatically to tools via ToolContext.
        - Ensure your final output strictly adheres to the required JSON format (`TutorInteractionResponse`).
        """,
        tools=orchestrator_tools,
        # output_schema=TutorInteractionResponse, # REMOVE - Orchestrator manages flow, doesn't generate this directly
        model=model_identifier
    )

    return orchestrator_agent 