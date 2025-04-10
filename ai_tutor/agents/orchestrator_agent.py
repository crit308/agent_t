from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING, Union, cast

from agents import Agent, Runner, RunConfig, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Use TYPE_CHECKING for TutorContext import
if TYPE_CHECKING:
    from ai_tutor.context import TutorContext # Use the enhanced context
# Import tool functions (assuming they will exist in a separate file)
from ai_tutor.tools.orchestrator_tools import (
    call_quiz_creator_agent,
    call_quiz_teacher_evaluate,
    update_user_model,
    get_user_model_status,
    reflect_on_interaction,
)

# Import models needed for type hints if tools return them
# Also import models needed for the output Union type
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, LessonContent
from ai_tutor.api_models import (
    TutorInteractionResponse, ExplanationResponse, QuestionResponse,
    FeedbackResponse, MessageResponse, ErrorResponse
)

from google.adk.tools import AgentTool # Import AgentTool wrapper

def create_orchestrator_agent(api_key: str = None) -> Agent['TutorContext']:
    """Creates the Orchestrator Agent for the AI Tutor."""

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # --- ADK Setup ---
    # model_name = "gemini-1.5-pro" # Or another capable model
    model_name = "gemini-1.5-flash" # Use flash for now
    
    # Create instances of the agents to be wrapped as tools
    # We need vector_store_id here if the agents need it during creation.
    # This implies Orchestrator needs access to vs_id, maybe via context/state.
    # For simplicity, assume it's available or passed differently. A better way
    # might be to configure tools externally and pass them in.
    # Placeholder: Assume vs_id is accessible. This needs refinement.
    placeholder_vs_id = "dummy-vs-id" # Replace with actual mechanism to get vs_id
    planner_agent_instance = create_planner_agent(placeholder_vs_id)
    teacher_agent_instance = create_interactive_teacher_agent(placeholder_vs_id)
    
    orchestrator_tools = [
        AgentTool(planner_agent_instance), # Wrap planner agent
        AgentTool(teacher_agent_instance), # Wrap teacher agent
        # Keep utility tools if Orchestrator still uses them directly
        update_user_model,
        get_user_model_status,
        reflect_on_interaction,
    ]

    orchestrator_agent = LLMAgent( # Use ADK LLMAgent
        name="TutorOrchestrator",
        instructions="""
        You are the central conductor of an AI tutoring session. Your primary goal is to guide the user towards mastering specific learning objectives identified by the Planner Agent.

        CONTEXT:
        - You operate based on the `current_focus_objective` provided in the `TutorContext`. This objective (topic, goal) is set by the Planner Agent.
        - If `current_focus_objective` is missing, your FIRST action MUST be to call `call_planner_agent` to get the initial focus.
        - You delegate the teaching of the `current_focus_objective` to the `call_teacher_agent` tool.
        - The `call_teacher_agent` tool runs autonomously and returns a `TeacherTurnResult` (indicating objective completion status) when finished.
        - `reflect_on_interaction` helps you analyze difficulties and adapt your strategy.
        - User's last input/action is provided in the prompt.

        **Core Responsibilities:**
        1.  **Get Objective:** If no `current_focus_objective` in state, call `call_planner_agent` tool. Store the returned `FocusObjective` in state. -> END TURN.
        2.  **Delegate Teaching:** If `current_focus_objective` exists, call the `call_teacher_agent` tool, passing the objective details (topic, goal, concepts) as arguments. -> END TURN.
        3.  **Process Teacher Result:** The `call_teacher_agent` tool will run until the objective is complete or failed. You will receive its final `TeacherTurnResult`.
        4.  **Update State:** Based on `TeacherTurnResult`, update the overall session state using `update_user_model` (e.g., mark concepts as mastered/struggled based on teacher's summary).
        5.  **Loop:**
            *   If the Teacher reported success (`objective_complete`), call `call_planner_agent` again to get the *next* focus objective. Store it. -> END TURN.
            *   If the Planner indicates no more objectives, the session is complete. Respond with a concluding message. -> END TURN.
            *   If the Teacher reported failure (`objective_failed`), decide how to proceed. Maybe call `reflect_on_interaction` or try the Planner again for a different approach, or end the session. -> END TURN.

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
        - Ensure your final output strictly adheres to the required JSON format (`TutorInteractionResponse`).
        """,
        tools=orchestrator_tools,
        output_schema=TutorInteractionResponse, # Or a simpler status model
        model=model_name,
    )

    return orchestrator_agent 