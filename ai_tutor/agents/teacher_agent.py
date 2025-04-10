from __future__ import annotations

import os
import json

from google.adk.agents import LLMAgent # Use ADK Agent
from google.adk.tools import BaseTool, FunctionTool, FilesRetrieval, ToolContext, LongRunningFunctionTool # ADK Tools
from google.adk.agents import types as adk_types # ADK types

from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import FocusObjective, QuizQuestion, ExplanationResult, TeacherTurnResult # Input/Output models
from typing import List, Callable, Optional, Any, Dict, TYPE_CHECKING, Union, AsyncGenerator
from ai_tutor.agents.utils import RoundingModelWrapper # Keep if rounding is needed for Gemini

# Import the new tools
# from ai_tutor.tools.teacher_tools import ask_user_question_and_get_answer # Import the long-running tool

if TYPE_CHECKING:
    from ai_tutor.context import TutorContext


# Define the output type for the Teacher Agent (after completing an objective)
# Output type for the Agent when called *as a tool* by Orchestrator
TeacherAgentToolOutput = TeacherTurnResult

def create_interactive_teacher_agent(vector_store_id: str) -> LLMAgent:
    """Creates an INTERACTIVE Teacher Agent."""

    # provider: OpenAIProvider = OpenAIProvider() # Use ADK models
    # Maybe use a slightly more capable model for interactive logic
    # base_model = provider.get_model("gpt-4o-2024-08-06")
    model_name = "gemini-1.5-flash" # Or other ADK supported model

    # ADK Tool Setup - Similar to Planner, replace FileSearchTool if needed
    file_search_tool = BaseTool(
        name="file_search",
        description="Use this tool to search for documents related to a topic",
        func=FilesRetrieval(vector_store_ids=[vector_store_id], max_num_results=3)
    )

    # Define the tools the *teacher itself* can use
    # Crucially, add the custom long-running tool
    teacher_tools = [
        file_search_tool, # Or replacement like get_document_content
        ask_user_question_and_get_answer, # The interactive tool
        # Potentially add call_quiz_creator_agent if teacher generates its own checks
    ]

    # Use LLMAgent, define input/output schemas
    teacher_agent = LLMAgent(
        name="InteractiveLessonTeacher",
        # Instructions now describe the internal autonomous loop
        instructions="""
        You are an autonomous AI Teacher responsible for guiding a student through a specific `FocusObjective` provided as input.

        YOUR CONTEXT:
        - You receive a `FocusObjective` detailing the `topic`, `learning_goal`, and `relevant_concepts`.
        - You maintain your own internal state/progress for this objective during your execution loop.
        - You use `file_search` (or similar) to get content details if needed.
        - You use `ask_user_question_and_get_answer` to pause your execution, ask the user a question, and wait for their answer.

        YOUR TASK:
        1.  **Plan Micro-steps:** Based on the `FocusObjective`, plan a sequence of micro-steps (e.g., Explain concept -> Provide Example -> Ask Check Question -> Explain related concept -> Ask Check Question).
        2.  **Execute Loop:** Iterate through your micro-plan:
            *   **Explain:** Generate a concise explanation for the current micro-step (using `file_search` if needed). Store this explanation.
            *   **Check Understanding:** Generate a relevant `QuizQuestion` (potentially using `call_quiz_creator_agent` tool if available). Call the `ask_user_question_and_get_answer` tool, passing the question data. **Execution Pauses Here.**
            *   **Resume & Evaluate:** When execution resumes, you will receive the user's answer index from the `ask_user_question_and_get_answer` tool result. Evaluate if the answer is correct.
            *   **Adapt:** Based on the evaluation:
                *   Correct: Move to the next micro-step. Increase internal mastery estimate.
                *   Incorrect: Re-explain differently, provide another example, or ask a simpler question. Decrease internal mastery estimate.
        3.  **Objective Completion:** Continue the loop until your internal assessment indicates the `learning_goal` of the `FocusObjective` is met OR you determine the user is stuck and cannot proceed.
        4.  **Return Final Result:** Output a single `TeacherTurnResult` JSON object indicating `objective_complete` or `objective_failed` along with a summary.

        **CRITICAL:**
        - Manage your own internal loop to achieve the `FocusObjective`.
        - Use the `ask_user_question_and_get_answer` tool when you need user input.
        - Your final output MUST be a single `TeacherTurnResult` JSON object when the objective is finished (or failed).
        """,
        tools=teacher_tools,
        input_schema=FocusObjective, # Define input schema
        output_schema=TeacherTurnResult, # Define output schema
        model=model_name, # Use model name string for ADK
        # No handoffs needed FROM the teacher in this model
    )
    return teacher_agent

# Removed old functions:
# - lesson_content_handoff_filter
# - create_teacher_agent
# - create_teacher_agent_without_handoffs
# - generate_lesson_content 