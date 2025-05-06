# src/agents/executor_loop.py
import asyncio
import json
from loguru import logger
from typing import Any, Dict # Add Any for now
from fastapi import WebSocket, WebSocketDisconnect # Import WebSocket
from fastapi.websockets import WebSocketState # Import WebSocketState for error sending
from pydantic import ValidationError # For LLM response validation
from src.agents.prompts import LEAN_EXECUTOR_PROMPT_TEMPLATE

# Canonical models & helpers
from ai_tutor.models.tool_calls import ToolCall
from ai_tutor.core.llm import LLMClient
from ai_tutor.context import TutorContext as SessionContext
from ai_tutor.api_models import (
    InteractionResponseData,
    ExplanationResponse,
    QuestionResponse,
    FeedbackResponse,
    MessageResponse,
    ErrorResponse,
)
from ai_tutor.core_models import UserModelState
from ai_tutor.agents.models import QuizQuestion

# Placeholder - Define or import the actual objective type later
ObjectiveType = Any

async def run_executor_loop(context: SessionContext, objective: ObjectiveType, websocket: WebSocket) -> None:
    """Runs a single turn of the lean executor: Build prompt, call LLM, parse, dispatch tool."""
    logger.info(f"Running lean executor turn for session {context.session_id}")
    llm = LLMClient()
    # History is managed by the caller (tutor_ws.py) and stored in context
    history: list[dict] = context.history or []

    # --- 1. External Termination Checks (Before LLM Call) ---
    # TODO: Implement mastery check
    # if is_objective_complete(context, objective):
    #    logger.info("Objective complete based on mastery. Ending session.")
    #    tool_call = ToolCall(name="end_session", args={"reason": "objective_complete"})
    #    await dispatch(tool_call, context, websocket) # Dispatch end_session
    #    # How to signal termination back to the caller?
    #    # Maybe dispatch returns a special status or raises an exception?
    #    # For now, dispatch handles sending the end_session message.
    #    return # End the turn

    # TODO: Implement token budget check
    # if context.tokens_used > context.token_budget:
    #    logger.warning("Token budget exceeded. Ending session.")
    #    tool_call = ToolCall(name="end_session", args={"reason": "token_budget_exceeded"})
    #    await dispatch(tool_call, context, websocket) # Dispatch end_session
    #    return # End the turn

    # --- 2. Build Prompt & Call LLM ---
    system_prompt = build_lean_prompt(objective, context.user_model_state)

    # Caller (tutor_ws.py) should have already added the user message to history
    turn_history = history
    if context.last_user_message:
        logger.debug("Using history provided in context (should include last user message)")
        context.last_user_message = None # Consume the message flag if it was set
    else:
        logger.debug("No new user message in context for this turn.")

    logger.debug(f"Calling lean executor LLM for session {context.session_id} with {len(turn_history)} history messages.")
    response_content = None # Initialize in case of errors before assignment
    tool_call: ToolCall | None = None # Initialize tool_call

    try:
        res = await llm.ai_chat(
            messages=turn_history + [{"role": "system", "content": system_prompt}],
            response_format={"type": "json_object"}
        )
        if not res.message or not res.message.content:
             raise ValueError("LLM returned empty message or content")
        response_content = res.message.content

        tool_call_data = json.loads(response_content)
        tool_call = ToolCall(**tool_call_data)
        logger.info(f"Lean executor LLM returned tool call: {tool_call.name}")

        # Append LLM response (tool call JSON) to history
        # The caller (tutor_ws.py) should be responsible for managing history persistence
        # We update the history list here, and the caller saves the context object
        history.append({"role": "assistant", "content": response_content})
        context.history = history # Update context's history for the caller to save

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Failed to parse/validate LLM ToolCall JSON: {e}. Raw: {response_content}")
        await send_error_to_frontend(websocket, "Processing Error", f"Failed to understand tutor's next step: {e}. Raw content: {response_content}")
        return # Exit turn on error
    except Exception as e:
        logger.exception(f"Unexpected error during LLM call in lean executor turn: {e}")
        await send_error_to_frontend(websocket, "Processing Error", f"An unexpected error occurred: {e}")
        return # Exit turn on error

    # --- 3. Dispatch Tool Call ---
    # Ensure tool_call is not None before dispatching
    if tool_call is None:
        logger.error("Tool call object is None after LLM response processing, cannot dispatch.")
        await send_error_to_frontend(websocket, "Internal Error", "Failed to generate a valid action.")
        return # Exit turn

    try:
        # Dispatch handles sending the message via the websocket
        await dispatch(tool_call, context, websocket)
        logger.debug(f"Dispatch completed for tool {tool_call.name}.")

    except WebSocketDisconnect:
        logger.warning(f"WebSocket disconnected during dispatch of tool {tool_call.name}. Cannot proceed.")
        # No further action possible, context saving might fail in caller
        return # Exit turn
    except Exception as dispatch_err:
        logger.exception(f"Error during dispatch of tool {tool_call.name}: {dispatch_err}")
        # Error already sent by dispatch, just exit the turn
        return

    logger.info(f"Lean executor turn finished for session {context.session_id}. Tool: {tool_call.name}")
    # Control returns to the main WebSocket loop in tutor_ws.py

# --- Dispatch Function ---
async def dispatch(call: ToolCall, ctx: SessionContext, websocket: WebSocket) -> bool:
    """Dispatches tool calls to UI or internal functions. Returns True if user reply is needed."""
    logger.info(f"Dispatching tool: {call.name} with args: {call.args}")
    needs_reply = False # Default assumption

    # --- CONSTRUCT RESPONSE for Frontend ---
    response_for_fe: Dict[str, Any] = {
        "user_model_state": ctx.user_model_state.model_dump(mode='json') if ctx.user_model_state else {},
        "whiteboard_actions": None,
        "content_type": "unknown",
        "data": {},
    }

    try:
        match call.name:
            case "explain":
                text = call.args.get("text", "...")
                explanation_payload = ExplanationResponse(explanation_text=text)
                interaction = InteractionResponseData(
                    content_type="explanation",
                    data=explanation_payload,
                    user_model_state=ctx.user_model_state,
                )
                await websocket.send_json(interaction.model_dump(mode="json"))
                needs_reply = True
            case "ask_question":
                question_text = call.args.get("question", "Missing question")
                options = call.args.get("options")
                question_id = call.args.get("question_id", f"q_{ctx.session_id}_{len(ctx.history)}") # Generate a more unique ID

                if options is None:
                   # Free-response question (no options)
                   quiz_q = QuizQuestion(
                       question=question_text,
                       options=[],
                       correct_answer_index=-1,
                       explanation="",
                       difficulty="Medium",
                       related_section=""
                   )
                   question_payload = QuestionResponse(
                       question_type="free_response",
                       question_data=quiz_q,
                       context_summary=None,
                   )
                   interaction = InteractionResponseData(
                       content_type="question",
                       data=question_payload,
                       user_model_state=ctx.user_model_state,
                   )
                   await websocket.send_json(interaction.model_dump(mode="json"))
                   needs_reply = True
                else:
                   # Build QuizQuestion object
                   quiz_q = QuizQuestion(
                       question=question_text,
                       options=options,
                       correct_answer_index=call.args.get("correct_answer_index", -1),
                       explanation=call.args.get("explanation", ""),
                       difficulty=call.args.get("difficulty", "Medium"),
                       related_section=call.args.get("related_section", "")
                   )
                   question_payload = QuestionResponse(
                       question_type="multiple_choice",
                       question_data=quiz_q,
                       context_summary=None,
                   )
                   interaction = InteractionResponseData(
                       content_type="question",
                       data=question_payload,
                       user_model_state=ctx.user_model_state,
                   )
                   await websocket.send_json(interaction.model_dump(mode="json"))
                   needs_reply = True
            case "draw":
                svg_data = call.args.get("svg")
                if svg_data:
                     message_payload = MessageResponse(
                         message_text=f"[Assistant wants to draw: {svg_data[:100]}...]",
                         message_type="status_update",
                     )
                     interaction = InteractionResponseData(
                         content_type="message",
                         data=message_payload,
                         user_model_state=ctx.user_model_state,
                     )
                     await websocket.send_json(interaction.model_dump(mode="json"))
                needs_reply = False
            case "reflect":
                reflection_content = call.args.get("thought", "No thought provided.")
                logger.info(f"Executing reflection (internal step): {reflection_content}")
                # No FE message, just internal processing (if any)
                needs_reply = False
            case "summarise_context":
                logger.info("Executing context summarization request (internal step)")
                # TODO: Implement actual context summarization logic -> update ctx.history
                # No FE message for this internal action
                needs_reply = False
            case "end_session":
                reason = call.args.get("reason", "unknown")
                logger.info(f"Ending session via tool call. Reason: {reason}")
                # TODO: Trigger background analysis task
                # from src.services.session_tasks import queue_session_analysis
                # asyncio.create_task(queue_session_analysis(ctx.session_id, ctx.user_id, ctx.folder_id))
                message_payload = MessageResponse(
                    message_text=f"Session ended: {reason}. Thank you!",
                    message_type="summary",
                )
                interaction = InteractionResponseData(
                    content_type="message",
                    data=message_payload,
                    user_model_state=ctx.user_model_state,
                )
                await websocket.send_json(interaction.model_dump(mode="json"))
                # Consider closing websocket AFTER sending the message
                # await websocket.close(code=1000, reason=f"Session ended: {reason}")
                needs_reply = False # Session ends, no further reply expected
            case _:
                logger.warning(f"Unknown tool name received: {call.name}")
                await send_error_to_frontend(websocket, "Unknown Action", f"Tutor tried an unknown action: {call.name}")
                needs_reply = False

    except WebSocketDisconnect:
         logger.warning(f"WebSocket disconnected during dispatch of tool {call.name}. Cannot send message.")
         raise # Re-raise to signal disconnection to the caller (run_executor_loop)
    except Exception as dispatch_err:
         logger.exception(f"Error dispatching tool call {call.name}: {dispatch_err}")
         # Attempt to send error to frontend if possible
         await send_error_to_frontend(websocket, "Dispatch Error", f"Failed to execute action '{call.name}': {dispatch_err}")
         needs_reply = False # Don't wait if dispatch failed
         # Optional: Re-raise? Depends if caller should halt completely.
         # raise dispatch_err

    # Return value indicates if the tool typically expects a user response
    # Although the loop structure change makes this less critical for control flow now.
    return needs_reply

# --- Prompt Building Function ---
def build_lean_prompt(objective: ObjectiveType, user_model: UserModelState | None) -> str:
    # Serialize user model for prompt
    user_model_str = "No user model available."  # Default if user_model is None
    if user_model:
        try:
           user_model_str = user_model.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Failed to serialize user model: {e}")
            user_model_str = f"Error serializing user model: {e}"

    # Safely extract objective details
    objective_topic = getattr(objective, 'objective_topic', 'N/A')
    objective_goal = getattr(objective, 'learning_goal', 'N/A')
    objective_threshold = getattr(objective, 'target_mastery', 0.8)

    # Build and return the prompt using the centralized template
    return LEAN_EXECUTOR_PROMPT_TEMPLATE.format(
        objective_topic=objective_topic,
        objective_goal=objective_goal,
        objective_threshold=objective_threshold,
        user_state_json=user_model_str
    )

# --- Error Sending Function ---
async def send_error_to_frontend(websocket: WebSocket, title: str, detail: str):
    """Sends a structured error message to the frontend."""
    try:
        error_payload = {
            "content_type": "error",
            "data": {"response_type": "error", "error_message": title, "technical_details": detail},
            "user_model_state": {} # Send empty state on error?
        }
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(error_payload)
            logger.info(f"Sent error to frontend: {title}")
        else:
            logger.warning(f"Cannot send error to frontend, WebSocket state is {websocket.client_state}")
    except Exception as e:
        logger.error(f"Failed to send error to frontend: {e}. Original error: {title} - {detail}")

# --- END Functions --- 