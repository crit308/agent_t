from __future__ import annotations
from uuid import UUID
from typing import Any, Dict, Optional
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from supabase import Client
from postgrest.exceptions import APIError

from ai_tutor.dependencies import get_supabase_client
from ai_tutor.session_manager import SessionManager
from ai_tutor.context import TutorContext, UserModelState
import json
import logging
import traceback
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect, WebSocketState
from ai_tutor.api_models import (
    InteractionResponseData, ExplanationResponse, QuestionResponse,
    FeedbackResponse, MessageResponse, ErrorResponse
)
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, FocusObjective
from fastapi import HTTPException
from ai_tutor.agents.planner_agent import determine_session_focus
from ai_tutor.interaction_logger import log_interaction
from ai_tutor.utils.tool_helpers import invoke  # Import invoke globally to avoid local shadowing
import asyncio
from ai_tutor.services.session_tasks import queue_session_analysis
import typing

# Import prompt template (moved to ai_tutor.prompts)
from ai_tutor.prompts import LEAN_EXECUTOR_PROMPT_TEMPLATE
from ai_tutor.models.tool_calls import ToolCall
from ai_tutor.core.llm import LLMClient

router = APIRouter()

# Use shared session manager instance (same as other routers) – no stateful behaviour here
session_manager = SessionManager()

# Helper to authenticate a websocket connection and return the Supabase user
ALLOW_URL_TOKEN = os.getenv("ENV", "prod") != "prod"

log = logging.getLogger(__name__)

# --- Validation Helper --- #

def validate_interaction_response(data_to_validate: Any, log_context: str = "") -> Optional[InteractionResponseData]:
    """Validates data against InteractionResponseData Pydantic model.

    Args:
        data_to_validate: The data object (ideally already InteractionResponseData or dict).
        log_context: String description for logging (e.g., 'Executor Response').

    Returns:
        The validated InteractionResponseData object if successful, otherwise None.
        Logs errors if validation fails.
    """
    try:
        if isinstance(data_to_validate, InteractionResponseData):
            # Already the correct type, re-validate to be sure (optional but safe)
            validated_data = InteractionResponseData.model_validate(data_to_validate.model_dump())
            log.debug(f"validate_interaction_response ({log_context}): Data already InteractionResponseData, validation successful.")
            return validated_data
        elif isinstance(data_to_validate, dict):
            validated_data = InteractionResponseData.model_validate(data_to_validate)
            log.debug(f"validate_interaction_response ({log_context}): Dict validated successfully.")
            return validated_data
        else:
            log.error(f"validate_interaction_response ({log_context}): Input data is not InteractionResponseData or dict, type: {type(data_to_validate)}.")
            return None
    except ValidationError as e:
        log.error(f"validate_interaction_response ({log_context}): Validation failed! Error: {e}. Raw Data: {data_to_validate}", exc_info=True)
        return None
    except Exception as e:
        log.error(f"validate_interaction_response ({log_context}): Unexpected error during validation! Error: {e}. Raw Data: {data_to_validate}", exc_info=True)
        return None

# --- End Validation Helper --- #

# Helper function to safely send JSON
async def safe_send_json(ws: WebSocket, data: Any, log_context: str = ""):
    """Attempts to send JSON data, catching errors if the socket is closed."""
    try:
        if ws.client_state == WebSocketState.CONNECTED:
             log.debug(f"safe_send_json ({log_context}): Sending data.")
             await ws.send_json(data)
        else:
             log.warning(f"safe_send_json ({log_context}): WebSocket not connected (state={ws.client_state}). Skipping send.")
    except (RuntimeError, WebSocketDisconnect) as e:
        # Catch errors specifically related to sending on a closed/closing socket
        log.warning(f"safe_send_json ({log_context}): Failed to send message, socket likely closed: {e}")
    except Exception as e:
        log.error(f"safe_send_json ({log_context}): Unexpected error during send: {e}", exc_info=True)

# Helper to safely close WebSocket
async def safe_close(ws: WebSocket, code: int = 1000, reason: Optional[str] = None):
    """Attempts to close the WebSocket connection gracefully."""
    try:
        if ws.client_state == WebSocketState.CONNECTED or ws.client_state == WebSocketState.CONNECTING:
            log.info(f"safe_close: Attempting to close WebSocket (code={code}, reason='{reason}'). Current state: {ws.client_state}")
            await ws.close(code=code, reason=reason)
            log.info("safe_close: WebSocket close call completed.")
        elif ws.client_state == WebSocketState.DISCONNECTED:
             log.info("safe_close: WebSocket already disconnected.")
        else:
             log.warning(f"safe_close: WebSocket in unexpected state {ws.client_state}, cannot close.")
    except (RuntimeError, WebSocketDisconnect) as e:
         log.warning(f"safe_close: Error during WebSocket close: {e}") # Log expected errors during close
    except Exception as e:
         log.error(f"safe_close: Unexpected error during close: {e}", exc_info=True)

# --- Removed Stub Functions --- #
# Removed run_planner_stub and run_executor_stub definitions

# --- Helper function to send standardized error responses --- #
async def send_error_response(ws: WebSocket, message: str, error_code: str, details: Optional[str] = None, state: Optional[UserModelState] = None):
    log.error(f"Sending error to client: Code={error_code}, Message='{message}', Details='{details}'")
    try:
        error_payload = ErrorResponse(
            error_message=message,  # Use correct field name
            error_code=error_code,
            technical_details=details  # Pass details to technical_details
        )
        state_obj = state if state else UserModelState()
        full_response = InteractionResponseData(
            content_type="error",
            data=error_payload,
            user_model_state=state_obj
        )
        await safe_send_json(ws, full_response.model_dump(mode='json'), f"Error Response Send ({error_code})")
    except Exception as send_err:
        log.critical(f"Failed to create/send structured error response (Code={error_code}): {send_err}", exc_info=True)
        try:
            await safe_send_json(ws, {"content_type": "error", "data": {"error_message": "An internal error occurred while reporting an error."}}, "Fallback Error Send")
        except Exception as fallback_err:
            log.critical(f"Failed to send fallback error message: {fallback_err}")

# --- WebSocket Authentication Helper ---
async def _authenticate_ws(ws: WebSocket, supabase: Client) -> typing.Any:
    """Authenticate a websocket connection using JWT from headers or query params."""
    # Try to get token from headers
    token = None
    auth_header = ws.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1]
    # Fallback: try to get token from query params
    if not token:
        token = ws.query_params.get("token")
    if not token:
        raise HTTPException(status_code=403, detail="Missing authentication token for websocket connection.")
    try:
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=403, detail="Invalid or expired token for websocket connection.")
        ws.state.user = user  # Attach user to websocket state for downstream use
        return user
    except Exception as e:
        log.error(f"WebSocket authentication failed: {e}")
        raise HTTPException(status_code=403, detail="Could not validate websocket credentials.")

@router.websocket("/ws/session/{session_id}")
async def tutor_stream(
    ws: WebSocket,
    session_id: UUID,
    supabase: Client = Depends(get_supabase_client),
):
    """Stream tutor interaction events for a session via WebSocket.

    The client must provide a valid Supabase JWT in the `Authorization` header (Bearer token).
    Each inbound JSON message is forwarded to the TutorFSM orchestrator. All streaming events
    emitted by the FSM are relayed back to the client in real-time.
    """
    log.info(f"WebSocket attempting connection for session {session_id}")
    user = None
    ctx: Optional[TutorContext] = None  # Initialize ctx before try
    session_ended_cleanly = False # Task 2.2: Initialize flag
    try:
        log.info(f"WebSocket: Authenticating for session {session_id}")
        user = await _authenticate_ws(ws, supabase)
        log.info(f"WebSocket: Authentication successful for user {user.id}, session {session_id}")
    except RuntimeError as auth_err:
        log.warning(f"WebSocket: Authentication failed for session {session_id}: {auth_err}")
        return
    except Exception as e:
         log.error(f"WebSocket: Unexpected error during authentication for session {session_id}: {e}\n{traceback.format_exc()}", exc_info=True)
         # Attempt to close gracefully, but don't try to send an error if auth failed early
         try:
              if ws.client_state != WebSocketState.DISCONNECTED: await ws.close(code=1008)
         except Exception: pass
         return

    try:
        row: Optional[Dict] = None
        try:
            log.info(f"WebSocket: Fetching context from DB for session {session_id}")
            select_resp = supabase.table("sessions").select("context_data").eq("id", str(session_id)).eq("user_id", str(user.id)).maybe_single().execute()
            row = select_resp.data
            log.info(f"WebSocket: DB fetch completed for session {session_id}. Data found: {'Yes' if row else 'No'}")
            if row:
                 log.debug(f"WebSocket: Raw data dict from DB for {session_id}: {row}")

        except APIError as api_err:
            if api_err.code == "204":
                log.info(f"WebSocket: No existing context found for session {session_id} (APIError code 204). Initializing fresh context.")
                row = None
            else:
                log.error(f"WebSocket: Supabase APIError fetching context for {session_id}: Code={api_err.code}, Message={api_err.message}, Details={api_err.details}", exc_info=True)
                await safe_send_json(ws, {"type": "error", "detail": "Internal server error fetching context."}, "DB APIError")
                await safe_close(ws, code=1011)
                return
        except Exception as db_err:
            log.error(f"WebSocket: Unexpected error fetching context from DB for {session_id}: {db_err}\n{traceback.format_exc()}", exc_info=True)
            await safe_send_json(ws, {"type": "error", "detail": "Internal server error fetching context."}, "DB Error")
            await safe_close(ws, code=1011)
            return

        try:
             if row and row.get("context_data"):
                  log.info(f"WebSocket: Hydrating TutorContext from DB data for {session_id}")
                  context_dict = row["context_data"]
                  if isinstance(context_dict, str):
                       context_dict = json.loads(context_dict)
                  context_dict.setdefault('session_id', str(session_id))
                  context_dict.setdefault('user_id', str(user.id))
                  ctx = TutorContext.model_validate(context_dict)
                  log.info(f"WebSocket: TutorContext hydrated successfully for {session_id}. Loaded folder_id: {ctx.folder_id}")
             else:
                  log.info(f"WebSocket: Initializing new TutorContext for {session_id} (no prior data or context_data missing)")
                  ctx = TutorContext(session_id=session_id, user_id=user.id, folder_id=None)
                  log.info(f"WebSocket: Initialized fresh context for {session_id}")

        except (ValidationError, TypeError, json.JSONDecodeError) as parse_error:
            log.error(f"WebSocket: Failed to parse/validate context_data for session {session_id}: {parse_error}\nRaw context_data: {row.get('context_data') if row else 'N/A'}", exc_info=True)
            await send_error_response(ws, "Internal server error processing context.", "CONTEXT_PARSE_ERROR", details=str(parse_error), state=ctx.user_model_state if ctx else None)
            await safe_close(ws, code=1011)
            return
        except Exception as ctx_err:
            log.error(f"WebSocket: Unexpected error initializing context for {session_id}: {ctx_err}\n{traceback.format_exc()}", exc_info=True)
            await send_error_response(ws, "Internal server error initializing context.", "CONTEXT_INIT_ERROR", details=str(ctx_err), state=ctx.user_model_state if ctx else None)
            await safe_close(ws, code=1011)
            return

        await ws.accept()
        log.info(f"WebSocket: Connection accepted for session {session_id}")

        # --- NEW: Send Whiteboard State on Reconnect ---
        if ctx and ctx.whiteboard_history:
            log.info(f"WebSocket: Found whiteboard history for session {session_id}. Sending state.")
            # Flatten the list of lists into a single list of actions
            all_actions = [action for action_list in ctx.whiteboard_history for action in action_list]
            log.debug(f"WebSocket: Sending {len(all_actions)} total whiteboard actions for hydration.")
            try:
                await safe_send_json(ws, {
                    "content_type": "whiteboard_state",
                    "data": {"actions": all_actions}
                }, "Whiteboard State Hydration")
                log.info(f"WebSocket: Whiteboard state sent successfully for session {session_id}.")
            except Exception as send_exc:
                log.error(f"WebSocket: Failed to send whiteboard state for session {session_id}: {send_exc}")
                # Decide if this is critical - maybe close connection?
                # For now, log and continue, FE might be partially broken.

        if not ctx:
             log.error(f"WebSocket: CRITICAL - Context object is None after loading/initialization for session {session_id}.")
             await send_error_response(ws, "Internal server error: context unavailable.", "CONTEXT_NULL_ERROR", state=ctx.user_model_state if ctx else None)
             await safe_close(ws, code=1011)
             return

        log.info(f"WebSocket: Context verified after accept. folder_id = {ctx.folder_id}")

        # --- Resume/State Handling Logic ---
        if getattr(ctx, 'current_quiz_question', None):
            log.info(f"WebSocket: Found pending question in context for session {session_id}. Sending to client.")
            pending_question_payload = {
                "type": "question",
                "question": ctx.current_quiz_question.model_dump(mode='json'),
                "topic": getattr(ctx, 'current_teaching_topic', "Unknown Topic")
            }
            await safe_send_json(ws, {
                 "content_type": "question",
                 "data": pending_question_payload,
                 "user_model_state": ctx.user_model_state.model_dump(mode='json')
            }, "Pending Question Send")
            # --- FIX: Ensure whiteboard state is included if question had actions ---
            # Currently, `draw_mcq_actions` adds actions to the response, which are saved.
            # The `whiteboard_state` message above handles the full history replay.
            # However, if we *only* send the pending question, it won't have its original actions.
            # Let's assume the `whiteboard_state` replay is sufficient for now.
            log.info(f"WebSocket: Pending question sent for session {session_id}.")
        else:
             log.info(f"WebSocket: No pending question found in context for session {session_id}.")

        log.info(f"WebSocket: Entering main receive loop")
        planner_run_complete = False
        if ctx and ctx.current_focus_objective:
            log.info("Existing FocusObjective found in loaded context. Skipping Planner.")
            planner_run_complete = True

        while True:
            try:
                # --- Ensure a focus objective exists (run planner if missing) ---
                if not ctx.current_focus_objective:
                    log.info(f"No focus objective found for session {session_id}. Running planner.")
                    try:
                        new_objective = await determine_session_focus(ctx)
                        ctx.current_focus_objective = new_objective
                        planner_run_complete = True
                        await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                        log.info(f"Planner determined objective '{new_objective.topic}' for session {session_id}.")
                    except Exception as plan_err:
                        log.error(f"Planner failed for session {session_id}: {plan_err}")
                        await _send_ws_error(ws, "Planning Error", "Could not determine lesson objective.")
                        continue  # wait for next client message

                payload_text = await ws.receive_text()
                log.debug(f"WebSocket: Received raw message for {session_id}: {payload_text}")
                payload = json.loads(payload_text)

                # Log User Interaction AFTER parsing
                if ctx and user: # Ensure context and user are available for logging
                     event_type = payload.get('type')
                     user_input_text = None
                     if event_type == 'user_message':
                          user_input_text = payload.get('data', {}).get('text', '')
                          if user_input_text:
                              await log_interaction(ctx, 'user', user_input_text, 'user_input', event_type=event_type)
                     elif event_type in ['next', 'answer', 'start']: # Log other user actions
                          await log_interaction(ctx, 'user', f"User action: {event_type}", 'user_action', event_type=event_type)
                # --- End User Interaction Logging ---

                event_data = payload.get('data', {})
                event_type = payload.get('type')
                user_input_text = event_data.get('text') if event_type == 'user_message' else None

                if not event_type:
                    log.warning(f"[WebSocket Warning] Received message without 'type' for {session_id}: {payload_text}")
                    await send_error_response(ws, "Message missing 'type' field.", "INVALID_PAYLOAD", state=ctx.user_model_state if ctx else None)
                    continue

                if event_type == 'ping' or event_type == 'system_tick':
                     log.debug(f"WebSocket: Received system message, ignoring for {session_id}.", message_type=event_type)
                     continue

                # Task 2.1 Refinement: Handle end_session event with status check
                elif event_type == 'end_session':
                    log.info(f"Received 'end_session' event from user for session {session_id}.")
                    session_ended_cleanly = True # Flag to prevent disconnect trigger

                    if ctx and user:
                        try:
                            log.info(f"Checking analysis status before triggering background task via 'end_session' for {session_id}")
                            # --- Check Status FIRST ---
                            supabase_client = await get_supabase_client()
                            status_check = supabase_client.table("sessions") \
                                .select("analysis_status") \
                                .eq("id", str(session_id)) \
                                .maybe_single() \
                                .execute()
                            current_status = status_check.data.get("analysis_status") if status_check.data else None
                            log.info(f"Current analysis_status for session {session_id} before 'end_session' trigger: '{current_status}'")

                            if current_status is None:
                                # --- Trigger Background Task ---
                                asyncio.create_task(queue_session_analysis(session_id, user.id, ctx.folder_id))
                                log.info(f"Background analysis task created for session {session_id}")

                                # --- Send Confirmation to Client ---
                                confirmation_payload = MessageResponse(
                                    response_type="message", # Use standard type
                                    text="Session ending signal received. Your progress analysis will begin shortly."
                                )
                                confirmation_response = InteractionResponseData(
                                    content_type="message", # Use 'message' content type
                                    data=confirmation_payload,
                                    user_model_state=ctx.user_model_state # Send final state
                                )
                                await safe_send_json(ws, confirmation_response.model_dump(mode='json'), "End Session Confirmation")
                                log.info(f"Sent end session confirmation to client for {session_id}")
                            else:
                                # Analysis already processing or done, inform user differently
                                log.warning(f"Session {session_id} analysis status is already '{current_status}'. Sending status update instead of triggering.")
                                status_payload = MessageResponse(
                                    response_type="message",
                                    text=f"Session analysis is already {current_status}."
                                )
                                status_response = InteractionResponseData(content_type="message", data=status_payload, user_model_state=ctx.user_model_state)
                                await safe_send_json(ws, status_response.model_dump(mode='json'), "End Session Status Update")
                                log.info(f"Sent end session status update ({current_status}) to client for {session_id}")

                        except Exception as trigger_err:
                            log.error(f"Failed to check status, trigger analysis, or send confirmation from 'end_session' for {session_id}: {trigger_err}", exc_info=True)
                            await send_error_response(ws, "Could not process session ending request.", "END_SESSION_FAIL", state=ctx.user_model_state if ctx else None)
                    else:
                        log.warning("Cannot trigger analysis or send confirmation: Context or user missing.")
                        # Optionally send an error back if connection is still open and ctx/user are missing unexpectedly
                        await send_error_response(ws, "Internal error: Cannot process end session request.", "END_SESSION_CONTEXT_MISSING", state=None)

                    # Close connection gracefully AFTER sending confirmation/error
                    log.info(f"Closing WebSocket connection for session {session_id} after handling 'end_session' request.")
                    await safe_close(ws, code=1000, reason="User ended session")
                    break # Exit the loop cleanly

                if not ctx:
                     log.error(f"WebSocket: Context became None before processing for session {session_id}")
                     await send_error_response(ws, "Internal server error: Session context lost.", "CONTEXT_LOST", state=ctx.user_model_state if ctx else None)
                     break

                # --- Lean Executor Logic --- #
                else:
                    # Append user message to history if provided
                    if user_input_text is not None:
                        ctx.history.append({"role": "user", "content": user_input_text})

                    await _run_executor_turn(ctx, ctx.current_focus_objective, ws)

                    # persist context
                    save_ok = False
                    try:
                        save_ok = await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                        log.info(f"WebSocket ({session_id}): Context saved successfully? {save_ok}")
                    except Exception as save_exc:
                        log.error(f"WebSocket ({session_id}): Error saving context in finally block: {save_exc}")
                    continue

            except WebSocketDisconnect as ws_disconnect:
                log.info(f"WebSocket ({session_id}): Client disconnected cleanly (code={ws_disconnect.code}, reason='{ws_disconnect.reason}').")
                break # Exit the loop
            except Exception as loop_err:
                log.error(f"WebSocket ({session_id}): Unhandled exception in main loop: {loop_err}\n{traceback.format_exc()}", exc_info=True)
                # Send a generic error if possible
                await send_error_response(ws, "An unexpected server error occurred.", "UNHANDLED_WS_LOOP_ERROR", details=str(loop_err), state=ctx.user_model_state if ctx else None)
                # Attempt to break cleanly, but the connection might already be broken
                break

    except WebSocketDisconnect as ws_disconnect_outer:
         log.info(f"WebSocket disconnected (outer loop/setup) for session_id={str(session_id)}: Code={ws_disconnect_outer.code}")
    except Exception as main_err:
         log.error(f"Unhandled exception in WebSocket handler for session {session_id}: {type(main_err).__name__}: {main_err}\n{traceback.format_exc()}", exc_info=True)
         # Use safe_send_json for the final error message attempt
         error_data = ErrorResponse(response_type="error", message="Internal server error encountered.")
         # Need to wrap error_data in InteractionResponseData structure if send_error_response isn't used
         full_error_response = InteractionResponseData(content_type="error", data=error_data, user_model_state=UserModelState()) # Provide default state
         await safe_send_json(ws, full_error_response.model_dump(mode='json'), "Main Error Send")
    finally:
        log.info(f"WebSocket ({session_id}): Entering finally block. Cleaning up.")
        # Task 2.2: Modify Disconnect Trigger
        if not session_ended_cleanly and ctx and user: # Check the flag!
            try:
                log.info(f"WebSocket ({session_id}): Saving final context state to DB before potential disconnect trigger.")
                # Ensure the context is saved before triggering analysis (attempt again)
                try:
                    await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                except Exception as save_exc:
                    log.error(f"WebSocket ({session_id}): Final save attempt failed: {save_exc}")

                # --- Check analysis_status BEFORE triggering ---
                supabase_client = await get_supabase_client()
                status_check = supabase_client.table("sessions") \
                    .select("analysis_status") \
                    .eq("id", str(session_id)) \
                    .maybe_single() \
                    .execute()

                current_status = status_check.data.get("analysis_status") if status_check.data else None
                log.info(f"WebSocket ({session_id}): Current analysis_status from DB on disconnect: '{current_status}'")

                if current_status is None: # Only trigger if analysis hasn't started/finished
                    log.info(f"Triggering background analysis due to unexpected disconnect for session {session_id}")
                    asyncio.create_task(queue_session_analysis(session_id, user.id, ctx.folder_id))
                    log.info(f"WebSocket ({session_id}): Background task scheduled on disconnect.")
                else:
                    log.info(f"Skipping analysis trigger on disconnect for session {session_id}: status is '{current_status}'.")
            except Exception as trigger_err:
                log.error(f"Failed to trigger background analysis on disconnect for {session_id}: {trigger_err}", exc_info=True)
            except Exception as final_save_err: # Catch potential save errors too
                log.error(f"WebSocket ({session_id}): Error saving context in finally block: {final_save_err}", exc_info=True)

        elif session_ended_cleanly:
            log.info(f"Skipping analysis trigger on disconnect: Session ended cleanly via 'end_session' event.")
        else:
            log.warning("Skipping analysis trigger on disconnect due to missing context or user.")

        # Make sure the socket is closed
        log.info(f"WebSocket ({session_id}): Ensuring WebSocket is closed in finally block.")
        if not session_ended_cleanly:
             await safe_close(ws) # Use the helper
        else:
             log.info(f"WebSocket ({session_id}): Connection already closed by 'end_session' handler. Skipping redundant close call in finally.")
        log.info(f"WebSocket ({session_id}): Finished finally block.")

# ===============================
# Lean Executor Helper Functions
# ===============================

def _build_lean_prompt(
    objective: "FocusObjective",
    user_model_state: UserModelState | None,
    last_action: Optional[str] | None = None,
) -> str:
    """Builds the system prompt for the lean executor turn.

    Args:
        objective: Current focus objective.
        user_model_state: Snapshot of the user's model.
        last_action: The last pedagogical action the tutor took (e.g. "explained", "asked").
    """
    user_state_json = "No user model available."
    if user_model_state:
        try:
            user_state_json = user_model_state.model_dump_json(indent=2)
        except Exception as e:
            log.error(f"_build_lean_prompt: Failed to serialise user model: {e}")
            user_state_json = str(user_model_state)

    return LEAN_EXECUTOR_PROMPT_TEMPLATE.format(
        objective_topic=getattr(objective, "topic", "N/A"),
        objective_goal=getattr(objective, "learning_goal", "N/A"),
        objective_threshold=getattr(objective, "target_mastery", 0.8),
        user_state_json=user_state_json,
        last_action_str=last_action or "None (start of objective)",
    )


async def _send_ws_error(ws: WebSocket, title: str, detail: str):
    """Send a structured error payload over the websocket."""
    try:
        err_payload = InteractionResponseData(
            content_type="error",
            data=ErrorResponse(error_message=title, technical_details=detail),
            user_model_state=UserModelState(),
        )
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json(err_payload.model_dump(mode="json"))
    except Exception as e:
        log.error(f"_send_ws_error: Failed to send error: {e}. Original: {title} - {detail}")


async def _dispatch_tool_call(call: ToolCall, ctx: TutorContext, ws: WebSocket):
    """Executes a ToolCall produced by the lean executor and streams the response."""
    log.info(f"_dispatch_tool_call: Handling tool '{call.name}' for session {ctx.session_id}")

    try:
        match call.name:
            case "explain":
                payload = ExplanationResponse(explanation_text=call.args.get("text", "..."))
                response = InteractionResponseData(
                    content_type="explanation",
                    data=payload,
                    user_model_state=ctx.user_model_state,
                )
                await ws.send_json(response.model_dump(mode="json"))
                ctx.last_pedagogical_action = "explained"
            case "ask_question":
                from ai_tutor.agents.models import QuizQuestion
                options: list[str] | None = call.args.get("options")
                q_text: str = call.args.get("question", "Missing question text")
                question = QuizQuestion(
                    question=q_text,
                    options=options or [],
                    correct_answer_index=call.args.get("correct_answer_index", -1),
                    explanation=call.args.get("explanation", ""),
                    difficulty=call.args.get("difficulty", "Medium"),
                    related_section=call.args.get("related_section", ""),
                )
                q_payload = QuestionResponse(
                    question_type="multiple_choice" if options else "free_response",
                    question_data=question,
                    context_summary=None,
                )
                await ws.send_json(
                    InteractionResponseData(
                        content_type="question",
                        data=q_payload,
                        user_model_state=ctx.user_model_state,
                    ).model_dump(mode="json")
                )
                ctx.last_pedagogical_action = "asked"
            case "draw":
                svg = call.args.get("svg", "[no svg]")
                msg = MessageResponse(
                    message_text=f"[Assistant wants to draw SVG: {svg[:60]}...]",
                    message_type="status_update",
                )
                await ws.send_json(
                    InteractionResponseData(
                        content_type="message",
                        data=msg,
                        user_model_state=ctx.user_model_state,
                    ).model_dump(mode="json")
                )
            case "reflect" | "summarise_context":
                # Internal – no FE output
                log.info(f"_dispatch_tool_call: Internal tool '{call.name}' executed (no FE output).")
            case "end_session":
                end_msg = MessageResponse(
                    message_text=f"Session ended: {call.args.get('reason', 'completed')}.",
                    message_type="summary",
                )
                await ws.send_json(
                    InteractionResponseData(
                        content_type="message",
                        data=end_msg,
                        user_model_state=ctx.user_model_state,
                    ).model_dump(mode="json")
                )
                # 'end_session' does not count as pedagogical action
            case _:
                await _send_ws_error(ws, "Unknown Action", f"Tool '{call.name}' is not recognised.")
    except WebSocketDisconnect:
        raise
    except Exception as e:
        log.exception(f"_dispatch_tool_call: Error executing '{call.name}': {e}")
        await _send_ws_error(ws, "Dispatch Error", str(e))


async def _run_executor_turn(ctx: TutorContext, objective: "FocusObjective", ws: WebSocket):
    """Single turn of the lean executor: build prompt → call LLM → dispatch."""
    llm = LLMClient()

    # Build prompt
    system_prompt = _build_lean_prompt(objective, ctx.user_model_state, ctx.last_pedagogical_action)

    # Prepare messages
    history = ctx.history or []
    messages = history + [{"role": "system", "content": system_prompt}]

    try:
        resp = await llm.chat(messages=messages, response_format={"type": "json_object"})

        # The wrapper may return either a raw content string (preferred) or a dict if
        # OpenAI returned JSON mode with no `content` field.
        if isinstance(resp, str):
            raw_json_str = resp
        elif isinstance(resp, dict):
            # Try to get "content" field first
            raw_json_str = resp.get("content") if isinstance(resp.get("content"), str) else None
            # Fallback: if the dict itself looks like a ToolCall already (has name & args)
            if raw_json_str is None and set(resp.keys()) >= {"name", "args"}:
                tool_call_data = resp
            elif raw_json_str is None:
                raise ValueError("LLM returned dict without 'content' or tool fields:")
        else:
            raise ValueError(f"Unexpected response type from LLMClient: {type(resp)}")

        # If we haven't parsed tool_call_data yet, parse from raw_json_str
        if 'tool_call_data' not in locals():
            if not raw_json_str:
                raise ValueError("LLM returned empty content")
            tool_call_data = json.loads(raw_json_str)

        call = ToolCall(**tool_call_data)

        # Append assistant message JSON to history for traceability
        history.append({"role": "assistant", "content": json.dumps(tool_call_data)})
        ctx.history = history

        await _dispatch_tool_call(call, ctx, ws)

    except (json.JSONDecodeError, ValidationError) as e:
        log.error(f"_run_executor_turn: LLM JSON parse/validation error: {e}")
        await _send_ws_error(ws, "Processing Error", str(e))
    except WebSocketDisconnect:
        raise
    except Exception as e:
        log.exception(f"_run_executor_turn: Unexpected error: {e}")
        await _send_ws_error(ws, "Internal Error", str(e))
# ===== End Lean Executor Helpers =====
