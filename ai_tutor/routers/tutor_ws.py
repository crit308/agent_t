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
import uuid # Added import
from typing import Dict # Ensure Dict is imported, though it likely is already

from ai_tutor.services.session_tasks import queue_session_analysis

# Import prompt template (moved to ai_tutor.prompts)
from ai_tutor.prompts import LEAN_EXECUTOR_PROMPT_TEMPLATE
from ai_tutor.models.tool_calls import ToolCall
from ai_tutor.core.llm import LLMClient
from ai_tutor.utils.llm_utils import retry_on_json_error # Import the wrapper
from ai_tutor.exceptions import ToolInputError  # Import custom tool input error

router = APIRouter()

# Use shared session manager instance (same as other routers) – no stateful behaviour here
session_manager = SessionManager()

# Helper to authenticate a websocket connection and return the Supabase user
ALLOW_URL_TOKEN = os.getenv("ENV", "prod") != "prod"

log = logging.getLogger(__name__)

# Dictionary to hold pending futures for board state requests
_pending_board_state_requests: Dict[str, asyncio.Future] = {}

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
                     log.debug(f"WebSocket: Received system message, ignoring for {session_id}.") # Removed , message_type=event_type
                     continue

                # --- Handle whiteboard_mode update --- #
                if 'whiteboard_mode' in payload:
                    new_mode = payload.get('whiteboard_mode')
                    if new_mode in ['chat_only', 'chat_and_whiteboard']:
                        if ctx.interaction_mode != new_mode:
                            log.info(f"WebSocket ({session_id}): Updating interaction_mode from '{ctx.interaction_mode}' to '{new_mode}'.")
                            ctx.interaction_mode = new_mode
                            # Persist this change immediately
                            if user:
                                await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                                log.info(f"WebSocket ({session_id}): Persisted interaction_mode='{new_mode}'.")
                            else:
                                log.warning(f"WebSocket ({session_id}): Cannot persist interaction_mode, user object is missing.")
                    else:
                        log.warning(f"WebSocket ({session_id}): Received invalid whiteboard_mode '{new_mode}'. Ignoring.")
                # --- End whiteboard_mode update --- #

                # --- Handle BOARD_STATE_RESPONSE from client --- #
                if event_type == 'BOARD_STATE_RESPONSE':
                    request_id = payload.get('request_id')
                    board_data = payload.get('payload') # This is expected to be List[Dict[str, Any]]
                    if request_id and request_id in _pending_board_state_requests:
                        future = _pending_board_state_requests[request_id]
                        if not future.done():
                            log.info(f"WebSocket ({session_id}): Received BOARD_STATE_RESPONSE for request_id={request_id}. Setting future result.")
                            future.set_result(board_data)
                            # The skill itself will remove the future from the dict upon completion/timeout/error.
                        elif future.done():
                            log.warning(f"WebSocket ({session_id}): Received BOARD_STATE_RESPONSE for already completed future request_id={request_id}. Ignoring.")
                        else: # Should not happen if request_id is in keys
                            log.warning(f"WebSocket ({session_id}): Future not found for request_id={request_id} in BOARD_STATE_RESPONSE, though key was present. Strange.")
                    else:
                        log.warning(f"WebSocket ({session_id}): Received BOARD_STATE_RESPONSE with no/invalid request_id '{request_id}' or no pending requests. Ignoring.")
                    continue # Skip further processing for this message type
                # --- End BOARD_STATE_RESPONSE handling --- #

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
    ctx: TutorContext,
    objective: "FocusObjective",
    user_model_state: UserModelState | None,
    last_action: Optional[str] | None = None,
) -> str:
    """Builds the prompt for the lean executor LLM, incorporating context and tool list."""
    # Initialize UserModelState if None
    current_user_state = user_model_state if user_model_state is not None else UserModelState()
    user_state_str = current_user_state.model_dump_json(indent=2) # Pretty print for LLM

    # Format last action
    last_action_str = str(last_action) if last_action else "None"

    # --- Define session_summary ---
    session_summary_text = "No session summary notes available."
    if user_model_state and user_model_state.session_summary_notes:
        # Join the list of notes into a single string
        session_summary_text = "\n".join(f"- {note}" for note in user_model_state.session_summary_notes)
        if not session_summary_text.strip(): # Handle empty notes case
             session_summary_text = "Session summary notes are empty."
    # --- End Define session_summary ---

    # --- Define user_model_summary (using user_state_str for now as a placeholder) ---
    # This is the full JSON dump of the user model state.
    # Consider creating a more concise summary if LEAN_EXECUTOR_PROMPT_TEMPLATE requires it.
    user_model_summary_text = user_state_str
    # --- End Define user_model_summary ---

    prompt = LEAN_EXECUTOR_PROMPT_TEMPLATE.format(
        session_summary=session_summary_text, 
        user_model_summary=user_model_summary_text, 
        objective_topic=objective.topic,
        objective_goal=objective.learning_goal,
        objective_threshold=getattr(objective, "target_mastery", 0.8), 
        objective_priority=objective.priority,
        objective_relevant_concepts=", ".join(objective.relevant_concepts) if objective.relevant_concepts else "None",
        objective_suggested_approach=objective.suggested_approach or "None",
        last_action_str=last_action_str,
        interaction_mode=ctx.interaction_mode 
    )
    return prompt


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
    supabase_client = await get_supabase_client() # Ensure supabase client is available
    tool_result = None # Initialize tool_result

    # --- Direct front-end tools that don't require backend skill invocation ---
    if call.name in ["explain", "ask_question", "feedback", "message", "error", "end_session"]:
        # Handle these tool calls immediately and return
        try:
            if call.name == "explain":
                payload = ExplanationResponse(explanation_text=call.args.get("text", "..."))
                content_type = "explanation"
            elif call.name == "ask_question":
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
                payload = QuestionResponse(
                    question_type="multiple_choice" if options else "free_response",
                    question_data=question,
                    context_summary=None,
                )
                content_type = "question"
            elif call.name == "feedback":
                from ai_tutor.agents.models import QuizFeedbackItem
                feedback_item = QuizFeedbackItem(**call.args)
                payload = FeedbackResponse(feedback=feedback_item)
                content_type = "feedback"
            elif call.name == "message":
                payload = MessageResponse(**call.args)
                content_type = "message"
            elif call.name == "error":
                payload = ErrorResponse(**call.args)
                content_type = "error"
            elif call.name == "end_session":
                payload = MessageResponse(message_text=f"Session ended: {call.args.get('reason', 'completed')}.", message_type="summary")
                content_type = "message"
                # Optionally trigger session analysis etc.

            response = InteractionResponseData(
                content_type=content_type,
                data=payload,
                user_model_state=ctx.user_model_state,
            )
            await safe_send_json(ws, response.model_dump(mode="json"), f"Direct Dispatch {call.name}")

            # Update last pedagogical action quickly
            if call.name == "explain":
                ctx.last_pedagogical_action = "explained"
            elif call.name == "ask_question":
                ctx.last_pedagogical_action = "asked"

            return  # Important: skip further processing for these calls
        except Exception as direct_err:
            log.error(f"_dispatch_tool_call: Error handling direct tool '{call.name}': {direct_err}", exc_info=True)
            await send_error_response(ws, "Error preparing response.", "DIRECT_TOOL_ERROR", details=str(direct_err), state=ctx.user_model_state)
            return

    try:
        if call.name == "get_board_state":
            # --- Skill: Get Whiteboard State (Special Handling) ---
            request_id = str(uuid.uuid4())
            future = asyncio.Future()
            _pending_board_state_requests[request_id] = future
            log.debug(f"WS ({ctx.session_id if ctx else 'UnknownSession'}): Stored future for board state request {request_id}")

            try:
                await safe_send_json(ws, {"type": "REQUEST_BOARD_STATE", "request_id": request_id}, f"RequestBoardStateSend_{request_id}")
                log.debug(f"WS ({ctx.session_id if ctx else 'UnknownSession'}): Waiting for board state response for {request_id} (timeout 10s)")
                tool_result = await asyncio.wait_for(future, timeout=10.0) # Assign to tool_result
                log.debug(f"WS ({ctx.session_id if ctx else 'UnknownSession'}): Received board state response for {request_id}: {tool_result}")
            except asyncio.TimeoutError:
                log.error(f"WS ({ctx.session_id if ctx else 'UnknownSession'}): Timeout waiting for board state response for request {request_id}")
                await send_error_response(ws, "Timeout getting whiteboard state.", "BOARD_STATE_TIMEOUT", state=ctx.user_model_state if ctx else None)
                return # Exit _dispatch_tool_call early for this skill
            except Exception as wait_err:
                log.error(f"WS ({ctx.session_id if ctx else 'UnknownSession'}): Error waiting for board state future {request_id}: {wait_err}", exc_info=True)
                await send_error_response(ws, "Error getting whiteboard state.", "BOARD_STATE_WAIT_ERROR", state=ctx.user_model_state if ctx else None)
                return # Exit _dispatch_tool_call early for this skill
            finally:
                removed_future = _pending_board_state_requests.pop(request_id, None)
                if removed_future:
                    log.debug(f"WS ({ctx.session_id if ctx else 'UnknownSession'}): Removed future for board state request {request_id}")
                else:
                    log.warning(f"WS ({ctx.session_id if ctx else 'UnknownSession'}): Attempted to remove future for {request_id}, but it was not found (already removed or error before send).")
            # tool_result now holds the specs or was handled if an error occurred.
        
        else:
            # --- All Other Skills: Use invoke helper with correct signature ---
            from ai_tutor.skills import SKILL_REGISTRY  # Local import to avoid circular deps
            skill_obj = SKILL_REGISTRY.get(call.name)
            if skill_obj is None:
                log.error(f"_dispatch_tool_call: Skill '{call.name}' not found in registry.")
                await send_error_response(ws, f"Unknown tool '{call.name}'.", "UNKNOWN_TOOL", state=ctx.user_model_state if ctx else None)
                return

            # Call invoke properly: pass the skill object, context, and unpacked args
            tool_result = await invoke(skill_obj, ctx=ctx, **call.args)

    except ToolInputError as e: # This now covers errors from invoke() for other skills
        log.error(f"ToolInputError in skill '{call.name}' for session {ctx.session_id}: {e}. Args: {call.args}", exc_info=True)
        await send_error_response(
            ws,
            message="The tutor encountered an issue with its tool arguments.",
            error_code="TOOL_INPUT_VALIDATION_ERROR",
            details=str(e),
            state=ctx.user_model_state
        )
        if ctx.history is None: ctx.history = []
        system_error_message = f"System: The previous tool call to '{call.name}' failed due to invalid arguments: {e}. Args: {json.dumps(call.args)}. Please review the arguments and the tool's schema, then try the call again with corrected arguments."
        ctx.history.append({"role": "system", "content": system_error_message})
        return # Exit _dispatch_tool_call early

    except WebSocketDisconnect:
        log.warning(f"WebSocket disconnected during tool '{call.name}' for session {ctx.session_id}.")
        raise # Re-raise to be handled by the main WebSocket handler
    except Exception as e: # General errors from invoke() or other parts of the dispatch before post-processing
        log.exception(f"_dispatch_tool_call: Error executing tool '{call.name}' for session {ctx.session_id}: {e}")
        await send_error_response(
            ws,
            message=f"An unexpected error occurred while trying to use the '{call.name}' tool.",
            error_code="TOOL_EXECUTION_ERROR",
            details=str(e),
            state=ctx.user_model_state
        )
        if ctx.history is None: ctx.history = []
        system_error_message = f"System: The previous tool call to '{call.name}' encountered an unexpected error: {e}. Args: {json.dumps(call.args)}. You may need to try a different approach or tool."
        ctx.history.append({"role": "system", "content": system_error_message})
        return # Exit _dispatch_tool_call early

    # --- Post-invocation processing based on tool_result (if no exception occurred above) ---
    # This section is reached if get_board_state succeeded, or if invoke() for other skills succeeded.
    try:
        if call.name == "get_board_state":
            # tool_result already contains the board state from the special handling above
            log.info(f"Tool '{call.name}' (handled specially) returned: {tool_result} for session {ctx.session_id}")
            if ctx.history is None: ctx.history = []
            # Only append to history if tool_result is not None (i.e., no timeout/error occurred that was handled by returning early)
            if tool_result is not None:
                 ctx.history.append({
                    "role": "system", 
                    "content": f"Observed whiteboard state: {json.dumps(tool_result)}" 
                })
            # No direct WebSocket message to send back to client here, as the skill was for data retrieval for the LLM.

        # Existing tool dispatching logic for tools that send messages to the client:
        elif call.name == "explain":
            payload = ExplanationResponse(explanation_text=call.args.get("text", "..."))
            response = InteractionResponseData(
                content_type="explanation",
                data=payload,
                user_model_state=ctx.user_model_state,
            )
            await safe_send_json(ws, response.model_dump(mode="json"), "Explain Dispatch")
            ctx.last_pedagogical_action = "explained"
        elif call.name == "ask_question":
            # ... (rest of ask_question logic remains the same)
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
            await safe_send_json(ws,
                InteractionResponseData(
                    content_type="question",
                    data=q_payload,
                    user_model_state=ctx.user_model_state,
                ).model_dump(mode="json"),
                "Ask Question Dispatch"
            )
            ctx.last_pedagogical_action = "asked"
        elif call.name == "draw":
            # ... (draw logic remains the same)
            # The draw skill/tool itself should be sending actions to the whiteboard_manager now.
            # This part of _dispatch_tool_call might need to be removed if 'draw' is fully handled by invoke.
            # For now, assuming draw still sends a message like this or invoke returns something to send.
            # If `invoke` for `draw` sends the whiteboard actions and returns None/True, this part is not needed.
            # However, the user's original plan for `draw` was: Args: {{ "svg": "<svg>...</svg>" }}
            # This implies the LLM generates SVG directly, not a list of actions. This is an older model.
            # The new model is `ADD_OBJECTS` etc. This `draw` tool seems like a legacy item.
            # Let's assume `invoke` handles `draw` and we don't need special logic here unless `invoke` returns something to send.
            if tool_result: # If invoke for draw returned something to send (e.g. a confirmation or the actions themselves)
                # This depends on what `invoke` for `draw` returns.
                # If draw is now like other whiteboard actions, it might not return here.
                # For now, let's keep the old message for a direct SVG from LLM, if that's what `tool_result` is.
                if isinstance(tool_result, dict) and "svg" in tool_result: # Legacy SVG draw
                     svg = tool_result.get("svg", "[no svg]")
                     actions_to_send = [{
                         "type": "ADD_OBJECTS", 
                         "objects": [
                            {"kind": "svg_string", "svg_string": svg, "id": str(uuid.uuid4()) }
                         ]
                     }]
                     # This should go through the whiteboard_manager or a similar mechanism
                     # For now, sending a placeholder message
                     msg = MessageResponse(
                        message_text=f"[Assistant wants to draw SVG via legacy draw tool: {svg[:60]}...]",
                        message_type="status_update",
                        actions=actions_to_send # EXAMPLE: How actions might be bundled
                     )
                     await safe_send_json(ws,
                        InteractionResponseData(
                            content_type="message", # Or dedicated whiteboard_action type
                            data=msg,
                            user_model_state=ctx.user_model_state,
                        ).model_dump(mode="json"),
                        "Legacy Draw Dispatch"
                     )
                # Else, if `draw` is updated to use `WhiteboardAction`s and `invoke` sends them, this part might not be hit or `tool_result` is just True/None.

        elif call.name in ["reflect", "summarise_context"]:
            # Internal – no FE output from these tools themselves.
            # Result might be logged or stored in context by the skill via invoke.
            log.info(f"_dispatch_tool_call: Internal tool '{call.name}' executed. Result: {tool_result}")
        elif call.name == "end_session":
            # ... (end_session logic remains the same)
            end_msg = MessageResponse(
                message_text=f"Session ended: {call.args.get('reason', 'completed')}.",
                message_type="summary",
            )
            await safe_send_json(ws,
                InteractionResponseData(
                    content_type="message",
                    data=end_msg,
                    user_model_state=ctx.user_model_state,
                ).model_dump(mode="json"),
                "End Session Dispatch"
            )
        elif isinstance(tool_result, dict) and tool_result.get("type") and \
             call.name in ["group_objects", "move_group", "delete_group", "draw_latex"]:
            # Specific whitelist for tools that we know produce a single action dict
            log.info(f"Dispatching whiteboard action from tool '{call.name}': {tool_result}")
            response = InteractionResponseData(
                content_type="message",  # Use existing allowed type
                data=MessageResponse(
                    response_type="status_update",
                    message_text=f"Tutor performed whiteboard action: {tool_result.get('type')}"
                ),
                whiteboard_actions=[tool_result],
                user_model_state=ctx.user_model_state,
            )
            await safe_send_json(ws, response.model_dump(mode="json"), f"{call.name} Whiteboard Action Dispatch")
        elif isinstance(tool_result, dict) and tool_result.get("type") in [
            "ADD_OBJECTS", "DELETE_OBJECTS", "UPDATE_OBJECTS", "SET_BACKGROUND"
        ]:
            log.info(
                f"Generic whiteboard dispatch for tool '{call.name}' action type '{tool_result.get('type')}'."
            )
            response = InteractionResponseData(
                content_type="message",
                data=MessageResponse(
                    response_type="status_update",
                    message_text=f"Tutor performed whiteboard action: {tool_result.get('type')}"
                ),
                whiteboard_actions=[tool_result],
                user_model_state=ctx.user_model_state,
            )
            await safe_send_json(ws, response.model_dump(mode="json"), f"Whiteboard Action Dispatch ({call.name})")
        # All other tools are assumed to be handled by invoke and might send their own messages via ws if needed,
        # or return results that are then processed or ignored here.
        # If a tool sends its own WS messages (like draw_... tools via whiteboard_manager),
        # then `tool_result` might be None or True, and no further action is needed here.
        # If a tool returns data for the LLM, it should be added to history like `get_board_state`.

        # Fallback for unhandled tools or tools that returned something unexpected by this dispatcher logic
        # This part may need refinement based on how `invoke` and skills are structured.
        # For now, if a tool was called by `invoke` and didn't match above, we assume `invoke` handled its WS communication.

    except WebSocketDisconnect:
        raise
    except Exception as e:
        log.exception(f"_dispatch_tool_call: Error executing tool '{call.name}': {e}")
        await _send_ws_error(ws, "Dispatch Error", str(e))


async def _run_executor_turn(ctx: TutorContext, objective: "FocusObjective", ws: WebSocket):
    """Single turn of the lean executor: build prompt → call LLM → dispatch."""
    llm = LLMClient()

    # Build prompt
    system_prompt = _build_lean_prompt(ctx, objective, ctx.user_model_state, ctx.last_pedagogical_action)

    # Prepare messages
    history = ctx.history or []
    messages = history + [{"role": "system", "content": system_prompt}]
    
    # Default LLM kwargs (can be overridden or extended)
    llm_kwargs: dict[str, Any] = {}
    # Safely pull temperature from context settings if available
    settings_obj = getattr(ctx, "settings", None)
    if settings_obj is not None and getattr(settings_obj, "executor_temperature", None) is not None:
        llm_kwargs["temperature"] = settings_obj.executor_temperature
    else:
        llm_kwargs["temperature"] = 0.7  # sensible default
    # Add other default LLM parameters here if needed

    try:
        # ------------------------------------------------------------ #
        #  Diagnostic logging before making the LLM call
        # ------------------------------------------------------------ #
        log.info("_run_executor_turn: Calling Executor LLM (messages=%d, temp=%s)", len(messages), llm_kwargs.get("temperature"))
        log.debug("_run_executor_turn: First 2 messages preview: %s", [
            {"role": m.get("role"), "content": str(m.get("content"))[:120] + ("..." if len(str(m.get("content"))) > 120 else "")}
            for m in messages[:2]
        ])

        # Wrap the llm.chat call with the retry wrapper
        resp = await retry_on_json_error(
            llm.chat, 
            messages=messages, 
            response_format={"type": "json_object"}, 
            **llm_kwargs
        )

        log.info("_run_executor_turn: LLM call completed successfully.")

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
                # If no content and not a direct tool call, log and raise error
                log.error(f"_run_executor_turn: LLM returned dict without 'content' or direct tool call structure. Response: {resp}")
                raise ValueError("LLM returned dict without 'content' or tool fields.")
        else:
            log.error(f"_run_executor_turn: Unexpected response type from LLMClient: {type(resp)}. Response: {resp}")
            raise ValueError(f"Unexpected response type from LLMClient: {type(resp)}")

        # If we haven't parsed tool_call_data yet, parse from raw_json_str
        if 'tool_call_data' not in locals():
            if not raw_json_str:
                log.error("_run_executor_turn: LLM returned empty or null JSON string.")
                raise ValueError("LLM returned empty content")
            tool_call_data = json.loads(raw_json_str)

        # --- Pre-validation: ensure required keys exist ---
        if not isinstance(tool_call_data, dict) or "name" not in tool_call_data or "args" not in tool_call_data:
            log.error(f"_run_executor_turn: LLM output missing 'name' or 'args'. Response: {tool_call_data}")
            await _send_ws_error(ws, "LLM Format Error", "AI response missing required 'name' or 'args' keys. Prompt reminded the format; please try again.")
            # Give feedback to the LLM in the next turn via system message
            if ctx.history is None:
                ctx.history = []
            ctx.history.append({
                "role": "system",
                "content": "System: Your previous response was not formatted correctly. Please return exactly one JSON object with top-level keys 'name' and 'args'."
            })
            return  # Skip further processing for this turn

        call = ToolCall(**tool_call_data)

        # Append assistant message JSON to history for traceability
        # Ensure history is initialized if it's None
        if ctx.history is None:
            ctx.history = []
        ctx.history.append({"role": "assistant", "content": json.dumps(tool_call_data)}) # Storing the raw JSON for the LLM call

        await _dispatch_tool_call(call, ctx, ws)

    except (json.JSONDecodeError, ValidationError) as e:
        log.error(f"_run_executor_turn: LLM JSON parse/validation error after retries: {e}. System Prompt: {system_prompt[:500]}...", exc_info=True)
        await _send_ws_error(ws, "Processing Error", f"Failed to process the AI's response. Details: {type(e).__name__}")
    except WebSocketDisconnect:
        log.info("_run_executor_turn: WebSocket disconnected during executor turn.")
        raise # Re-raise to be handled by the main WebSocket loop
    except Exception as e:
        log.exception(f"_run_executor_turn: Unexpected error: {e}. System Prompt: {system_prompt[:500]}...", exc_info=True)
        await _send_ws_error(ws, "Internal Error", "An unexpected error occurred while processing your request.")
# ===== End Lean Executor Helpers =====

# --- NEW: Global LLMClient Instance --- #
# Consider initializing LLMClient once if it's stateless and thread-safe
# global_llm_client = LLMClient()
# Then use global_llm_client in _run_executor_turn if appropriate.
# For now, keeping it instantiated per call for simplicity, assuming it's lightweight.
# --- END NEW ---
