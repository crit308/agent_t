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
from ai_tutor.agents.executor_agent import run_executor
from ai_tutor.exceptions import ExecutorError
from agents.run_context import RunContextWrapper
from ai_tutor.interaction_logger import log_interaction
import asyncio
from ai_tutor.services.session_tasks import queue_session_analysis

router = APIRouter()

# Use shared session manager instance (same as other routers) – no stateful behaviour here
session_manager = SessionManager()

# Helper to authenticate a websocket connection and return the Supabase user
ALLOW_URL_TOKEN = os.getenv("ENV", "prod") != "prod"

log = logging.getLogger(__name__)

async def _authenticate_ws(ws: WebSocket, supabase: Client) -> Any:
    """Validate the `Authorization` header for a WebSocket connection.

    Returns the Supabase `User` object on success, otherwise raises and closes the socket.
    """
    # Try Authorization header first
    auth_header = ws.headers.get("authorization")  # headers keys are lower‑cased
    # Fallback: allow token via query param (for browser clients)
    if not auth_header and ALLOW_URL_TOKEN:
        token = ws.query_params.get("token")
        if token:
            auth_header = f"Bearer {token}"
    if not auth_header or not auth_header.startswith("Bearer "):
        # 1008 – policy violation (invalid or missing credentials)
        await ws.close(code=1008, reason="Missing or invalid Authorization header or token")
        raise RuntimeError("Unauthorized")
    # Extract JWT
    jwt = auth_header.split(" ", 1)[1]
    try:
        user_response = supabase.auth.get_user(jwt)
        user = user_response.user
        if user is None:
            await ws.close(code=1008, reason="Invalid or expired token")
            raise RuntimeError("Unauthorized")
        return user
    except Exception as exc:  # noqa: BLE001
        # Close the socket immediately; the caller should handle the RuntimeError
        await ws.close(code=1008, reason="Could not validate credentials")
        raise RuntimeError("Unauthorized") from exc

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
    # Prepare details as a dict if provided, otherwise None
    details_dict = {"raw_message": details} if details else None
    try:
        error_payload = ErrorResponse(
            response_type="error", # Ensure this mandatory field is set
            message=message, 
            error_code=error_code,
            details=details_dict # Pass dict or None
        )
        # Ensure user_model_state is valid or default
        state_obj = state if state else UserModelState() # Use provided or default empty state
        # Wrap the error payload in InteractionResponseData
        full_response = InteractionResponseData(
            content_type="error",
            data=error_payload,
            user_model_state=state_obj
        )
        # Send the wrapped response
        await safe_send_json(ws, full_response.model_dump(mode='json'), f"Error Response Send ({error_code})")
    except Exception as send_err:
        # Log error during error response creation/sending itself
        log.critical(f"Failed to create/send structured error response (Code={error_code}): {send_err}", exc_info=True)
        # Fallback: try sending a very simple error message if the structured one fails
        try:
            await safe_send_json(ws, {"content_type": "error", "data": {"message": "An internal error occurred while reporting an error."}}, "Fallback Error Send")
        except Exception as fallback_err:
            log.critical(f"Failed to send fallback error message: {fallback_err}")

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

                save_context_needed = False
                response_to_send: Optional[InteractionResponseData] = None
                status_from_executor: Optional[str] = None

                try:
                    log.info(f"WebSocket: Processing event type: {event_type}")

                    # --- Planner Logic --- #
                    if not planner_run_complete and (event_type == 'start' or event_type == 'user_message'):
                        log.info("WebSocket: First interaction or new message before objective, running Planner.")
                        try:
                            planner_objective = await determine_session_focus(ctx)
                            ctx.current_focus_objective = planner_objective
                            log.info("WebSocket: Planner successful, objective stored in context.")
                            save_context_needed = True # Save context after planner
                            planner_run_complete = True

                            # --- Send Initial Message (if it was a user message triggering the planner) ---
                            # If the planner was triggered by 'start', we might wait for the first user message
                            # If triggered by user message, we might respond immediately or let executor handle it
                            if event_type == 'user_message':
                                log.info("Planner ran due to user message. Letting Executor handle the first response.")
                                # Let the flow continue to the executor block below
                            elif event_type == 'start':
                                 # Maybe send a generic greeting after planner?
                                 initial_msg_payload = MessageResponse(
                                     response_type="message",
                                     text="Okay, I've analyzed the materials and have a focus in mind. Let me know when you're ready to begin!"
                                 )

                                 response_to_send = InteractionResponseData(
                                     content_type="message",
                                     data=initial_msg_payload,
                                     user_model_state=ctx.user_model_state,
                                 )
                                 status_from_executor = "awaiting_user_input"  # Set status explicitly
                                 log.info("WebSocket: Sending initial message after planner (start event).")
                                 # Response will be sent later

                        except Exception as planner_err:
                             log.error(f"WebSocket: Planner failed: {planner_err}", exc_info=True)
                             await send_error_response(ws, "Failed to plan the session.", "PLANNER_ERROR", details=f"Planner error: {type(planner_err).__name__}: {planner_err}", state=ctx.user_model_state)
                             continue # Skip to next message if planner fails

                    # --- Executor Logic --- # 
                    # This block runs if:
                    # 1. Planner just completed (and didn't set a response_to_send for 'start' event)
                    # 2. Planner was already complete and we received 'user_message', 'next', or 'answer'
                    if not response_to_send: # Only run executor if planner didn't already prepare a response
                        if not planner_run_complete:
                            log.warning(f"WebSocket: Executor logic reached but planner hasn't run. Event: {event_type}. This shouldn't happen if planner handles 'start'/'user_message' first.")
                            await send_error_response(ws, "Cannot process yet, session not initialized.", "STATE_ERROR", state=ctx.user_model_state if ctx else None)
                            continue
                        
                        # Check if there is a focus objective before calling executor
                        if not ctx.current_focus_objective:
                             log.warning(f"Executor called (event: {event_type}) but no focus objective found in context. Skipping.")
                             await send_error_response(ws, f"Cannot process '{event_type}': session goal not set.", "EXECUTOR_ERROR", details="No focus objective available.", state=ctx.user_model_state if ctx else None)
                             continue
                            
                        # All event types ('user_message', 'next', 'answer', potentially others) go to the executor
                        log.info(f"WebSocket: Invoking Executor for event type '{event_type}'")
                        try:
                            # --- FIX: Use await directly, not async for --- #
                            executor_result_data, status_from_executor = await run_executor(
                                ctx,
                                user_input_text, # Will be None for 'next', 'answer'
                                event_type,      # Pass the type explicitly
                                event_data       # Pass the data dict (contains answer_index if type is 'answer')
                            )
                            # --- END FIX --- #

                            log.info(f"WebSocket: Executor returned -> Response Type='{executor_result_data.content_type}', Status='{status_from_executor}'")
                            response_to_send = executor_result_data # Assign the InteractionResponseData part
                            save_context_needed = True # Assume executor run might change context

                            # --- Log Agent Response (moved here as we have the full response now) --- #
                            try:
                                log_content_agent = json.dumps(response_to_send.model_dump(mode='json'))
                                await log_interaction(ctx, "agent", log_content_agent, response_to_send.content_type, event_type)
                            except Exception as log_err:
                                log.error(f"WebSocket ({ctx.session_id}): Failed to log agent response: {log_err}", exc_info=True)
                            # --- End Log Agent Response --- #

                        except ExecutorError as executor_custom_err:
                             log.error(f"WebSocket: Executor failed with ExecutorError: {executor_custom_err}", exc_info=True)
                             current_state = ctx.user_model_state if ctx else UserModelState()
                             await send_error_response(ws, f"Failed to process your request ('{event_type}').", "EXECUTOR_ERROR", details=f"Executor error: {executor_custom_err}", state=current_state)
                             continue # Continue loop to wait for next message
                        except Exception as executor_err:
                             log.error(f"WebSocket: Executor failed with unexpected Exception: {executor_err}", exc_info=True)
                             current_state = ctx.user_model_state if ctx else UserModelState()
                             await send_error_response(ws, f"Failed to process your request ('{event_type}').", "EXECUTOR_ERROR", details=f"Unexpected Executor error: {type(executor_err).__name__}", state=current_state)
                             continue # Continue loop

                    # --- Save Context (Always save after successful processing leading to a response) --- #
                    # Note: Markdown says "Save context after EVERY executor turn". Let's ensure it runs if response_to_send exists.
                    if save_context_needed and response_to_send: # Ensure we only save if processing was potentially successful 
                        try:
                            log.info("WebSocket: Attempting to save context after processing.")
                            # Ensure we pass the *potentially modified* ctx from executor
                            success = await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                            if success:
                                log.info("WebSocket: Context saved successfully after processing.")
                            else:
                                log.warning("WebSocket: update_session_context returned False, context may not be saved.")
                        except Exception as save_err:
                             # Log error but don't break connection, try to send response anyway
                             log.error(f"WebSocket: Error saving context: {save_err}", exc_info=True)
                             # Maybe send an error response here? Or just log?
                             # Let's log for now, as the main response might still be useful.
                             # await send_error_response(ws, "Failed to save session progress.", "CONTEXT_SAVE_FAILED", details=f"Database error: {type(save_err).__name__}", state=ctx.user_model_state)

                    # --- Send Response & Handle Status --- #
                    # Moved agent logging to *after* successful execution
                    if response_to_send and status_from_executor:
                        log.info(f"WebSocket: Sending {response_to_send.content_type} response... Status: {status_from_executor}")
                        response_dict = response_to_send.model_dump(mode='json')
                        response_dict.pop('status', None) # Ensure status isn't duplicated in payload
                        await safe_send_json(ws, response_dict, f"{response_to_send.content_type.capitalize()} Response Send")

                        # Handle flow based on status from executor
                        log.info(f"WebSocket: Handling executor status: '{status_from_executor}'")
                        if status_from_executor == 'awaiting_user_input':
                            log.info(f"WebSocket: Status is '{status_from_executor}'. Waiting for next client message.")
                            continue # Continue loop normally
                        elif status_from_executor == 'objective_complete':
                            log.info(f"WebSocket: Status is '{status_from_executor}'. Objective complete. Clearing objective and resetting planner.")
                            # Clear objective and reset planner flag
                            ctx.current_focus_objective = None
                            planner_run_complete = False 
                            save_context_needed = True # Need to save the cleared objective
                            log.info("WebSocket: Cleared current_focus_objective and reset planner_run_complete flag. Will run Planner on next message.")
                            # Save the context *again* to persist the cleared objective
                            try:
                                log.info("WebSocket: Attempting to save context after objective completion.")
                                success = await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                                if success: log.info("WebSocket: Context saved successfully after objective completion.")
                                else: log.warning("WebSocket: Context save failed after objective completion.")
                            except Exception as save_err_complete:
                                log.error(f"WebSocket: Error saving context after objective completion: {save_err_complete}", exc_info=True)
                            continue # Wait for next user message to trigger planner
                        # --- Removed 'quiz_completed' special handling, assume it behaves like objective_complete or awaiting_user_input ---
                        # elif status_from_executor == 'quiz_completed': ...
                        else:
                            log.warning(f"WebSocket: Received unknown or unhandled status '{status_from_executor}' from executor. Defaulting to awaiting input.")
                            continue # Default behaviour
                    else:
                        # This case might happen if planner runs, sets save_context_needed, but decides not to send a message immediately.
                        # Or if an error occurred before response_to_send was set.
                        if not response_to_send and planner_run_complete:
                             log.warning("WebSocket: Reached end of processing block with no response to send (after executor expected). Ensure all paths (including errors) are handled.")
                        if not status_from_executor and response_to_send:
                             log.warning("WebSocket: Response generated but no status returned from executor. Defaulting to awaiting input.")
                             await safe_send_json(ws, response_to_send.model_dump(mode='json'), "Response Send (No Status)")
                             continue

                except Exception as e:
                    log.error(f"Unexpected error in main processing logic: {e}", exc_info=True)
                    await send_error_response(ws, "Internal server error during processing.", "INTERNAL_SERVER_ERROR", details=str(e), state=ctx.user_model_state)
                    continue # Attempt to recover by waiting for next message

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
                # Ensure the context is saved before triggering analysis
                await ctx.save(supabase)
                log.info(f"WebSocket ({session_id}): Context saved successfully.")

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
