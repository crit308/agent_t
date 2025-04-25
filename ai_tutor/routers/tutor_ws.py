from __future__ import annotations
from uuid import UUID
from typing import Any, Dict, Optional
from functools import lru_cache
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from supabase import Client
from postgrest.exceptions import APIError

from ai_tutor.dependencies import get_supabase_client, get_openai
from ai_tutor.session_manager import SessionManager
from ai_tutor.context import TutorContext, UserModelState
from ai_tutor.auth import verify_token  # Re‑use existing auth logic for header token verification
import json
from fastapi_utils.tasks import repeat_every
import structlog
from dataclasses import asdict, is_dataclass, fields
import logging
import traceback
from pydantic import ValidationError  # Import ValidationError
from ai_tutor.agents.models import QuizQuestion # Ensure QuizQuestion is imported
from starlette.websockets import WebSocketDisconnect, WebSocketState # Import WebSocketState
from ai_tutor.agents.tutor_agent_factory import build_tutor_agent
from agents import Runner  # Import Runner separately
from agents.stream_events import StreamEvent, RunItemStreamEvent, RawResponsesStreamEvent
from agents.items import MessageOutputItem, ToolCallOutputItem, HandoffOutputItem
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputText
from ai_tutor.api_models import (
    InteractionResponseData, ExplanationResponse, QuestionResponse,
    FeedbackResponse, MessageResponse, ErrorResponse, UserModelState # Ensure UserModelState is imported here
)
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem, FocusObjective
from fastapi import HTTPException

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

# --- Add Phase 1 Stub Functions --- #
async def run_planner_stub(ctx: TutorContext) -> FocusObjective:
    log.info("[Stub] Running Planner Stub")
    # Return a fixed, simple objective based on backend_change.md example
    objective = FocusObjective(
        topic="Introduction",
        learning_goal="Understand the purpose of this stub session",
        priority=5,
        target_mastery=0.8, # Required field
        # relevant_concepts=[], # Optional fields
        # suggested_approach="Explain basics"
    )
    log.info("[Stub] Planner Stub finished", objective=objective.model_dump())
    return objective

async def run_executor_stub(ctx: TutorContext, user_input: str) -> MessageResponse:
    log.info(f"[Stub] Running Executor Stub with input: '{user_input}'")
    # Return a fixed, simple message response based on backend_change.md example
    current_goal = ctx.current_focus_objective.learning_goal if ctx.current_focus_objective else 'None'
    response_text = f"Executor stub received: '{user_input}'. Session goal is '{current_goal}'"
    response = MessageResponse(
        response_type="message",
        text=response_text
    )
    log.info("[Stub] Executor Stub finished", response=response.model_dump())
    return response
# --- End Phase 1 Stub Functions --- #

@router.websocket("/ws/session/{session_id}")
async def tutor_stream(
    ws: WebSocket,
    session_id: UUID,
    supabase: Client = Depends(get_supabase_client),
):
    """Stream tutor interaction events for a session via WebSocket.

    The client must provide a valid Supabase JWT in the `Authorization` header (Bearer token).
    Each inbound JSON message is forwarded to the TutorFSM orchestrator. All streaming events
    emitted by the FSM are relayed back to the client in real‑time.
    """
    log.info(f"WebSocket attempting connection for session {session_id}")
    user = None
    ctx: Optional[TutorContext] = None  # Initialize ctx before try
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
            await safe_send_json(ws, {"type": "error", "detail": "Internal server error processing context."}, "Context Parse Error")
            await safe_close(ws, code=1011)
            return
        except Exception as ctx_err:
            log.error(f"WebSocket: Unexpected error initializing context for {session_id}: {ctx_err}\n{traceback.format_exc()}", exc_info=True)
            await safe_send_json(ws, {"type": "error", "detail": "Internal server error initializing context."}, "Context Init Error")
            await safe_close(ws, code=1011)
            return

        await ws.accept()
        log.info(f"WebSocket: Connection accepted for session {session_id}")

        if not ctx:
             log.error(f"WebSocket: CRITICAL - Context object is None after loading/initialization for session {session_id}.")
             await safe_send_json(ws, {"type": "error", "detail": "Internal server error: context unavailable."}, "Context Null Error")
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
        # Use a flag in the connection scope, not on ctx, as ctx might be reloaded
        planner_run_complete = False 
        # Check if context ALREADY has an objective from a previous connection to this session
        if ctx and ctx.current_focus_objective:
            log.info("Existing FocusObjective found in loaded context. Skipping Planner Stub.")
            planner_run_complete = True

        while True:
            try:
                payload_text = await ws.receive_text()
                log.debug(f"WebSocket: Received raw message for {session_id}: {payload_text}")
                payload = json.loads(payload_text)
                event_data = payload.get('data', {})

                # Determine event type robustly
                if 'type' in payload:
                    event_type = payload['type']
                elif 'event_type' in payload:
                    event_type = payload['event_type']
                    log.warning(f"[WebSocket Warning] Received message using deprecated 'event_type' for {session_id}: {payload_text}")
                else:
                    log.warning(f"[WebSocket Warning] Received message without 'type' or 'event_type' for {session_id}: {payload_text}")
                    await safe_send_json(ws, {"type": "error", "detail": "Message missing 'type' field."}, "Missing Type")
                    continue

                incoming: Dict[str, Any] = {'type': event_type, 'data': event_data}

                # Allow 'start' to initiate the agent run, similar to 'user_message'
                if incoming['type'] == 'ping' or incoming['type'] == 'system_tick':
                     log.debug(f"WebSocket: Received system message, ignoring for {session_id}.", message_type=incoming['type'])
                     continue # Skip processing ping/tick

                # Check if context is valid before proceeding
                if not ctx:
                     log.error(f"WebSocket: Context became None before processing for session {session_id}")
                     await safe_send_json(ws, {"type": "error", "detail": "Internal server error: Session context lost."}, "Context Lost Error")
                     break

                # Prepare the input for the agent
                # For 'start', provide a generic initial prompt or handle it specifically in the agent
                # For 'user_message', use the text provided
                agent_input = ""
                if incoming['type'] == 'start':
                    agent_input = "Start the lesson." # Or handle empty string if agent expects that
                    log.info(f"WebSocket: Initiating agent run for session {session_id} due to 'start' event.")
                elif incoming['type'] == 'user_message' and 'text' in incoming.get('data', {}):
                    agent_input = incoming["data"]["text"]
                    log.info(f"WebSocket: Invoking Agent for session {session_id} with user message.")
                else:
                     log.warning(f"WebSocket received unhandled message type '{incoming['type']}' or invalid 'user_message' data, skipping agent run.")
                     await safe_send_json(ws, {"type": "ack", "detail": f"Agent received unhandled type: {incoming['type']}"}, "Unhandled Type Ack")
                     continue

                # --- Agent/Stub Logic Invocation (Phase 1 Implementation) ---
                try:
                    log.info(f"WebSocket: Processing event type: {event_type}")
                    save_context_needed = False # Flag to indicate if context needs saving
                    response_to_send: Optional[InteractionResponseData] = None

                    # --- Planner Logic (Run on first message) ---
                    if not planner_run_complete:
                        log.info("WebSocket: First message received, running Planner Stub.")
                        try:
                            planner_objective = await run_planner_stub(ctx)
                            ctx.current_focus_objective = planner_objective # Store result in context
                            log.info("WebSocket: Planner Stub successful, objective stored in context.")
                            save_context_needed = True # Need to save the new objective
                            planner_run_complete = True # Mark planner as run for this connection

                            # Send confirmation message back to client
                            planner_confirm_msg = MessageResponse(response_type="message", text="Session ready. What would you like to work on first?")
                            response_to_send = InteractionResponseData(content_type="message", data=planner_confirm_msg, user_model_state=ctx.user_model_state)
                            log.info("WebSocket: Prepared planner confirmation message.")

                        except Exception as planner_err:
                             log.error(f"WebSocket: Planner Stub failed: {planner_err}", exc_info=True)
                             # Send specific error for planner failure
                             await send_error_response(
                                 ws,
                                 message="Failed to plan the session.",
                                 error_code="PLANNER_STUB_ERROR",
                                 details=f"Planner error: {type(planner_err).__name__}",
                                 state=ctx.user_model_state
                             )
                             # Continue or break? Let's continue but don't proceed to save/send response
                             continue

                    # --- Executor Logic (Run on subsequent user_message) ---
                    elif event_type == 'user_message' and agent_input:
                        log.info("WebSocket: Subsequent message received, running Executor Stub.")
                        if not ctx.current_focus_objective:
                             log.warning("Executor Stub called but no focus objective found in context. Skipping.")
                             await send_error_response(
                                 ws, "Cannot process message: session goal not set.", "EXECUTOR_STUB_ERROR",
                                 details="No focus objective available.", state=ctx.user_model_state
                             )
                             continue
                        try:
                            executor_result: MessageResponse = await run_executor_stub(ctx, agent_input)
                            # No explicit context modification in stub, but could happen in real agent
                            # save_context_needed = True # Uncomment if executor modifies context
                            log.info("WebSocket: Executor Stub successful.")

                            # Wrap executor response
                            response_to_send = InteractionResponseData(
                                content_type="message", # Per Coordination Point 3
                                data=executor_result,
                                user_model_state=ctx.user_model_state # Always include user model state
                            )
                            log.info("WebSocket: Prepared executor response message.")

                        except Exception as executor_err:
                             log.error(f"WebSocket: Executor Stub failed: {executor_err}", exc_info=True)
                             await send_error_response(
                                 ws,
                                 message="Failed to process your message.",
                                 error_code="EXECUTOR_STUB_ERROR",
                                 details=f"Executor error: {type(executor_err).__name__}",
                                 state=ctx.user_model_state
                             )
                             continue
                    else:
                        # Handle cases like receiving 'start' after planner already ran, or other unhandled types
                        log.warning(f"WebSocket: Received event '{event_type}' when not expected or not handled in stub logic.")
                        # Optionally send an ack or ignore
                        # await safe_send_json(ws, {"type": "ack", "detail": f"Event '{event_type}' received."}, "Event Ack")
                        continue # Skip saving/sending for unhandled cases in this state

                    # --- Save Context if Needed --- #
                    if save_context_needed:
                        try:
                            log.info("WebSocket: Attempting to save context after stub execution.")
                            success = await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                            if success:
                                log.info("WebSocket: Context saved successfully after stub execution.")
                            else:
                                log.warning("WebSocket: update_session_context returned False, context may not be saved.")
                        except HTTPException as save_exc:
                            log.error(f"WebSocket: Failed to save context after stub execution: Status={save_exc.status_code}, Detail={save_exc.detail}", exc_info=False)
                            await send_error_response(
                                ws, "Failed to save session progress.", "CONTEXT_SAVE_FAILED",
                                details=f"Database error: {save_exc.status_code}", state=ctx.user_model_state
                            )
                            # Continue after save failure, don't break the loop unless critical
                        except Exception as generic_save_err:
                             log.error(f"WebSocket: Unexpected error saving context after stub execution: {generic_save_err}", exc_info=True)
                             await send_error_response(
                                 ws, "An unexpected error occurred while saving session progress.", "INTERNAL_SERVER_ERROR",
                                 details=f"Save exception: {type(generic_save_err).__name__}", state=ctx.user_model_state
                             )

                    # --- Send Response --- #
                    if response_to_send:
                        log.info(f"WebSocket: Sending {response_to_send.content_type} response to client.")
                        await safe_send_json(ws, response_to_send.model_dump(mode='json'), f"{response_to_send.content_type.capitalize()} Response Send")
                    else:
                        # This shouldn't happen if logic is correct, but log if it does
                        log.warning("WebSocket: Reached end of try block with no response to send.")

                except Exception as e:
                    log.error(f"Unexpected error in stub logic: {e}", exc_info=True)
                    await send_error_response(
                        ws,
                        message="Internal server error during stub processing.",
                        error_code="INTERNAL_SERVER_ERROR",
                        details=str(e),
                        state=ctx.user_model_state
                    )
                    continue

            except WebSocketDisconnect as ws_disconnect:
                log.info(f"WebSocket disconnected by client during receive/processing for session {session_id}: Code={ws_disconnect.code}, Reason={ws_disconnect.reason}")
                break # Exit the receive loop

    except WebSocketDisconnect as ws_disconnect_outer:
         log.info(f"WebSocket disconnected (outer loop/setup) for session_id={str(session_id)}: Code={ws_disconnect_outer.code}")
    except Exception as main_err:
         log.error(f"Unhandled exception in WebSocket handler for session {session_id}: {type(main_err).__name__}: {main_err}\n{traceback.format_exc()}", exc_info=True)
         # Use safe_send_json for the final error message attempt
         error_data = ErrorResponse(response_type="error", message="Internal server error encountered.")
         await safe_send_json(ws, error_data.model_dump(mode='json'), "Main Error Send")
    finally:
        log.info(f"WebSocket: Entering finally block for session {session_id}")
        if ctx and hasattr(ctx, 'user_model_state') and hasattr(ctx.user_model_state, 'concepts'):
            try:
                 log.info(f"WebSocket: Processing final mastery updates for {session_id}")
                 mastery_prev = {} # Reconstruct prev mastery if needed, or load from ctx if stored
                 for topic, mastery_state in ctx.user_model_state.concepts.items():
                     prev = mastery_prev.get(topic)
                     new_mastery = mastery_state.mastery
                     new_confidence = mastery_state.confidence
                     if prev is None or abs(new_mastery - prev) >= 0.05:
                          log.debug(f"Sending mastery update for {topic}: {new_mastery}")
                          await safe_send_json(ws, {"type": "mastery_update", "topic": topic, "mastery": new_mastery, "confidence": new_confidence}, "Mastery Update Send")
            except Exception as mastery_err:
                 log.error(f"Error sending mastery updates for {session_id}: {mastery_err}", exc_info=True)
        else:
            log.warning(f"WebSocket: Skipping final mastery updates for session {session_id} because context (ctx) or concepts are not available.")
        log.info(f"WebSocket: Exiting handler for session {session_id}")
        await safe_close(ws, code=1000, reason="Handler finished") 
