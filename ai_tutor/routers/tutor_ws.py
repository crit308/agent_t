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
from ai_tutor.skills.evaluate_quiz import evaluate_quiz
from ai_tutor.skills.update_user_model import update_user_model
from agents.run_context import RunContextWrapper

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
            user_model_state=state_obj, # Use the valid state object
            status="error" # Set status within InteractionResponseData
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
                event_data = payload.get('data', {})
                event_type = payload.get('type')

                if not event_type:
                    log.warning(f"[WebSocket Warning] Received message without 'type' for {session_id}: {payload_text}")
                    await send_error_response(ws, "Message missing 'type' field.", "INVALID_PAYLOAD", state=ctx.user_model_state if ctx else None)
                    continue

                if event_type == 'ping' or event_type == 'system_tick':
                     log.debug(f"WebSocket: Received system message, ignoring for {session_id}.", message_type=event_type)
                     continue

                if not ctx:
                     log.error(f"WebSocket: Context became None before processing for session {session_id}")
                     await send_error_response(ws, "Internal server error: Session context lost.", "CONTEXT_LOST", state=ctx.user_model_state if ctx else None)
                     break

                save_context_needed = False
                response_to_send: Optional[InteractionResponseData] = None
                wrapper = RunContextWrapper(ctx) # Create wrapper once per loop

                try:
                    log.info(f"WebSocket: Processing event type: {event_type}")

                    # --- Planner Logic --- #
                    if not planner_run_complete and (event_type == 'start' or event_type == 'user_message'):
                        log.info("WebSocket: First interaction, running Planner.")
                        try:
                            planner_objective = await determine_session_focus(ctx)
                            ctx.current_focus_objective = planner_objective
                            log.info("WebSocket: Planner successful, objective stored in context.")
                            save_context_needed = True
                            planner_run_complete = True

                            # --- Send Initial Message --- #
                            initial_msg_payload = MessageResponse(response_type="message", text="Okay, I've analyzed the materials and have a focus in mind. How can I help you get started?")
                            response_to_send = InteractionResponseData(
                                content_type="message",
                                data=initial_msg_payload,
                                user_model_state=ctx.user_model_state,
                                status="awaiting_user_input"
                            )
                            log.info("WebSocket: Sending initial message after planner.")
                            # (Response will be sent later in the loop)
                            # --- End Initial Message --- #

                        except Exception as planner_err:
                             log.error(f"WebSocket: Planner failed: {planner_err}", exc_info=True)
                             await send_error_response(ws, "Failed to plan the session.", "PLANNER_ERROR", details=f"Planner error: {type(planner_err).__name__}: {planner_err}", state=ctx.user_model_state)
                             continue # Skip to next message if planner fails

                    # --- Direct Answer Evaluation Logic --- #
                    if event_type == 'answer':
                        log.info("WebSocket: Answer event received, evaluating directly.")
                        answer_index = event_data.get('answer_index')
                        if answer_index is None or not isinstance(answer_index, int):
                            log.warning(f"WebSocket: Invalid or missing 'answer_index' in answer payload: {event_data}")
                            await send_error_response(ws, "Invalid answer payload.", "INVALID_PAYLOAD", details="Missing or invalid 'answer_index'.", state=ctx.user_model_state)
                            continue
                        
                        if not ctx.current_focus_objective:
                            log.warning("Cannot evaluate answer: current_focus_objective is missing.")
                            await send_error_response(ws, "Cannot evaluate answer: session goal not set.", "EVALUATION_ERROR", details="No focus objective available.", state=ctx.user_model_state)
                            continue
                            
                        try:
                            # 1. Evaluate the quiz answer
                            feedback_item: QuizFeedbackItem = await evaluate_quiz(wrapper, answer_index)
                            log.info("WebSocket: evaluate_quiz skill successful.")
                            
                            # 2. Update the user model based on feedback
                            outcome = 'correct' if feedback_item.is_correct else 'incorrect'
                            # Use topic from focus objective as primary source
                            topic = ctx.current_focus_objective.topic 
                            if not topic:
                                # Fallback to topic from question if needed (less ideal)
                                topic = getattr(feedback_item, 'related_section', None) or "Unknown Topic"
                                log.warning(f"FocusObjective missing topic, falling back to topic: {topic}")
                            
                            update_msg = await update_user_model(wrapper, topic=topic, outcome=outcome, details=feedback_item.improvement_suggestion)
                            log.info(f"WebSocket: update_user_model skill successful: {update_msg}")
                            save_context_needed = True # Context was modified by skills

                            # 3. Prepare response
                            response_to_send = InteractionResponseData(
                                content_type="feedback",
                                data=feedback_item, # Send the feedback item as data
                                user_model_state=ctx.user_model_state, # Send the *updated* state
                                status="awaiting_user_input" # Always wait after feedback
                            )
                            log.info("WebSocket: Prepared feedback response message.")

                        except Exception as skill_err:
                             log.error(f"WebSocket: Error during answer evaluation skills: {skill_err}", exc_info=True)
                             await send_error_response(ws, "Failed to evaluate your answer.", "EVALUATION_ERROR", details=f"Evaluation error: {type(skill_err).__name__}: {skill_err}", state=ctx.user_model_state)
                             continue

                    # --- Executor Logic (User Message or Start AFTER planner) --- #
                    elif event_type == 'user_message' or event_type == 'start':
                        # This block now runs only *after* the planner has run (planner_run_complete is True)
                        # Or if it's the first message that *also* ran the planner above.
                        # If response_to_send is already set (by planner), skip executor for this turn.
                        if response_to_send:
                             log.info(f"WebSocket: Skipping executor as response already prepared (likely initial message).")
                        elif not planner_run_complete:
                             log.warning(f"WebSocket: Executor called with type '{event_type}' but planner hasn't run. This shouldn't happen.")
                             await send_error_response(ws, "Cannot process yet, session not initialized.", "STATE_ERROR", state=ctx.user_model_state)
                             continue
                        else:
                            user_input_text = event_data.get('text') if event_type == 'user_message' else None # Use None if 'start' after planner
                            log.info(f"WebSocket: '{event_type}' event received (post-planner), running Executor.")
                            if not ctx.current_focus_objective:
                                 log.warning("Executor called but no focus objective found in context. Skipping.")
                                 await send_error_response(ws, "Cannot process message: session goal not set.", "EXECUTOR_ERROR", details="No focus objective available.", state=ctx.user_model_state)
                                 continue
                            try:
                                executor_result = await run_executor(ctx, user_input=user_input_text)
                                log.info(f"WebSocket: Executor result content_type: {executor_result.content_type}, data type: {type(executor_result.data).__name__}")
                                # Assume executor modifies context internally (e.g., via update_user_model call by LLM)
                                # and returns the *final* state in InteractionResponseData
                                save_context_needed = True # Save context after executor runs
                                log.info("WebSocket: Executor successful.")
                                response_to_send = executor_result
                                log.info("WebSocket: Prepared executor response message.")

                            # Catch the specific ExecutorError
                            except ExecutorError as executor_custom_err:
                                 log.error(f"WebSocket: Executor failed with ExecutorError: {executor_custom_err}", exc_info=True)
                                 await send_error_response(ws, "Failed to process your message.", "EXECUTOR_ERROR", details=f"Executor error: {executor_custom_err}", state=ctx.user_model_state)
                                 continue
                            except Exception as executor_err:
                                 # Catch any other unexpected errors from executor
                                 log.error(f"WebSocket: Executor failed with unexpected Exception: {executor_err}", exc_info=True)
                                 await send_error_response(ws, "Failed to process your message.", "EXECUTOR_ERROR", details=f"Unexpected Executor error: {type(executor_err).__name__}", state=ctx.user_model_state)
                                 continue
                    
                    # --- ADDED: Executor Logic for 'next' event ---
                    elif event_type == 'next':
                        log.info(f"WebSocket: Invoking Executor for session {session_id} due to 'next' event.")
                        if not planner_run_complete:
                             log.warning(f"WebSocket: Executor called with type '{event_type}' but planner hasn't run. This shouldn't happen.")
                             await send_error_response(ws, "Cannot process yet, session not initialized.", "STATE_ERROR", state=ctx.user_model_state)
                             continue
                        if not ctx.current_focus_objective:
                             log.warning("Executor called for 'next' but no focus objective found in context. Skipping.")
                             await send_error_response(ws, "Cannot process next step: session goal not set.", "EXECUTOR_ERROR", details="No focus objective available for 'next'.", state=ctx.user_model_state)
                             continue
                        try:
                            # Call executor with placeholder input for 'next'
                            executor_result = await run_executor(ctx, user_input="[NEXT]") 
                            # Log the result (Already added in previous step, ensure it's detailed enough)
                            log.info(f"tutor_ws after executor ('next'): Result content_type='{executor_result.content_type}', Data type='{type(executor_result.data).__name__}'")
                            save_context_needed = True # Assume context might be modified
                            log.info("WebSocket: Executor successful for 'next' event.")
                            response_to_send = executor_result
                            log.info("WebSocket: Prepared executor response message for 'next'.")

                        except ExecutorError as executor_custom_err:
                             log.error(f"WebSocket: Executor failed for 'next' with ExecutorError: {executor_custom_err}", exc_info=True)
                             await send_error_response(ws, "Failed to process the next step.", "EXECUTOR_ERROR", details=f"Executor error on 'next': {executor_custom_err}", state=ctx.user_model_state)
                             continue # Skip to next message
                        except Exception as e:
                             log.error(f"WebSocket: Unexpected error processing 'next' for session {session_id}: {e}", exc_info=True)
                             await send_error_response(ws, "Internal server error processing next step.", "INTERNAL_SERVER_ERROR", details=f"Unexpected error on 'next': {type(e).__name__}", state=ctx.user_model_state)
                             continue # Skip to next message

                    else:
                        log.warning(f"WebSocket: Received event '{event_type}' when not expected or not handled.")
                        # Optional: Send error for unhandled types
                        # await send_error_response(ws, f"Unknown command type: {event_type}", "UNKNOWN_COMMAND", state=ctx.user_model_state)
                        continue

                    # --- Save Context --- #
                    if save_context_needed:
                        try:
                            log.info("WebSocket: Attempting to save context after processing.")
                            success = await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                            if success:
                                log.info("WebSocket: Context saved successfully after processing.")
                            else:
                                log.warning("WebSocket: update_session_context returned False, context may not be saved.")
                        except Exception as save_err:
                             # Log error but don't break connection, try to send response anyway
                             log.error(f"WebSocket: Error saving context: {save_err}", exc_info=True)
                             await send_error_response(ws, "Failed to save session progress.", "CONTEXT_SAVE_FAILED", details=f"Database error: {type(save_err).__name__}", state=ctx.user_model_state)

                    # --- Send Response & Handle Status --- #
                    if response_to_send:
                        # Log before sending
                        log.info(f"tutor_ws: Attempting to send {response_to_send.content_type} to client...")
                        log.info(f"WebSocket: Sending {response_to_send.content_type} response to client.")
                        await safe_send_json(ws, response_to_send.model_dump(mode='json'), f"{response_to_send.content_type.capitalize()} Response Send")
                        status = getattr(response_to_send, 'status', None)
                        # For Phase 2, all successful statuses lead to waiting for the user
                        if status == "awaiting_user_input" or status == "objective_complete" or status == "explanation_delivered":
                            log.info(f"WebSocket: Status is '{status}'. Waiting for next user message.")
                        elif status == "error":
                            log.error("WebSocket: Status is 'error'. Error payload was sent.")
                        continue # Continue loop after sending response
                    else:
                        log.warning("WebSocket: Reached end of processing block with no response to send.")

                except Exception as e:
                    log.error(f"Unexpected error in main processing logic: {e}", exc_info=True)
                    await send_error_response(ws, "Internal server error during processing.", "INTERNAL_SERVER_ERROR", details=str(e), state=ctx.user_model_state)
                    continue # Attempt to recover by waiting for next message

            except WebSocketDisconnect as ws_disconnect:
                log.info(f"WebSocket disconnected by client during receive/processing for session {session_id}: Code={ws_disconnect.code}, Reason={ws_disconnect.reason}")
                break

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
