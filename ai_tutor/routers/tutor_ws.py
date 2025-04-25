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
    FeedbackResponse, MessageResponse, ErrorResponse
)
from ai_tutor.agents.models import QuizQuestion, QuizFeedbackItem

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

        log.info(f"WebSocket: Entering main receive loop for session {session_id}")
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

                # --- Agent Logic ---
                try:
                    log.info(f"WebSocket: Invoking Agent for session {session_id} with input: {agent_input[:50]}...")
                    tutor_agent = build_tutor_agent()
                    # Ensure agent_input is defined based on 'start' or 'user_message'
                    agent_input = ""
                    if incoming['type'] == 'start':
                        agent_input = "Start the lesson."
                    elif incoming['type'] == 'user_message' and 'text' in incoming.get('data', {}):
                        agent_input = incoming["data"]["text"]
                    else: # Should have been caught earlier, but safeguard
                        log.error(f"Invalid agent trigger: {incoming['type']}")
                        raise ValueError("Invalid trigger for agent run")

                    run_result_stream = Runner.run_streamed(
                        tutor_agent,
                        input=agent_input,
                        context=ctx,
                    )

                    final_response_sent = False # Flag to ensure we only send one final payload
                    async for stream_event in run_result_stream.stream_events():
                        log.debug(f"Processing stream event type: {getattr(stream_event, 'type', 'Unknown')}")

                        response_payload = None
                        content_type = None

                        # --- Handle Raw Text Deltas ---
                        if isinstance(stream_event, RawResponsesStreamEvent):
                             raw_data = stream_event.data
                             if hasattr(raw_data, 'type') and raw_data.type == 'response.output_text.delta':
                                  delta_text = getattr(raw_data, 'delta', '')
                                  if delta_text:
                                       # Send raw delta for immediate display
                                       raw_delta_payload = {
                                           "type": "raw_delta", # Custom type for frontend hook
                                           "delta": delta_text
                                       }
                                       await safe_send_json(ws, raw_delta_payload, "Delta Send")
                                       continue # Don't process further if it's just a delta

                        # --- Handle Final Item Completions ---
                        elif isinstance(stream_event, RunItemStreamEvent) and stream_event.item:
                            item = stream_event.item
                            if stream_event.name == 'message_output_created' and isinstance(item, MessageOutputItem):
                                # --- FIX: Correctly access text content ---
                                final_text = ""
                                # The raw_item usually holds the OpenAI response structure
                                if item.raw_item and hasattr(item.raw_item, 'content') and item.raw_item.content:
                                    # Content is often a list, iterate through parts
                                    for part in item.raw_item.content:
                                         # Check if the part is a text part
                                         if isinstance(part, ResponseOutputText) and hasattr(part, 'text'):
                                              final_text += part.text
                                # --- End FIX ---

                                if final_text:
                                    content_type = "message"
                                    response_payload = MessageResponse(
                                         response_type="message",
                                         text=final_text.strip() # Strip potential whitespace
                                     )
                                    log.info(f"Detected completed MessageOutputItem: {final_text[:50]}...")
                                else:
                                    log.warning("MessageOutputItem created but no text content found.")

                            elif stream_event.name == 'tool_call_output_created' and isinstance(item, ToolCallOutputItem):
                                tool_name = item.tool_name
                                tool_output = item.output
                                log.info(f"Detected completed ToolCallOutputItem for tool: {tool_name}")

                                if tool_name == 'create_quiz' and isinstance(tool_output, QuizQuestion):
                                     content_type = "question"
                                     response_payload = QuestionResponse(
                                         response_type="question", question=tool_output,
                                         topic=ctx.current_teaching_topic or "Unknown"
                                     )
                                     ctx.current_quiz_question = tool_output
                                     ctx.user_model_state.pending_interaction_type = 'checking_question'

                                elif tool_name == 'evaluate_quiz' and isinstance(tool_output, QuizFeedbackItem):
                                     content_type = "feedback"
                                     response_payload = FeedbackResponse(
                                         response_type="feedback", feedback=tool_output,
                                         topic=ctx.current_teaching_topic or "Unknown"
                                     )
                                     ctx.current_quiz_question = None
                                     ctx.user_model_state.pending_interaction_type = None

                                else:
                                     log.warning(f"Unhandled completed tool output for {tool_name}")

                        # --- Send Final Structured Payload ---
                        if response_payload and content_type and not final_response_sent:
                            api_response = InteractionResponseData(
                                content_type=content_type,
                                data=response_payload,
                                user_model_state=ctx.user_model_state
                            )
                            await safe_send_json(ws, api_response.model_dump(mode='json'), f"Final Payload Send ({content_type})")
                            log.info(f"Sent final formatted {content_type} response to client.")
                            final_response_sent = True
                            # break # Uncomment if you want to stop after first final payload

                    # --- Handle case where stream finishes without sending a final payload ---
                    if not final_response_sent:
                        log.warning(f"Agent stream for session {session_id} finished without sending a structured response.")
                        fallback_payload = MessageResponse(response_type="message", text="Finished processing.")
                        fallback_response = InteractionResponseData(content_type="message", data=fallback_payload, user_model_state=ctx.user_model_state)
                        await safe_send_json(ws, fallback_response.model_dump(mode='json'), "Fallback Send")

                    log.info(f"WebSocket: Agent stream processing finished for session {session_id}")

                    # --- Persist context AFTER the agent turn completes ---
                    if ctx:
                        try:
                            log.info(f"WebSocket: Updating context in DB after agent run for {session_id}")
                            await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                            log.info(f"WebSocket: Context updated successfully in DB for {session_id}")
                        except Exception as save_err:
                            log.error(f"WebSocket: Failed to save context after agent run for session {session_id}: {save_err}\n{traceback.format_exc()}", exc_info=True)
                    else:
                        log.warning(f"WebSocket: Context object was None after agent run, cannot save state for session {session_id}")

                except Exception as agent_err:
                     log.error(f"WebSocket: Agent error processing event for session {session_id}: {agent_err}\n{traceback.format_exc()}", exc_info=True)
                     error_data = ErrorResponse(response_type="error", message="Internal tutor error. Please try again.")
                     await safe_send_json(ws, error_data.model_dump(mode='json'), "Agent Error Send")

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