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
from agents import Runner, set_event_handler, on_event

router = APIRouter()

# Use shared session manager instance (same as other routers) – no stateful behaviour here
session_manager = SessionManager()

# --- NEW: LLM Cost Tracking Hook ---
@on_event("llm_end")
async def _acc_cost(evt):
    # Cost calculation based on GPT-4o mini pricing ($0.15 / 1M input, $0.60 / 1M output as of July 2024)
    # Simplified to cents per 1K tokens
    cost_cents_input = evt.usage.prompt_tokens * 0.00015  # $0.00015 per token -> $0.15 / 1M tokens
    cost_cents_output = evt.usage.completion_tokens * 0.00060 # $0.0006 per token -> $0.60 / 1M tokens
    cost_cents = cost_cents_input + cost_cents_output
    ctx = evt.context              # TutorContext attached to RunContext
    if ctx:
        # Accumulate cost for the turn
        current_turn_cost = getattr(ctx, "_turn_cost_cents", 0)
        ctx._turn_cost_cents = current_turn_cost + cost_cents
        log.debug(f"Accumulated LLM cost for turn: {ctx._turn_cost_cents:.4f} cents (Event cost: {cost_cents:.4f}) Session: {ctx.session_id}")
    else:
        log.warning("LLM cost event received but no context found.")
# --- END NEW Hook ---

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
         try: await ws.close(code=1008)
         except: pass
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
                await ws.close(code=1011, reason="Internal server error fetching context.")
                return
        except Exception as db_err:
            log.error(f"WebSocket: Unexpected error fetching context from DB for {session_id}: {db_err}\n{traceback.format_exc()}", exc_info=True)
            await ws.close(code=1011, reason="Internal server error fetching context.")
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
            await ws.close(code=1011, reason="Internal server error processing context.")
            return
        except Exception as ctx_err:
            log.error(f"WebSocket: Unexpected error initializing context for {session_id}: {ctx_err}\n{traceback.format_exc()}", exc_info=True)
            await ws.close(code=1011, reason="Internal server error initializing context.")
            return

        await ws.accept()
        log.info(f"WebSocket: Connection accepted for session {session_id}")

        if not ctx:
             log.error(f"WebSocket: CRITICAL - Context object is None after loading/initialization for session {session_id}.")
             await ws.close(code=1011, reason="Internal server error: context unavailable.")
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
            await ws.send_json({
                 "content_type": "question",
                 "data": pending_question_payload,
                 "user_model_state": ctx.user_model_state.model_dump(mode='json')
            })
            log.info(f"WebSocket: Pending question sent for session {session_id}.")
        else:
             log.info(f"WebSocket: No pending question found in context for session {session_id}.")

        log.info(f"WebSocket: Entering main receive loop for session {session_id}")
        while True:
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
                await ws.send_json({"type": "error", "detail": "Message missing 'type' field."})
                continue

            incoming: Dict[str, Any] = {'type': event_type, 'data': event_data}

            if incoming['type'] in ['ping', 'system_tick']:
                log.debug(f"WebSocket: Received system message, ignoring for {session_id}.", message_type=incoming['type'])
                continue

            log.info(f"WebSocket: Processing event for session {session_id}. Event type: {incoming['type']}")
            if not ctx:
                 log.error(f"WebSocket: Context became None before processing for session {session_id}")
                 await ws.send_json({"type": "error", "detail": "Internal server error: Session context lost."})
                 break

            # --- Agent Logic ---
            if incoming['type'] != 'user_message' or 'text' not in incoming.get('data', {}):
                log.warning(f"WebSocket received non-text message type '{incoming['type']}', skipping agent run.")
                # Send an ack or appropriate response if needed?
                await ws.send_json({"type": "ack", "detail": f"Agent received: {incoming['type']}"})
                continue # Skip agent run for non-text messages

            try:
                log.info(f"WebSocket: Invoking Agent for session {session_id}")
                tutor_agent = build_tutor_agent()
                user_message = incoming["data"]["text"]

                # Define the context persistence hook
                async def on_tool_end(event):
                    tool_name = event.get('tool_name', 'unknown_tool')
                    # Persist context after any tool finishes execution
                    log.info(f"[on_tool_end hook] Tool '{tool_name}' finished for session {session_id}. Saving context.")
                    if ctx: # Ensure ctx is still valid
                        try:
                            await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                            log.info(f"[on_tool_end hook] Context saved successfully for session {session_id}.")
                        except Exception as save_err:
                            log.error(f"[on_tool_end hook] Failed to save context for session {session_id}: {save_err}\n{traceback.format_exc()}", exc_info=True)
                    else:
                         log.warning(f"[on_tool_end hook] Context object was None, skipping save for session {session_id}.")

                # Register the event handler for the runner
                set_event_handler("tool_end", on_tool_end)

                async for delta in Runner.run_streamed(
                    tutor_agent,
                    input=user_message, # Pass the user's text message
                    context=ctx,        # Pass the loaded TutorContext
                ):
                    # Ensure delta is serializable - Runner should yield dicts
                    if isinstance(delta, dict):
                        await ws.send_json(delta)
                    else:
                        log.warning(f"Agent stream yielded non-dict delta: {type(delta)}. Converting to string.")
                        await ws.send_json({"type": "message", "text": str(delta)}) # Fallback

                log.info(f"WebSocket: Agent stream finished for session {session_id}")

                # --- NEW: Send Cost Summary ---
                if ctx and hasattr(ctx, "_turn_cost_cents"):
                    turn_cost = getattr(ctx, "_turn_cost_cents", 0)
                    if turn_cost > 0:
                        log.info(f"Sending cost summary for turn: {turn_cost:.3f} cents. Session: {session_id}")
                        await ws.send_json({
                            "type": "cost_summary",
                            "cents": round(turn_cost, 3),
                        })
                        # Reset cost for the next turn
                        ctx._turn_cost_cents = 0
                    else:
                        log.debug(f"Skipping cost summary send, cost is zero. Session: {session_id}")
                # --- END NEW Cost Summary ---

            except Exception as agent_err:
                log.error(f"WebSocket: Agent error processing event for session {session_id}: {agent_err}\n{traceback.format_exc()}", exc_info=True)
                await ws.send_json({"type": "error", "detail": f"Internal tutor error. Please try again."})
                # Decide if we should break or continue listening
                # continue # Let's continue for now

    except WebSocketDisconnect:
         log.info(f"WebSocket disconnected by client (main loop or setup) for session_id={str(session_id)}")
    except Exception as main_err:
         log.error(f"Unhandled exception in WebSocket main processing for session {session_id}: {type(main_err).__name__}: {main_err}\n{traceback.format_exc()}", exc_info=True)
         try:
             await ws.send_json({"type": "error", "detail": "Internal server error encountered."})
             await ws.close(code=1011, reason="Internal server processing error.")
         except:
             pass
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
                          await ws.send_json({
                               "type": "mastery_update",
                               "topic": topic,
                               "mastery": new_mastery,
                               "confidence": new_confidence
                          })
            except Exception as mastery_err:
                 log.error(f"Error sending mastery updates for {session_id}: {mastery_err}", exc_info=True)
        else:
            log.warning(f"WebSocket: Skipping final mastery updates for session {session_id} because context (ctx) or concepts are not available.")
        log.info(f"WebSocket: Exiting handler for session {session_id}")
        try:
            if ws.client_state != WebSocketState.DISCONNECTED:
                 log.info(f"WebSocket: Attempting close in finally block for {session_id}. Current state: {ws.client_state}")
                 await ws.close()
                 log.info(f"WebSocket: Close call completed for {session_id}")
            else:
                 log.info(f"WebSocket: Socket already disconnected in finally block for {session_id}.")
        except RuntimeError as e:
             if 'Cannot call "send" once a close message has been sent' in str(e):
                  log.warning(f"WebSocket: Tried to close already closed connection for {session_id}.")
             else:
                  log.error(f"WebSocket: Error during WebSocket close for {session_id}: {e}", exc_info=True)
        except Exception as close_err:
            log.error(f"WebSocket: Generic error during WebSocket close for {session_id}: {close_err}", exc_info=True) 