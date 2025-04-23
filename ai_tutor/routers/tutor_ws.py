from __future__ import annotations
from uuid import UUID
from typing import Any, Dict
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

router = APIRouter()

# Use shared session manager instance (same as other routers) – no stateful behaviour here
session_manager = SessionManager()

# Helper to authenticate a websocket connection and return the Supabase user
ALLOW_URL_TOKEN = os.getenv("ENV", "prod") != "prod"

log = structlog.get_logger(__name__)

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
    # Authenticate (before accept if you prefer strict policy). We accept only after successful auth.
    try:
        user = await _authenticate_ws(ws, supabase)
    except RuntimeError:
        return  # Socket already closed in helper

    # --- Hydrate context & last_event from DB (D-2) ---
    try:
        row = (
            supabase.table("sessions")
            .select("context_json", "last_event_json", "current_question_json")
            .eq("id", str(session_id))
            .eq("user_id", str(user.id))
            .maybe_single()
            .execute()
        ).data
    except APIError as e:
        if e.code == "204":   # no content – first connection for this session
            row = None
        else:
            raise

    # Import TutorFSM here, inside the function, before it's used
    from ai_tutor.fsm import TutorFSM

    if row and row["context_json"]:
        ctx = TutorContext.model_validate_json(row["context_json"])
        last_event = json.loads(row["last_event_json"])
        current_question_json = row.get("current_question_json")
    else:
        ctx = TutorContext(session_id=session_id, user_id=user.id)
        last_event = None
        current_question_json = None

    # Accept the connection
    await ws.accept()

    # Resume mid-objective if reconnecting and we were awaiting user input
    if ctx.state == 'awaiting_user' and last_event:
        fsm_resume = TutorFSM(ctx)
        resume_result = await fsm_resume.on_user_message(last_event)
        # Send resumed result to client
        if hasattr(resume_result, 'model_dump'):
            await ws.send_json(resume_result.model_dump(mode='json'))
        elif isinstance(resume_result, dict):
            await ws.send_json(resume_result)
        else:
            await ws.send_json({'response': resume_result})
        # Persist resumed context
        await session_manager.update_session_context(supabase, session_id, user.id, ctx)

    # --- Send current_question_json if present (for race-free resume) ---
    if current_question_json:
        await ws.send_json({"type": "run_item_stream_event", "item": current_question_json})

    # --- Resume pending question if needed (D-3) ---
    if ctx.user_model_state.pending_interaction_type == "checking_question":
        # Send the cached QuestionOutputItem over WS before continuing
        if last_event and last_event.get("event_type") == "system_question":
            await ws.send_json(last_event["data"])  # Assumes QuestionOutputItem is in last_event['data']

    # Heartbeat: send system_tick every 60 seconds to keep connection alive
    @repeat_every(seconds=60)
    async def heartbeat():
        await ws.send_json({"event_type": "system_tick"})

    # Snapshot mastery values to detect later updates
    mastery_prev = {topic: state.mastery for topic, state in ctx.user_model_state.concepts.items()}

    tutor_ctx = ctx  # Use hydrated context for rest of handler

    try:
        while True:
            try:
                # Read raw payload text
                payload_text = await ws.receive_text()
                payload = json.loads(payload_text)
                event_data = payload.get('data', {}) # Extract data first

                # Determine the event type, prioritizing 'type' over 'event_type'
                if 'type' in payload:
                    event_type = payload['type']
                elif 'event_type' in payload:
                    event_type = payload['event_type'] # Use event_type if type is missing
                    print(f"[WebSocket Warning] Received message using deprecated 'event_type': {payload_text}")
                else:
                    # Handle messages without type/event_type
                    print(f"[WebSocket Warning] Received message without 'type' or 'event_type': {payload_text}")
                    await ws.send_json({"type": "error", "detail": "Message missing 'type' field."})
                    continue # Skip processing this message

                # Construct the 'incoming' dict for the FSM with the correct 'type' key
                incoming: Dict[str, Any] = {'type': event_type, 'data': event_data}

                # --- Check for internal/system message types BEFORE handling specific handlers or passing to FSM ---
                if incoming['type'] in ['ping', 'system_tick']:
                    log.debug("Received system message, ignoring.", message_type=incoming['type'])
                    continue # Ignore pings and system ticks, don't pass to FSM

                # --- Handle specific types like 'pace_change', 'help_request' --- 
                # Check incoming['type'] now
                if incoming['type'] == "pace_change":
                    value = incoming['data'].get("value") # Get value from data field
                    try:
                        factor = float(value)
                        tutor_ctx.user_model_state.learning_pace_factor = factor
                        await session_manager.update_session_context(supabase, session_id, user.id, tutor_ctx)
                        # Acknowledge to client
                        await ws.send_json({"type": "pace_change", "value": factor})
                    except Exception as e:
                        await ws.send_json({"type": "error", "detail": f"Invalid pace_change value: {e}"})
                    continue # Go to next WebSocket message
                
                if incoming['type'] == "help_request":
                    mode = incoming['data'].get("mode") # Get mode from data field
                    if mode == "stuck":
                        tutor_ctx.user_model_state.pending_interaction_type = "summary_prompt"
                        tutor_ctx.user_model_state.pending_interaction_details = {"reason": "stuck"}
                        # Log help_request for analytics
                        try:
                            supabase.table("actions").insert({
                                "session_id": str(session_id),
                                "user_id": str(user.id),
                                "action_type": "help_request",
                                "action_details": {"mode": mode}
                            }).execute()
                        except Exception as e:
                            log.warning("ws_help_request_log_failed", error=str(e))
                        await session_manager.update_session_context(supabase, session_id, user.id, tutor_ctx)
                        # Acknowledge reception
                        await ws.send_json({"type": "help_request_ack"})
                    else:
                        await ws.send_json({"type": "error", "detail": "Invalid help_request mode"})
                    continue # Go to next WebSocket message
                # --- End specific type handling ---

            except WebSocketDisconnect:  # client closed connection
                log.info("WebSocket disconnected by client", session_id=str(session_id))
                break
            except json.JSONDecodeError:
                log.warning("WebSocket received invalid JSON", session_id=str(session_id))
                await ws.send_json({"type": "error", "detail": "Malformed JSON message."})
                continue
            except Exception as e:
                log.error("WebSocket receive error", session_id=str(session_id), error=str(e), exc_info=True)
                await ws.send_json({"type": "error", "detail": f"Error processing message: {str(e)}"})
                continue

            # Use our new FSM to handle the incoming event
            try:
                fsm = TutorFSM(tutor_ctx)
                result = await fsm.on_user_message(incoming) # Pass the correctly structured dict
                # Send back the final response from the FSM
                if hasattr(result, 'model_dump'):
                    await ws.send_json(result.model_dump(mode='json'))
                elif isinstance(result, dict):
                    await ws.send_json(result)
                else:
                    # Handle unexpected FSM return types
                    log.warning("Unexpected FSM result type", result_type=type(result), result=result, session_id=str(session_id))
                    await ws.send_json({'type': 'message', 'text': f'Tutor response: {str(result)}'})
                # Persist context after FSM transition
                await session_manager.update_session_context(supabase, session_id, user.id, tutor_ctx)
            except Exception as e:
                # Log the error and inform the client
                log.error("FSMError", error=str(e), session_id=str(session_id), exc_info=True)
                await ws.send_json({"type": "error", "detail": f"Internal tutor error. Please try again."})
                # Consider if context should be persisted here or if the error state needs handling
                # For now, just report and continue the loop

    finally:
        # Graceful close if we exit the loop for any reason
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
        # After streaming events, detect and emit mastery updates
        for topic, mastery_state in tutor_ctx.user_model_state.concepts.items():
            prev = mastery_prev.get(topic)
            new_mastery = mastery_state.mastery
            new_confidence = mastery_state.confidence
            if prev is None or abs(new_mastery - prev) >= 0.05:
                await ws.send_json({
                    "type": "mastery_update",
                    "topic": topic,
                    "mastery": new_mastery,
                    "confidence": new_confidence
                })
        # Update previous mastery snapshot
        mastery_prev = {t: s.mastery for t, s in tutor_ctx.user_model_state.concepts.items()} 