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
from ai_tutor.agents.orchestrator_agent import create_orchestrator_agent, get_orchestrator_cached
from ai_tutor.context import TutorContext, UserModelState
from ai_tutor.auth import verify_token  # Re‑use existing auth logic for header token verification
from agents import Runner, RunConfig
import json
from fastapi_utils.tasks import repeat_every
import structlog

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
    Each inbound JSON message is forwarded to the Orchestrator agent. All streaming events
    emitted by the agent are relayed back to the client in real‑time.
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

    # Create orchestrator agent once per process (cached).
    agent = get_orchestrator_cached()

    tutor_ctx = ctx  # Use hydrated context for rest of handler

    try:
        while True:
            try:
                # Read raw payload text and normalize fields
                payload = json.loads(await ws.receive_text())
                if 'event_type' not in payload and 'type' in payload:
                    payload = {'event_type': payload['type'], **payload.get('data', {})}
                incoming: Dict[str, Any] = payload
                # Handle pace slider change: update learning_pace_factor
                if incoming.get("type") == "pace_change":
                    value = incoming.get("value")
                    try:
                        factor = float(value)
                        tutor_ctx.user_model_state.learning_pace_factor = factor
                        await session_manager.update_session_context(supabase, session_id, user.id, tutor_ctx)
                        # Acknowledge to client
                        await ws.send_json({"type": "pace_change", "value": factor})
                    except Exception as e:
                        await ws.send_json({"type": "error", "detail": f"Invalid pace_change value: {e}"})
                    continue
                # Handle "I'm stuck" help request: set pending summary prompt
                if incoming.get("type") == "help_request":
                    mode = incoming.get("mode")
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
                    continue
            except WebSocketDisconnect:  # client closed connection
                break
            except Exception:  # Malformed JSON, etc.
                await ws.send_json({"type": "error", "detail": "Malformed JSON message."})
                continue

            # Run the orchestrator in streaming mode
            try:
                stream = Runner.run_streamed(
                    agent, input=incoming, context=tutor_ctx,
                    run_config=RunConfig(workflow_name="TutorSession", group_id=str(session_id)),
                )
                async for evt in stream.stream_events():
                    await ws.send_json(evt.model_dump())   # token‑level latency
            except Exception as exc:  # noqa: BLE001
                # Relay the exception back and continue (or you could terminate)
                await ws.send_json({"type": "error", "detail": str(exc)})
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