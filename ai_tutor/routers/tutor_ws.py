from __future__ import annotations
from uuid import UUID
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from supabase import Client

from ai_tutor.dependencies import get_supabase_client
from ai_tutor.session_manager import SessionManager
from ai_tutor.agents.orchestrator_agent import create_orchestrator_agent
from ai_tutor.context import TutorContext
from ai_tutor.auth import verify_token  # Re‑use existing auth logic for header token verification
from agents import Runner, RunConfig

router = APIRouter()

# Use shared session manager instance (same as other routers) – no stateful behaviour here
session_manager = SessionManager()

# Helper to authenticate a websocket connection and return the Supabase user
async def _authenticate_ws(ws: WebSocket, supabase: Client) -> Any:
    """Validate the `Authorization` header for a WebSocket connection.

    Returns the Supabase `User` object on success, otherwise raises and closes the socket.
    """
    # Try Authorization header first
    auth_header = ws.headers.get("authorization")  # headers keys are lower‑cased
    # Fallback: allow token via query param (for browser clients)
    if not auth_header:
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

    # Load tutor context from DB; reject if not available / not owned by user
    tutor_ctx: TutorContext | None = await session_manager.get_session_context(
        supabase=supabase, session_id=session_id, user_id=user.id  # type: ignore[arg-type]
    )
    if tutor_ctx is None:
        await ws.accept()
        await ws.send_json({"type": "error", "detail": "Session not found or not authorized."})
        await ws.close(code=1011)
        return

    # All good – ensure initial focus objective has been generated via the Plan endpoint
    if tutor_ctx.current_focus_objective is None:
        # Inform client to call the Plan endpoint before streaming
        await ws.accept()
        await ws.send_json({
            "type": "error",
            "detail": "Initial planning incomplete. Please call POST /sessions/{session_id}/plan and then reconnect."
        })
        await ws.close(code=1008)
        return
    # Accept the connection
    await ws.accept()

    # Snapshot mastery values to detect later updates
    mastery_prev = {topic: state.mastery for topic, state in tutor_ctx.user_model_state.concepts.items()}

    # Create orchestrator agent once per connection. If you later need per‑message config, re‑create.
    orchestrator_agent = create_orchestrator_agent()

    try:
        while True:
            try:
                incoming: Dict[str, Any] = await ws.receive_json()
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
                            print(f"[WS help_request] Failed to log help_request event: {e}")
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
                stream_result = Runner.run_streamed(
                    orchestrator_agent,
                    input=incoming,
                    context=tutor_ctx,
                    run_config=RunConfig(workflow_name="TutorSession", group_id=str(session_id)),
                )

                async for evt in stream_result.stream_events():
                    await ws.send_json(evt.model_dump())  # Pydantic BaseModel -> dict for JSON serialisation
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