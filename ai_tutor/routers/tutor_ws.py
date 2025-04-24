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

        from ai_tutor.fsm import TutorFSM
        log.info(f"WebSocket: Entering main receive loop for session {session_id}")
        while True:
            payload_text = await ws.receive_text()
            log.debug(f"WebSocket: Received raw message for {session_id}: {payload_text}")
            payload = json.loads(payload_text)
            event_data = payload.get('data', {})

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

            log.info(f"WebSocket: Passing event to FSM for session {session_id}. Event type: {incoming['type']}")
            if not ctx:
                 log.error(f"WebSocket: Context became None before FSM call for session {session_id}")
                 await ws.send_json({"type": "error", "detail": "Internal server error: Session context lost."})
                 break

            fsm = TutorFSM(ctx)
            try:
                 final_response_data = await fsm.on_user_message(incoming)
                 log.info(f"WebSocket: FSM processed event for {session_id}. Response type: {getattr(final_response_data, 'response_type', type(final_response_data))}")

                 if hasattr(final_response_data, 'model_dump'):
                      await ws.send_json(final_response_data.model_dump(mode='json'))
                 elif isinstance(final_response_data, dict):
                      await ws.send_json(final_response_data)
                 else:
                      log.warning(f"WebSocket: Unexpected FSM result type for {session_id}", result_type=type(final_response_data))
                      await ws.send_json({'type': 'message', 'text': f'Tutor response: {str(final_response_data)}'})

                 log.info(f"WebSocket: Updating context in DB after FSM step for {session_id}")
                 await session_manager.update_session_context(supabase, session_id, user.id, ctx)
                 log.info(f"WebSocket: Context updated in DB for {session_id}")

            except Exception as fsm_err:
                 log.error(f"WebSocket: FSMError processing event for session {session_id}: {fsm_err}\n{traceback.format_exc()}", exc_info=True)
                 await ws.send_json({"type": "error", "detail": f"Internal tutor error. Please try again."})

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