import asyncio
import uuid
import logging
from typing import List, Dict, Any

from fastapi import WebSocket

from ai_tutor.context import TutorContext
from ai_tutor.skills.utils.skill_wrapper import skill, RunContextWrapper
from ai_tutor.routers.tutor_ws import safe_send_json # Assuming safe_send_json can be imported

log = logging.getLogger(__name__)

# Define a timeout for waiting for the frontend response (e.g., 10 seconds)
BOARD_STATE_RESPONSE_TIMEOUT = 10.0 

@skill
async def get_board_state(ctx_wrapper: RunContextWrapper[TutorContext], ws: WebSocket) -> List[Dict[str, Any]]:
    """
    Requests the current state of all objects on the whiteboard from the frontend.

    This skill sends a WebSocket message to the client, waits for a response containing
    the whiteboard objects' specifications, and then returns them.
    """
    ctx = ctx_wrapper.context
    request_id = str(uuid.uuid4())
    future = asyncio.Future()

    if ctx.pending_board_state_requests is None: # Should be initialized by Pydantic, but as a safeguard
        ctx.pending_board_state_requests = {}
        
    ctx.pending_board_state_requests[request_id] = future
    log.info(f"get_board_state: Stored future for request_id={request_id}")

    try:
        await safe_send_json(
            ws,
            {"type": "REQUEST_BOARD_STATE", "request_id": request_id},
            log_context=f"get_board_state (req_id: {request_id})"
        )
        log.info(f"get_board_state: Sent REQUEST_BOARD_STATE for request_id={request_id}")

        # Wait for the future to be resolved by the incoming message handler
        board_specs = await asyncio.wait_for(future, timeout=BOARD_STATE_RESPONSE_TIMEOUT)
        log.info(f"get_board_state: Received board state for request_id={request_id}. Objects: {len(board_specs)}")
        return board_specs
    except asyncio.TimeoutError:
        log.warning(f"get_board_state: Timeout waiting for board state response for request_id={request_id}")
        # Decide on behavior: raise error, return empty list, or return specific error object
        return [{"error": "Timeout waiting for board state from client."}] 
    except Exception as e:
        log.error(f"get_board_state: Error during board state retrieval for request_id={request_id}: {e}", exc_info=True)
        return [{"error": f"An unexpected error occurred: {str(e)}"}]
    finally:
        # Clean up the future from the dictionary once done (either success, timeout, or error)
        if request_id in ctx.pending_board_state_requests:
            del ctx.pending_board_state_requests[request_id]
            log.debug(f"get_board_state: Cleaned up future for request_id={request_id}") 