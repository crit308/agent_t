import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from starlette.websockets import WebSocketState

from src.context import SessionContext, session_manager
from src.agents.planner_agent import run_planner_agent # Assuming planner agent exists
from src.agents.executor_agent import run_executor # We just created this
from src.models.interaction_models import InteractionRequest, InteractionResponseData
# Need authentication mechanism
# from src.api.v1.auth import get_current_user # Replace with your actual auth

logger = logging.getLogger(__name__)

router = APIRouter()

# Placeholder for authentication - replace with your actual dependency
async def get_current_user_placeholder(token: str | None = None):
    # In a real app, decode the token and fetch user data
    if token: 
        logger.info(f"Token received: {token[:10]}...")
        # Dummy user ID for now
        return {"user_id": "dummy_user_for_ws"}
    logger.warning("No token provided for WebSocket connection.")
    # Allow connection for now, but might want to raise HTTPException in production
    # raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user_id": "anonymous_user"}

@router.websocket("/ws/session/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    # token: str | None = Query(None), # Example: Get token from query param
    # user: dict = Depends(get_current_user_placeholder) # Use placeholder auth
):
    user_id = f"user_for_session_{session_id}" # Simplified user ID based on session for now
    logger.info(f"WebSocket connection attempt for session: {session_id} by user: {user_id}")

    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session: {session_id}")

    try:
        # Get or create session context
        context = await session_manager.get_session_context(session_id, user_id)
        if not context:
             # This should ideally not happen if session creation endpoint works
            logger.error(f"Failed to get or create context for session {session_id}")
            await websocket.send_json({"error": "Session context could not be loaded."})
            await websocket.close(code=1008) # Policy Violation or similar code
            return

        logger.info(f"Session context loaded for {session_id}. KB ID: {context.kb_id}, VS ID: {context.vector_store_id}")

        is_first_message = True # Track if this is the first message in this connection

        while websocket.application_state == WebSocketState.CONNECTED:
            try:
                data = await websocket.receive_text() # Or receive_json if FE sends structured data
                logger.info(f"Received message for session {session_id}: {data[:100]}...") # Log truncated data

                # Determine if we need to run the planner or the executor
                # Basic Logic: Planner runs on the very first message *if* no plan exists yet.
                # Executor runs on all subsequent messages *or* if a plan already exists.
                
                response_data: InteractionResponseData | None = None

                # Check if a plan (current focus) exists. If not, run planner first.
                if not context.current_focus_objective:
                    logger.info(f"No focus objective found for session {session_id}. Running planner.")
                    # Assuming planner takes the initial goal or first message as input
                    # The planner should update the context with the focus objective
                    planner_input = data # Use the received text as input for the planner
                    response_data = await run_planner_agent(context, planner_input)
                    is_first_message = False # Planner has run, subsequent messages go to executor
                else:
                    logger.info(f"Focus objective exists for session {session_id}. Running executor.")
                    # Plan exists, run the executor with the user's input
                    executor_input = data
                    response_data = await run_executor(context, executor_input)
                    is_first_message = False # Executor ran

                # Send the response back to the client
                if response_data:
                    logger.info(f"Sending response type '{response_data.content_type}' for session {session_id}")
                    await websocket.send_json(response_data.model_dump())
                else:
                    # Handle cases where no response is generated (e.g., internal errors logged)
                    logger.warning(f"No response data generated for session {session_id}")
                    await websocket.send_json({"status": "error", "message": "Internal processing error."})

                # Persist context changes after the interaction
                await session_manager.save_session_context(context)
                logger.debug(f"Session context saved for {session_id}")

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session: {session_id}")
                await session_manager.save_session_context(context) # Save on disconnect
                break # Exit the loop
            except Exception as e:
                logger.exception(f"Error during WebSocket interaction for session {session_id}: {e}", exc_info=True)
                # Attempt to send an error message to the client if possible
                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"status": "error", "message": f"An unexpected error occurred: {e}"})
                # Consider whether to break or continue depending on the error severity
                break # Break on error for safety

    except Exception as e:
        # Catch errors during initial connection or context loading
        logger.exception(f"Failed to establish WebSocket connection or load context for session {session_id}: {e}", exc_info=True)
        # Ensure websocket is closed if it was opened
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011) # Internal Server Error
    finally:
        logger.info(f"Closing WebSocket connection handler for session {session_id}")
        # Clean up resources if necessary, though context saving is handled in loops/exceptions

# Remove the marker tag causing the linter error
# </rewritten_file> 