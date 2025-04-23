from fastapi import APIRouter, HTTPException, Depends, Request
from supabase import Client
from gotrue.types import User  # To type hint the user object
from uuid import UUID

from ai_tutor.session_manager import SessionManager
from ai_tutor.dependencies import get_supabase_client  # Get supabase client dependency
from ai_tutor.auth import verify_token  # Get auth dependency
from ai_tutor.api_models import SessionResponse

router = APIRouter()
session_manager = SessionManager()

@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=201,
    summary="Create New Tutoring Session",
    tags=["Session Management"]
)
async def create_new_session(
    request: Request,
    supabase: Client = Depends(get_supabase_client)
):
    """
    Creates a new tutoring session.

    Optionally links the session to a folder if `folder_id` is provided in the JSON body.
    """
    # Retrieve the authenticated user
    user: User = request.state.user

    # Parse incoming JSON body
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Extract and validate folder_id if present
    folder_id = None
    if isinstance(body, dict) and 'folder_id' in body:
        folder_str = body.get('folder_id')
        if folder_str:
            try:
                folder_id = UUID(folder_str)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid folder_id format. Must be a UUID string."
                )
            # Placeholder: verify user owns this folder
            print(f"Verifying ownership for folder: {folder_id}")

    # Create session through SessionManager
    session_id = await session_manager.create_session(
        supabase,
        user.id,
        folder_id
    )

    # Return raw JSON without Pydantic validation
    return {"session_id": str(session_id)}

# Removed the duplicate /sessions endpoint
# @router.post(
#     "/sessions",
#     response_model=None,
#     summary="Create New Tutoring Session",
#     tags=["Session Management"]
# )
# async def create_new_session(
#     request: Request, # Access user from request state
#     supabase: Client = Depends(get_supabase_client)
# ):
#     user: User = request.state.user # Get user from verified token
#     # TODO: Verify user owns the folder_id before creating session (or rely on FK constraint/RLS)
#     session_id: UUID = await session_manager.create_session(supabase, user.id)
#     return {"session_id": str(session_id)} 