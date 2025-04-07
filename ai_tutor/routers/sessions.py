from fastapi import APIRouter, HTTPException, Depends, Request, Body
from supabase import Client
from gotrue.types import User # To type hint the user object
from uuid import UUID # Import UUID

from ai_tutor.session_manager import SessionManager
from ai_tutor.api_models import SessionResponse, FolderResponse # Add FolderResponse for typing if needed
from ai_tutor.dependencies import get_supabase_client # Get supabase client dependency
from ai_tutor.auth import verify_token # Get auth dependency
from pydantic import BaseModel

router = APIRouter()
session_manager = SessionManager()

@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=201,
    summary="Create New Tutoring Session",
    tags=["Session Management"]
)
async def create_new_session_for_folder(
    request: Request, # Access user from request state
    folder_id: UUID = Body(..., embed=True, description="The ID of the folder to associate the session with."), # Get folder_id from body
    supabase: Client = Depends(get_supabase_client)
):
    """Creates a new session linked to a specific folder."""
    user: User = request.state.user # Get user from verified token
    # TODO: Verify user owns the folder_id before creating session (or rely on FK constraint/RLS)
    session_id: UUID = await session_manager.create_session(supabase, user.id, folder_id)
    return SessionResponse(session_id=session_id)

@router.post(
    "/sessions",
    response_model=SessionResponse,
    summary="Create New Tutoring Session",
    tags=["Session Management"]
)
async def create_new_session(
    request: Request, # Access user from request state
    supabase: Client = Depends(get_supabase_client)
):
    user: User = request.state.user # Get user from verified token
    # TODO: Verify user owns the folder_id before creating session (or rely on FK constraint/RLS)
    session_id: UUID = await session_manager.create_session(supabase, user.id)
    return SessionResponse(session_id=session_id) 