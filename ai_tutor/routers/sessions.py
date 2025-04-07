from fastapi import APIRouter, HTTPException, Depends, Request
from supabase import Client
from gotrue.types import User # To type hint the user object

from ai_tutor.session_manager import SessionManager
from ai_tutor.api_models import SessionResponse
from ai_tutor.dependencies import get_supabase_client # Get supabase client dependency
from ai_tutor.auth import verify_token # Get auth dependency

router = APIRouter()
session_manager = SessionManager()

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
    session_id = await session_manager.create_session(supabase, user.id)
    return SessionResponse(session_id=session_id) 