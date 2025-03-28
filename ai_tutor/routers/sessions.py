from fastapi import APIRouter, HTTPException

from ai_tutor.session_manager import SessionManager
from ai_tutor.api_models import SessionResponse

router = APIRouter()
session_manager = SessionManager()

@router.post(
    "/sessions",
    response_model=SessionResponse,
    summary="Create New Tutoring Session",
    tags=["Session Management"]
)
async def create_new_session():
    session_id = session_manager.create_session()
    return SessionResponse(session_id=session_id) 