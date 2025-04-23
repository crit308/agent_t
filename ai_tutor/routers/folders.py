from fastapi import APIRouter, Depends, HTTPException, Request
from supabase import Client, PostgrestAPIResponse
from gotrue.types import User
from typing import List
from uuid import UUID

from ai_tutor.dependencies import get_supabase_client
from ai_tutor.auth import verify_token
from ai_tutor.api_models import FolderCreateRequest, FolderResponse, FolderListResponse

router = APIRouter(
    prefix="/folders",
    tags=["Folder Management"],
    dependencies=[Depends(verify_token)] # Secure all folder endpoints
)

@router.post("/", response_model=None, status_code=201)
async def create_folder(
    folder_data: FolderCreateRequest,
    request: Request,
    supabase: Client = Depends(get_supabase_client)
):
    """Creates a new folder for the authenticated user."""
    user: User = request.state.user
    try:
        response: PostgrestAPIResponse = supabase.table("folders").insert({
            "user_id": user.id,
            "name": folder_data.name
        }).execute()

        # Check if data is a list and not empty
        if response.data and isinstance(response.data, list) and len(response.data) > 0:
            inserted_folder = response.data[0] # Get the first (and only) inserted item
            # Ensure created_at is stringified for the Pydantic model
            return {
                "id": str(inserted_folder["id"]),
                "name": inserted_folder["name"],
                "created_at": str(inserted_folder.get("created_at"))
            }
        else:
            # Log potential error if available, or just the response
            error_details = getattr(response, 'error', 'No specific error details')
            print(f"Error creating folder: {error_details}")
            raise HTTPException(status_code=400, detail=f"Failed to create folder: {response.error.message if response.error else 'Unknown error'}")
    except Exception as e:
        print(f"Exception creating folder: {e}")
        raise HTTPException(status_code=500, detail=f"Database error during folder creation: {str(e)}")

@router.get("/", response_model=None)
async def list_folders(
    request: Request,
    supabase: Client = Depends(get_supabase_client)
):
    """Lists all folders belonging to the authenticated user."""
    user: User = request.state.user
    try:
        response: PostgrestAPIResponse = supabase.table("folders").select("id, name, created_at").eq("user_id", user.id).order("created_at", desc=True).execute()

        if response.data:
            # Convert created_at to string for each folder
            folders = [
                {
                    "id": str(item["id"]),
                    "name": item["name"],
                    "created_at": str(item["created_at"]),
                }
                for item in response.data
            ]
            return {"folders": folders}
        elif isinstance(response.data, list) and not response.data:
            return {"folders": []}
        else:
            # Handle unexpected response structure or potential error not caught by exception
            print(f"Unexpected response structure or error listing folders: {response}") # Log the whole response
            raise HTTPException(status_code=500, detail="Failed to list folders due to unexpected response.")

    except Exception as e:
        print(f"Exception listing folders: {e}")
        raise HTTPException(status_code=500, detail=f"Database error listing folders: {str(e)}")

# Add DELETE /folders/{folder_id} and PUT /folders/{folder_id} (for renaming) endpoints as needed 