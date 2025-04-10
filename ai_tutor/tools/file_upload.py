import os
import base64
from typing import List, Optional
from datetime import datetime
import logging
from pydantic import BaseModel, Field
from supabase import Client
from uuid import UUID

logger = logging.getLogger(__name__)

class UploadedFile(BaseModel):
    """Represents an uploaded file that has been processed."""
    supabase_path: str
    file_id: Optional[str] = Field(None, description="OpenAI File ID (if used)") # Make optional
    filename: str
    vector_store_id: Optional[str] = Field(None, description="OpenAI Vector Store ID (if used)") # Make optional


class FileUploadManager:
    """Manages file uploads to Supabase Storage."""

    def __init__(self, supabase: Client, vector_store_id: Optional[str] = None):
        """Initialize the FileUploadManager."""
        self.uploaded_files = []
        self.vector_store_id = vector_store_id  # Keep for backwards compatibility
        self.supabase = supabase

    async def upload_and_process_file(self, file_path: str, user_id: UUID, folder_id: UUID, existing_vector_store_id: Optional[str] = None) -> UploadedFile:
        """
        Upload a file to Supabase Storage.
        Updates the corresponding folder record.
        """
        filename = os.path.basename(file_path)
        supabase_path = f"{user_id}/{folder_id}/{filename}"

        # 1. Upload to Supabase Storage
        try:
            with open(file_path, "rb") as file:
                self.supabase.storage.from_("documents").upload(
                    path=supabase_path,
                    file=file,
                    file_options={"content-type": "application/octet-stream"}
                )
            logger.info(f"Successfully uploaded {filename} to Supabase Storage at {supabase_path}")
        except Exception as e:
            raise Exception(f"Failed to upload {filename} to Supabase Storage: {e}")

        # Update folder record (optional, remove vector_store_id field if no longer needed)
        try:
            update_data = {"updated_at": datetime.now().isoformat()}
            update_resp = self.supabase.table("folders").update(update_data).eq("id", str(folder_id)).eq("user_id", user_id).execute()
            if not update_resp.data:
                logger.warning(f"Failed to update folder {folder_id} after Supabase upload: {update_resp.error}")
        except Exception as upd_e:
            logger.error(f"Error updating folder {folder_id} after Supabase upload: {upd_e}")

        # Return info about the Supabase upload
        uploaded_file = UploadedFile(
            supabase_path=supabase_path,
            filename=filename
        )
        self.uploaded_files.append(uploaded_file)
        return uploaded_file

    def get_vector_store_id(self) -> Optional[str]:
        """Get the vector store ID. Returns None if no files have been uploaded."""
        return self.vector_store_id  # Keep method for backwards compatibility


async def upload_document(file_path: str, user_id: str) -> str:
    """
    DEPRECATED: Uploads happen via API. Uploads document to Supabase and OpenAI.
    
    Args:
        file_path: The path to the file to upload.
        user_id: The ID of the user uploading the file.
        
    Returns:
        A confirmation message with Supabase path, file ID, and vector store ID.
    """
    return "Error: Document upload should be handled via the API endpoint, not this tool."