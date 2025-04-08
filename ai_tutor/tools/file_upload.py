import os
import base64
from typing import List, Optional, TYPE_CHECKING
import openai
from pydantic import BaseModel
from supabase import Client
from uuid import UUID

from agents import function_tool

if TYPE_CHECKING:
    from ai_tutor.api import get_supabase_client


class UploadedFile(BaseModel):
    """Represents an uploaded file that has been processed."""
    supabase_path: str
    file_id: str
    filename: str
    vector_store_id: str


class FileUploadManager:
    """Manages the upload and processing of files for the AI tutor."""
    
    def __init__(self, supabase: Client, vector_store_id: Optional[str] = None):
        """Initialize the FileUploadManager."""
        # API key is handled globally by the SDK setup
        self.client = openai.Client() # Relies on globally configured key/client
        self.uploaded_files = []
        self.vector_store_id = vector_store_id # Initialize with passed ID
        self.supabase = supabase
        self.bucket_name = "document_uploads"
    
    async def upload_and_process_file(self, file_path: str, user_id: UUID, folder_id: UUID, existing_vector_store_id: Optional[str] = None) -> UploadedFile:
        """
        Upload a file to Supabase Storage, then to OpenAI, and add to Vector Store.
        Uses existing_vector_store_id if provided, otherwise creates a new one.
        Updates the corresponding folder record with the vector store ID.
        """
        # Use provided existing ID or the manager's stored ID
        self.vector_store_id = existing_vector_store_id or self.vector_store_id

        filename = os.path.basename(file_path)
        supabase_path = f"{user_id}/{folder_id}/{filename}"

        # 1. Upload to Supabase Storage
        try:
            with open(file_path, "rb") as file:
                upload_response = self.supabase.storage.from_(self.bucket_name).upload(
                    path=supabase_path,
                    file=file,
                    file_options={"content-type": "application/octet-stream"}
                )
            print(f"Successfully uploaded {filename} to Supabase Storage at {supabase_path}")
        except Exception as e:
            raise Exception(f"Failed to upload {filename} to Supabase Storage: {e}")

        # 2. Upload the file content to OpenAI for assistants
        with open(file_path, "rb") as file:
            response = self.client.files.create(
                file=file,
                purpose="assistants"
            )
            
            file_id = response.id
            
            print(f"Successfully uploaded file content: {filename}, OpenAI File ID: {file_id}")
            
            # Create a vector store if one doesn't exist *or wasn't provided*
            vector_store_created_now = False
            if not self.vector_store_id:
                # Create a vector store with a meaningful name
                vs_response = self.client.vector_stores.create(
                    name=f"AI Tutor Vector Store - {filename}"
                )
                self.vector_store_id = vs_response.id
                vector_store_created_now = True
                print(f"Created NEW vector store: {self.vector_store_id}")
            else:
                print(f"Using EXISTING vector store: {self.vector_store_id}")

            # Add the file to the vector store
            self.client.vector_stores.files.create(
                vector_store_id=self.vector_store_id,
                file_id=file_id
            )
            
            print(f"Added file {file_id} to vector store {self.vector_store_id}")
            
            # Check status of the file in vector store
            files_status = self.client.vector_stores.files.list(
                vector_store_id=self.vector_store_id
            )
            print(f"Vector store files status: {files_status}")
            
            # Update the folder record with the vector_store_id (if newly created or maybe always to be safe)
            try:
                update_resp = self.supabase.table("folders").update(
                    {"vector_store_id": self.vector_store_id}
                ).eq("id", str(folder_id)).eq("user_id", user_id).execute()
                if update_resp.data:
                    print(f"Updated folder {folder_id} with vector_store_id {self.vector_store_id}")
                else: print(f"Warning: Failed to update folder {folder_id} with vector_store_id: {update_resp.error}")
            except Exception as upd_e:
                print(f"Error updating folder {folder_id} with vector_store_id: {upd_e}")
            
            # Create and return an uploaded file
            uploaded_file = UploadedFile(
                supabase_path=supabase_path,
                file_id=file_id,
                filename=filename,
                vector_store_id=self.vector_store_id
            )
            
            self.uploaded_files.append(uploaded_file)
            return uploaded_file

    def get_vector_store_id(self) -> str:
        """Get the vector store ID. Returns None if no files have been uploaded."""
        return self.vector_store_id
    
    def get_uploaded_files(self) -> List[UploadedFile]:
        """Get a list of all uploaded files."""
        return self.uploaded_files


@function_tool
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