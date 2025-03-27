import os
import base64
from typing import List
import openai
from pydantic import BaseModel

from agents import function_tool


class UploadedFile(BaseModel):
    """Represents an uploaded file that has been processed."""
    file_id: str
    filename: str
    vector_store_id: str


class FileUploadManager:
    """Manages the upload and processing of files for the AI tutor."""
    
    def __init__(self):
        """Initialize the FileUploadManager."""
        # API key is handled globally by the SDK setup
        self.client = openai.Client() # Relies on globally configured key/client
        self.uploaded_files = []
        self.vector_store_id = None
    
    def upload_and_process_file(self, file_path: str) -> UploadedFile:
        """Upload a file to OpenAI and create a vector store for it."""
        # Upload the file to OpenAI for assistants
        with open(file_path, "rb") as file:
            response = self.client.files.create(
                file=file,
                purpose="assistants"
            )
            
            file_id = response.id
            filename = os.path.basename(file_path)
            
            print(f"Successfully uploaded file: {filename}, ID: {file_id}")
            
            # Create a vector store if one doesn't exist yet
            if not self.vector_store_id:
                # Create a vector store with a meaningful name
                vs_response = self.client.vector_stores.create(
                    name=f"AI Tutor Vector Store - {filename}"
                )
                self.vector_store_id = vs_response.id
                print(f"Created vector store: {self.vector_store_id}")
            
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
            
            # Create and return an uploaded file
            uploaded_file = UploadedFile(
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
def upload_document(file_path: str) -> str:
    """
    Upload a document to be used by the AI tutor.
    
    Args:
        file_path: The path to the file to upload.
        
    Returns:
        A confirmation message with the file ID and vector store ID.
    """
    # Create a file upload manager
    manager = FileUploadManager() # API key handled globally
    
    # Upload and process the file
    try:
        uploaded_file = manager.upload_and_process_file(file_path)
        return f"Successfully uploaded {uploaded_file.filename}. File ID: {uploaded_file.file_id}. Vector Store ID: {uploaded_file.vector_store_id}"
    except Exception as e:
        return f"Error uploading file: {str(e)}" 