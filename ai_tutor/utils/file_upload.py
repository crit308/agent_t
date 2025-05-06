import os
import base64
from typing import List, Optional, TYPE_CHECKING
import openai
import asyncio
import time
from pydantic import BaseModel
from supabase import Client
from uuid import UUID

if TYPE_CHECKING:
    from ai_tutor.dependencies import get_supabase_client

# ---------------------------------------------------------------------------
# Utility data structure
# ---------------------------------------------------------------------------
class UploadedFile(BaseModel):
    """Represents an uploaded file that has been processed."""
    supabase_path: str
    file_id: Optional[str] = None  # Allow None when queuing embeddings
    filename: str
    vector_store_id: Optional[str] = None  # May be None until vector store is created


# ---------------------------------------------------------------------------
# Main upload manager – **utility**, not a callable skill
# ---------------------------------------------------------------------------
class FileUploadManager:
    """Manages the upload and processing of files for the AI tutor."""

    def __init__(self, supabase: Client, vector_store_id: Optional[str] = None):
        # API key is handled globally by the SDK setup
        self.client = openai.Client()  # Relies on globally configured key/client
        self.uploaded_files: List[UploadedFile] = []
        self.vector_store_id = vector_store_id
        self.supabase = supabase
        self.bucket_name = "document_uploads"

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------
    async def upload_and_process_file(
        self,
        file_path: str,
        user_id: UUID,
        folder_id: UUID,
        existing_vector_store_id: Optional[str] = None,
        queue_embedding: bool = False,
    ) -> UploadedFile:
        """Upload a file and push it through OpenAI + vector-store pipeline."""

        self.vector_store_id = existing_vector_store_id or self.vector_store_id
        filename = os.path.basename(file_path)
        supabase_path = f"{user_id}/{folder_id}/{filename}"

        # ---- 1. Upload binary to Supabase Storage --------------------------------
        try:
            with open(file_path, "rb") as file:
                self.supabase.storage.from_(self.bucket_name).upload(
                    path=supabase_path,
                    file=file,
                    file_options={"content-type": "application/octet-stream"},
                )
        except Exception as e:
            raise RuntimeError(f"Failed to upload {filename} to Supabase: {e}")

        # Fast path: async embedding queue only
        if queue_embedding:
            return await self._handle_async_embedding(
                supabase_path, filename, user_id, folder_id
            )

        # ---- 2. Upload to OpenAI files endpoint ----------------------------------
        with open(file_path, "rb") as file:
            response = self.client.files.create(file=file, purpose="assistants")
        file_id = response.id

        # Ensure vector store exists ------------------------------------------------
        vector_store_created_now = False
        if not self.vector_store_id:
            vs_response = self.client.vector_stores.create(
                name=f"AI Tutor Vector Store - {filename}"
            )
            self.vector_store_id = vs_response.id
            vector_store_created_now = True
        # Add file to vector store
        self.client.vector_stores.files.create(
            vector_store_id=self.vector_store_id, file_id=file_id
        )

        # Poll status --------------------------------------------------------------
        await self._poll_file_processing(file_id)

        # Record vector_store_id on folder row (safe upsert)
        try:
            self.supabase.table("folders").update({"vector_store_id": self.vector_store_id}).eq(
                "id", str(folder_id)
            ).eq("user_id", user_id).execute()
        except Exception:
            pass

        uploaded = UploadedFile(
            supabase_path=supabase_path,
            file_id=file_id,
            filename=filename,
            vector_store_id=self.vector_store_id,
        )
        self.uploaded_files.append(uploaded)
        return uploaded

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    async def _poll_file_processing(self, file_id: str, *, timeout: int = 120, interval: int = 2):
        start = time.time()
        while time.time() - start < timeout:
            retrieved = self.client.vector_stores.files.retrieve(
                vector_store_id=self.vector_store_id, file_id=file_id
            )
            status = retrieved.status
            if status == "completed":
                return
            if status in {"failed", "cancelled", "expired"}:
                raise RuntimeError(f"OpenAI file processing {status} for {file_id}.")
            await asyncio.sleep(interval)
        raise TimeoutError(f"File processing timed out for {file_id} after {timeout}s")

    async def _handle_async_embedding(
        self,
        supabase_path: str,
        filename: str,
        user_id: UUID,
        folder_id: UUID,
    ) -> UploadedFile:
        if not self.vector_store_id:
            vs_response = self.client.vector_stores.create(
                name=f"AI Tutor Vector Store - {filename}"
            )
            self.vector_store_id = vs_response.id
            # Record vector_store on folder
            try:
                self.supabase.table("folders").update({"vector_store_id": self.vector_store_id}).eq(
                    "id", str(folder_id)
                ).eq("user_id", user_id).execute()
            except Exception:
                pass
        # Mark file pending embedding
        self.supabase.table("uploaded_files").insert(
            {
                "supabase_path": supabase_path,
                "user_id": str(user_id),
                "folder_id": str(folder_id),
                "embedding_status": "pending",
            }
        ).execute()
        return UploadedFile(
            supabase_path=supabase_path,
            filename=filename,
            vector_store_id=self.vector_store_id,
        )

    # ---------------------------------------------------------------------
    # Convenience accessors
    # ---------------------------------------------------------------------
    def get_vector_store_id(self) -> Optional[str]:
        return self.vector_store_id

    def get_uploaded_files(self) -> List[UploadedFile]:
        return self.uploaded_files 