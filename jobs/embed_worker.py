import asyncio
import os
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def poll_and_embed():
    while True:
        # 1. Fetch pending files
        resp = supabase.table("uploaded_files").select("*").eq("embedding_status", "pending").limit(10).execute()
        files = resp.data or []
        if not files:
            await asyncio.sleep(10)
            continue

        # 2. Batch embed (placeholder for actual embedding logic)
        for i, file in enumerate(files):
            file_id = file.get("id")
            supabase_path = file.get("supabase_path")
            print(f"Embedding file: {supabase_path}")
            # TODO: Download file, run embedding, upload to OpenAI, update vector store, etc.
            # Simulate progress
            for progress in range(0, 101, 20):
                # Emit progress event (placeholder)
                supabase.realtime.channel("embedding_progress").send({
                    "event": "processing_update",
                    "payload": {
                        "file_id": file_id,
                        "progress": progress
                    }
                })
                await asyncio.sleep(0.5)
            # 3. Mark as completed
            supabase.table("uploaded_files").update({"embedding_status": "completed"}).eq("id", file_id).execute()
            print(f"Completed embedding for {supabase_path}")
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(poll_and_embed()) 