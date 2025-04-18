import asyncio
import os
from supabase import create_client, Client
from ai_tutor.utils.embedding_utils import get_or_create_embedding
import openai

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def embed_chunk_with_openai(text):
    # Replace with actual OpenAI embedding call
    # For demonstration, simulate a unique vector_id per text
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding_id"]

async def process_chunks(chunks):
    for chunk in chunks:
        vector_id = await get_or_create_embedding(
            chunk, embed_chunk_with_openai
        )
        # Attach vector_id to upsert payload or further processing
        print(f"Chunk embedded with vector_id: {vector_id}")

async def poll_and_embed():
    while True:
        # 1. Fetch pending files
        resp = supabase.table("uploaded_files").select("*").eq("embedding_status", "pending").limit(10).execute()
        files = resp.data or []
        if not files:
            await asyncio.sleep(10)
            continue

        # 2. Batch embed (simulate chunking and embedding logic)
        for i, file in enumerate(files):
            file_id = file.get("id")
            supabase_path = file.get("supabase_path")
            print(f"Embedding file: {supabase_path}")
            # TODO: Download file, chunk it, and embed each chunk
            # Simulate chunking
            chunks = [f"chunk-{j}-{supabase_path}" for j in range(3)]  # Simulate 3 chunks per file
            await process_chunks(chunks)
            # Simulate progress
            for progress in range(0, 101, 20):
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