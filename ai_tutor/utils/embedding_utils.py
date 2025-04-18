import hashlib
from ai_tutor.dependencies import get_supabase_client

SUPABASE_CLIENT = get_supabase_client()

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

async def get_or_create_embedding(chunk: str, embed_fn) -> str:
    h = sha256(chunk)
    row = (
        SUPABASE_CLIENT
        .table("embeddings_cache")
        .select("vector_id")
        .eq("hash", h)
        .maybe_single()
        .execute()
    ).data
    if row:
        return row["vector_id"]

    vector_id = await embed_fn(chunk)  # <- your OpenAI embed call
    SUPABASE_CLIENT.table("embeddings_cache").insert(
        {"hash": h, "vector_id": vector_id}
    ).execute()
    return vector_id 