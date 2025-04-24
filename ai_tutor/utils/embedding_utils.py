import hashlib, math
from typing import Sequence, List
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

# New: cached cosine similarity for two text snippets
_EMBED_CACHE: dict[str, List[float]] = {}

async def _embed(text: str, embed_fn):
    if text in _EMBED_CACHE:
        return _EMBED_CACHE[text]
    vec = await embed_fn(text)
    _EMBED_CACHE[text] = vec  # type: ignore[arg-type]
    return vec

async def cosine_similarity(a: str, b: str, embed_fn) -> float:
    va, vb = await _embed(a, embed_fn), await _embed(b, embed_fn)
    dot = sum(x*y for x, y in zip(va, vb))
    na = math.sqrt(sum(x*x for x in va))
    nb = math.sqrt(sum(x*x for x in vb))
    if na == 0 or nb == 0:
        return 0.0
    return dot/(na*nb) 