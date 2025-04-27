import logging
from typing import List, Dict, Optional
from ai_tutor.skills import skill
from ai_tutor.dependencies import get_supabase_client
# from agents.run_context import RunContextWrapper # If context needed - Removed as not used directly in skill
from uuid import UUID

logger = logging.getLogger(__name__)

# Simple summarization placeholder - replace with LLM call if needed
def summarize_chunk(chunk_texts: List[str]) -> str:
    combined = "\n".join(chunk_texts)
    # Basic truncation - replace with actual summarization
    return (combined[:450] + '...') if len(combined) > 500 else combined

@skill(cost="low") # Low cost assumes DB fetch is cheap
async def read_interaction_logs(session_id: UUID, max_tokens: int = 1500) -> str:
    """Reads interaction logs for a session, chunking and summarizing if needed to fit within max_tokens."""
    logger.info(f"Reading interaction logs for session: {session_id}")
    supabase = await get_supabase_client()
    # Fetch logs ordered by time
    response = await supabase.table("interaction_logs").select("role, content_type, content").eq("session_id", str(session_id)).order("created_at", desc=False).execute()

    if not response.data:
        logger.warning(f"No interaction logs found for session {session_id}")
        return "No interactions logged for this session."

    full_log_text = ""
    current_chunk_texts = []
    current_chunk_tokens = 0
    # Basic token estimation (split by space)
    token_limit_per_chunk = 500 # Target chunk size for summarization

    for log in response.data:
        role = log.get('role')
        content = log.get('content', '')
        log_line = f"[{role.upper()}]: {content}"
        log_tokens = len(log_line.split())

        if current_chunk_tokens + log_tokens > token_limit_per_chunk:
            # Summarize current chunk and start new one
            chunk_summary = summarize_chunk(current_chunk_texts)
            full_log_text += chunk_summary + "\n---\n"
            current_chunk_texts = [log_line]
            current_chunk_tokens = log_tokens
        else:
            current_chunk_texts.append(log_line)
            current_chunk_tokens += log_tokens

    # Add the last chunk (summarized)
    if current_chunk_texts:
        full_log_text += summarize_chunk(current_chunk_texts)

    # Truncate final summary if it exceeds max_tokens
    final_tokens = len(full_log_text.split())
    if final_tokens > max_tokens:
         logger.warning(f"Truncating summarized interaction log ({final_tokens} tokens > max {max_tokens})")
         # Simple truncation - could be smarter
         estimated_chars = max_tokens * 4 # Rough estimate
         full_log_text = full_log_text[:estimated_chars] + "... (Truncated)"

    logger.info(f"Returning summarized interaction log for session {session_id}. Length: {len(full_log_text)}")
    return full_log_text 