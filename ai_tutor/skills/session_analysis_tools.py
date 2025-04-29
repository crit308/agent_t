import logging
from typing import List, Dict, Optional
from ai_tutor.skills import skill
from ai_tutor.dependencies import get_supabase_client
from ai_tutor.core.llm import LLMClient
# from agents.run_context import RunContextWrapper # If context needed - Removed as not used directly in skill
from uuid import UUID
import asyncio # Added import

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM_PROMPT = (
    "You are an expert summarizer. Briefly summarize the key events and takeaways from the "
    "following interaction log snippet between an AI Tutor Agent and a User. Focus on learning "
    "progress, questions asked, answers given (correct/incorrect), and topics covered. "
    "Output only the summary text."
)

# Simple summarization placeholder - replace with LLM call if needed
# def summarize_chunk(chunk_texts: List[str]) -> str: # Removed placeholder
#     combined = "\n".join(chunk_texts)
#     # Basic truncation - replace with actual summarization
#     return (combined[:450] + '...') if len(combined) > 500 else combined

@skill(cost="medium") # Increased cost estimate due to LLM calls
async def read_interaction_logs(session_id: UUID, max_tokens: int = 1500) -> str:
    """Reads interaction logs for a session, chunking and summarizing using an LLM if needed to fit within max_tokens."""
    logger.info(f"Reading interaction logs for session: {session_id}")
    supabase = await get_supabase_client()
    # Fetch logs ordered by time
    response = supabase.table("interaction_logs") \
        .select("role, content_type, content") \
        .eq("session_id", str(session_id)) \
        .order("created_at", desc=False) \
        .execute()

    if not response.data:
        logger.warning(f"No interaction logs found for session {session_id}")
        return "No interactions logged for this session."

    # Initialize LLM client once
    try:
        llm_client = LLMClient(model_name="o3-mini") # Use correct argument name 'model_name'
    except Exception as e:
        logger.error(f"Failed to initialize LLMClient: {e}. Cannot perform summarization.")
        return "Error: Could not initialize summarization service."

    full_log_text = ""
    current_chunk_texts = []
    # Simple token estimation (split by space). Consider a proper tokenizer for more accuracy.
    # Reduced limit slightly to account for prompt/overhead and use word count as proxy
    token_limit_per_chunk = 400
    estimated_summary_tokens_per_chunk = 100 # Estimate reduction factor

    async def get_chunk_summary(texts: List[str]) -> str:
        chunk_content = "\n".join(texts)
        if not chunk_content.strip():
             return "" # Skip empty chunks
        try:
            messages = [
                {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": chunk_content},
            ]
            # Use text response format
            summary = await llm_client.chat(messages, response_format={'type': 'text'})
            logger.debug(f"Successfully summarized chunk for session {session_id}")
            return summary
        except Exception as e:
            logger.error(f"LLM summarization failed for chunk in session {session_id}: {e}")
            # Fallback: Return a placeholder or truncated original text
            # return (chunk_content[:300] + "... [Summarization Failed]")
            return "[Chunk summarization failed]" # Using placeholder as fallback

    # Process logs chunk by chunk
    estimated_total_summary_tokens = 0
    for log in response.data:
        role = log.get('role')
        content = log.get('content', '')
        log_line = f"[{role.upper()}]: {content}"
        # Basic token estimation (split by space)
        log_tokens = len(log_line.split())

        # Check if adding this log exceeds the chunk limit OR if adding the estimated summary size
        # would exceed the overall max_tokens limit.
        if current_chunk_texts and (len("\n".join(current_chunk_texts).split()) + log_tokens > token_limit_per_chunk):

            chunk_summary = await get_chunk_summary(current_chunk_texts)
            estimated_total_summary_tokens += len(chunk_summary.split())

            # Safety check before appending: ensure adding summary won't exceed max_tokens
            if estimated_total_summary_tokens > max_tokens:
                 logger.warning(f"Stopping summarization early for session {session_id} due to max_tokens limit ({max_tokens}).")
                 full_log_text += "... [Log truncated due to length before full summarization]"
                 break # Stop processing further logs

            full_log_text += chunk_summary + "\n---\n"
            current_chunk_texts = [log_line]
        else:
            current_chunk_texts.append(log_line)

        # Check if we already broke out of the loop
        if "... [Log truncated due to length before full summarization]" in full_log_text:
            break

    # Process the last chunk if it hasn't been truncated away
    if current_chunk_texts and "... [Log truncated due to length before full summarization]" not in full_log_text:
        chunk_summary = await get_chunk_summary(current_chunk_texts)
        estimated_total_summary_tokens += len(chunk_summary.split())
        if estimated_total_summary_tokens <= max_tokens:
            full_log_text += chunk_summary
        else:
             # Append truncated summary if possible, or just truncation message
             remaining_tokens = max_tokens - (estimated_total_summary_tokens - len(chunk_summary.split()))
             if remaining_tokens > 50: # Only add truncated summary if space allows
                 estimated_chars = remaining_tokens * 4 # Rough estimate
                 full_log_text += chunk_summary[:estimated_chars] + "... (Truncated Summary)"
             else:
                 full_log_text += "... [Final chunk summary truncated due to length]"


    # Final check - If somehow the summary still exceeds max_tokens (e.g., bad estimation), truncate brutally.
    # This is less likely now with checks during chunk processing, but serves as a final safety net.
    final_tokens = len(full_log_text.split())
    if final_tokens > max_tokens:
         logger.warning(f"Final summarized log still exceeds max_tokens ({final_tokens} > {max_tokens}). Applying final truncation for session {session_id}.")
         # Simple character-based truncation based on token estimate
         estimated_chars = max_tokens * 4 # Rough estimate
         full_log_text = full_log_text[:estimated_chars] + "... (Final Truncation)"

    logger.info(f"Returning summarized interaction log for session {session_id}. Approx tokens: {final_tokens}. Length: {len(full_log_text)}")
    return full_log_text 