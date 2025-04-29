"""
LLM abstraction for AI Tutor – simple wrapper around OpenAI chat.
"""
import os
import asyncio
from typing import Any, Dict, List
import openai

class LLMClient:
    """Simple wrapper for OpenAI's ChatCompletion API."""
    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-2025-04-14")
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        **openai_kwargs: Any,
    ) -> str | Dict[str, Any]:
        # Call the synchronous OpenAI API method in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **openai_kwargs
            )
        )
        # If the reply is regular text, return it; if content is None (JSON-mode),
        # fall back to the full message dict so callers can parse it.
        msg = response.choices[0].message
        return msg.content or msg.model_dump() 