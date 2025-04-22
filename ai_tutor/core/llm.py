"""
LLM abstraction for AI Tutor – simple wrapper around OpenAI chat.
"""
import os
from typing import Any, Dict, List
import openai

class LLMClient:
    """Simple wrapper for OpenAI's ChatCompletion API."""
    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def chat(self, messages: List[Dict[str, Any]]) -> str:
        response = await openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content 