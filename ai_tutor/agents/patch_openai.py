"""
This module patches the OpenAIResponsesModel to add reasoning effort for o3-mini
"""

from functools import wraps
from agents.models.openai_responses import OpenAIResponsesModel

# Store the original method
original_create = OpenAIResponsesModel._client.responses.create

@wraps(original_create)
async def patched_create(*args, **kwargs):
    """Add reasoning parameter for o3-mini model"""
    if kwargs.get("model") == "o3-mini" and "reasoning" not in kwargs:
        # Add low reasoning effort for o3-mini
        kwargs["reasoning"] = {"effort": "low"}
    return await original_create(*args, **kwargs)

def apply_patches():
    """Apply all patches to OpenAI API"""
    # Replace the responses.create method with our patched version
    OpenAIResponsesModel._client.responses.create = patched_create
    print("Applied patch for o3-mini reasoning effort") 