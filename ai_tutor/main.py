import os
import sys

# Add the parent directory to the Python path so we can import the AI tutor package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the CLI module
from agents import set_default_openai_api
from openai import AsyncOpenAI
from agents import set_default_openai_client

# We need to directly modify the OpenAI client here
from openai.resources.responses import Responses
from ai_tutor.agents.models import PrecisionControlEncoder

# Keep a reference to the original methods
import openai
import json as _stdlib_json
import decimal

# --- START JSON Deserialization Patch ---

# Keep the original loads function
original_json_loads = _stdlib_json.loads

def _limit_float_precision_hook(value, max_places=8):
    """Applies precision limiting logic."""
    try:
        # Use string formatting for robust precision control
        formatted_string = f"{float(value):.{max_places}f}"
        limited_float = float(formatted_string)
        # Double-check string representation
        str_val_check = str(limited_float)
        if '.' in str_val_check and len(str_val_check.split('.')[1]) > max_places:
             int_part, decimal_part = str_val_check.split('.', 1)
             limited_float = float(f"{int_part}.{decimal_part[:max_places]}")
        return limited_float
    except (ValueError, TypeError):
        return value # Return original value if conversion fails

def safe_precision_object_hook(dct):
    """Object hook for json.loads to limit float precision by creating a new dict."""
    new_dct = {}
    for key, value in dct.items():
        if isinstance(value, float):
            new_dct[key] = _limit_float_precision_hook(value, 8)
        # No explicit recursion needed here for dicts/lists;
        # json.loads calls the hook for nested dicts automatically.
        # We just copy non-float values.
        else:
            new_dct[key] = value
    return new_dct

def patched_json_loads(*args, **kwargs):
    """Wrapper for json.loads that applies the safe precision limiting object_hook."""
    # Set the object_hook that creates new dictionaries
    kwargs['object_hook'] = safe_precision_object_hook
    return original_json_loads(*args, **kwargs)

# Apply the patch globally
_stdlib_json.loads = patched_json_loads
print("Applied global patch to json.loads (Safe Hook) for precision control.")

# --- END JSON Deserialization Patch ---

# We'll keep the o3-mini reasoning patch, but remove the precision trimming
original_create = Responses.create

async def patched_create_reasoning_only(self, *args, **kwargs):
    """Add reasoning parameter for o3-mini model without data trimming"""
    # Add reasoning parameter if needed
    if kwargs.get("model") == "o3-mini" and "reasoning" not in kwargs:
        # Add low reasoning effort for o3-mini
        kwargs["reasoning"] = {"effort": "low"}
        print("DEBUG: Added reasoning=low for o3-mini")

    # Call the original create method WITHOUT data trimming here
    return await original_create(self, *args, **kwargs)

# Apply the reasoning-only patch
Responses.create = patched_create_reasoning_only
print("Applied patch for o3-mini with low reasoning effort (precision handled in handoff filters)")

if __name__ == "__main__":
    # Import the CLI's parser
    from ai_tutor.cli import parser
    
    # Check if first argument is not a subcommand, then we assume old-style command
    # and we'll use the "tutor" subcommand
    if len(sys.argv) > 1 and sys.argv[1] not in ['tutor', 'analyze', '-h', '--help']:
        # Insert the 'tutor' subcommand as the first argument
        sys.argv.insert(1, 'tutor')
        print("Running in 'tutor' mode with automatic subcommand insertion")
    
    # Parse arguments with potentially modified sys.argv
    args = parser.parse_args()
    
    # Default to tutor command if none specified
    if not hasattr(args, 'command') or not args.command:
        args.command = 'tutor'
    
    # Set default API key if provided in environment
    if not args.api_key:
        # Use the hard-coded API key if environment variable is not set
        api_key = "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"
        # Set the API key as an environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        args.api_key = api_key
        print(f"Using API key from code and setting as OPENAI_API_KEY environment variable")
    else:
        # If API key is provided by command line, still set it as an environment variable
        os.environ["OPENAI_API_KEY"] = args.api_key
        print(f"Setting provided API key as OPENAI_API_KEY environment variable")
    
    # Ensure we're using the Responses API for o3-mini model
    set_default_openai_api("responses")
    
    # Create a custom OpenAI client with reasoning effort configuration
    openai_client = AsyncOpenAI(api_key=args.api_key)
    set_default_openai_client(openai_client)
    
    print("Precision control attempted via global json.loads patch and handoff filters.")
    print("Set OPENAI_API_KEY environment variable for API and tracing")
    
    # Run the CLI's main function
    import asyncio
    from ai_tutor.cli import main
    asyncio.run(main(args)) 