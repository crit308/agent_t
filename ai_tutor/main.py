import os
import sys

# Add the parent directory to the Python path so we can import the AI tutor package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the CLI module
from ai_tutor.cli import main
from agents import set_default_openai_api
from openai import AsyncOpenAI
from agents import set_default_openai_client

# We need to directly modify the OpenAI client here
from openai.resources.responses import Responses

# Add reasoning parameter to o3-mini calls
original_create = Responses.create

async def patched_create(self, *args, **kwargs):
    """Add reasoning parameter for o3-mini model"""
    if kwargs.get("model") == "o3-mini" and "reasoning" not in kwargs:
        # Add low reasoning effort for o3-mini
        kwargs["reasoning"] = {"effort": "low"}
    
    return await original_create(self, *args, **kwargs)

# Apply the patch
Responses.create = patched_create

if __name__ == "__main__":
    # Use the CLI's main function
    from ai_tutor.cli import parser
    
    args = parser.parse_args()
    
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
    
    print("Applied patch for o3-mini with low reasoning effort")
    
    # Run the CLI's main function
    import asyncio
    asyncio.run(main(args.files, args.api_key, args.output)) 