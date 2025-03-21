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
    
    print("Applied patch for o3-mini with low reasoning effort")
    
    # Run the CLI's main function
    import asyncio
    from ai_tutor.cli import main
    asyncio.run(main(args)) 