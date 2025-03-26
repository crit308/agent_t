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

# Monkey patch the json module used by OpenAI to control floating point precision
import openai
import json as _stdlib_json
import decimal

# Store the original dumps function for later restoration if needed
original_json_dumps = _stdlib_json.dumps

# Create a function to trim all floats to 15 decimal places or fewer
def trim_float_precision(obj, max_decimals=15):
    """Recursively process any object to limit float precision."""
    if isinstance(obj, float):
        # Format with exact decimal places to ensure control
        return float(f"{obj:.{max_decimals}f}")
    elif isinstance(obj, dict):
        return {k: trim_float_precision(v, max_decimals) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(trim_float_precision(item, max_decimals) for item in obj)
    else:
        return obj

# Create a wrapper function that uses our precision-controlled encoder
def precision_json_dumps(*args, **kwargs):
    # Process the arguments first to limit any float precision
    if args and len(args) > 0:
        args = list(args)
        args[0] = trim_float_precision(args[0], max_decimals=15)
        args = tuple(args)
    
    # Use our custom encoder by default if not specified
    if 'cls' not in kwargs:
        kwargs['cls'] = PrecisionControlEncoder
        # Set max_decimals to 15 by default (below OpenAI's 16 limit)
        kwargs['max_decimals'] = 15
    
    return original_json_dumps(*args, **kwargs)

# Apply the patch to the json module
_stdlib_json.dumps = precision_json_dumps

# Patch the client's request method to ensure all floating point values are trimmed
try:
    # Get the OpenAI client's request method
    from openai.http_client import HttpClient
    original_request = HttpClient.request
    
    # Create a patched version
    async def patched_request(self, *args, **kwargs):
        # Process the request data to limit decimal places in any JSON
        if 'json' in kwargs and kwargs['json'] is not None:
            kwargs['json'] = trim_float_precision(kwargs['json'], max_decimals=15)
        
        # Continue with the original request
        return await original_request(self, *args, **kwargs)
    
    # Apply the patch
    HttpClient.request = patched_request
    print("Applied precision control patch to OpenAI client request method")
except Exception as e:
    print(f"Warning: Could not patch OpenAI client request method: {e}")

# Try multiple possible module paths for file search tool
file_search_modules = [
    'agents.tools.file_search',  # Original attempt
    'agents.file_search',  # Simpler path
    'agents.tools.search.file_search',  # More nested path
    'src.agents.tools.file_search',  # With src prefix
    'src.agents.file_search',  # Another possibility
]

original_file_search_run = None
patched = False

for module_path in file_search_modules:
    try:
        module = __import__(module_path, fromlist=['file_search_run'])
        
        if hasattr(module, 'file_search_run'):
            original_file_search_run = module.file_search_run
            
            # Create a patched version that limits decimal places in scores
            async def patched_file_search_run(*args, **kwargs):
                """Patch the file search run function to limit decimal places in scores."""
                # Call the original function
                results = await original_file_search_run(*args, **kwargs)
                
                # Process the results to limit decimal places
                if isinstance(results, dict) and 'results' in results:
                    for result in results['results']:
                        if 'score' in result and isinstance(result['score'], float):
                            # Limit to 15 decimal places (below API's 16 limit)
                            result['score'] = round(result['score'], 15)
                            
                            # Extra safety check - convert to string and back
                            result['score'] = float(f"{result['score']:.15f}")
                            
                return results
            
            # Apply the patch
            module.file_search_run = patched_file_search_run
            print(f"Applied precision control patch to file search results via {module_path}")
            patched = True
            break
    except (ImportError, AttributeError) as e:
        continue

if not patched:
    print("Warning: Could not patch file search function - tried multiple module paths but none worked")
    
# Add reasoning parameter to o3-mini calls
original_create = Responses.create

async def patched_create(self, *args, **kwargs):
    """Add reasoning parameter for o3-mini model and sanitize any input data"""
    # First sanitize any input data to limit decimal places
    if len(args) > 0:
        args = list(args)
        args[0] = trim_float_precision(args[0], max_decimals=15)
        args = tuple(args)
    
    for key in kwargs:
        kwargs[key] = trim_float_precision(kwargs[key], max_decimals=15)
    
    # Add reasoning parameter
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
    print("Applied precision control for floating point values to prevent decimal place errors")
    
    # Run the CLI's main function
    import asyncio
    from ai_tutor.cli import main
    asyncio.run(main(args)) 