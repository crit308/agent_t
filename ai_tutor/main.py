import os
import sys

# Add the parent directory to the Python path so we can import the AI tutor package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the CLI module
from ai_tutor.cli import main

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
    
    # Run the CLI's main function
    import asyncio
    asyncio.run(main(args.files, args.api_key, args.output)) 