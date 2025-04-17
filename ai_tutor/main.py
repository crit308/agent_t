import os
import sys
import logging # Import standard logging
from typing import List, Optional

from dotenv import load_dotenv # Import dotenv
load_dotenv() # Load environment variables from .env file at the start

import asyncio # Import asyncio
# Add project src directory to sys.path so local 'agents' overrides pip-installed package
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import SDK configuration functions directly from 'agents'
from agents import (
    set_default_openai_key, set_default_openai_api, set_default_openai_client
)
from ai_tutor.output_logger import get_logger

# Keep a reference to the original methods
import json as _stdlib_json


if __name__ == "__main__":
    # Import the parser object defined in cli.py
    from ai_tutor.cli import parser
    
    # Parse command-line args
    args = parser.parse_args()
    
    # Set logger level
    for logger_name in ["openai", "openai_agents", "httpx"]:
        # Use standard logging configuration
        logging.getLogger(logger_name).setLevel(args.log_level.upper())
    
    # SDK Tracing is configured via RunConfig or globally via agents.set_tracing_... functions
    
    # --- SDK Configuration ---
    # Ensure API key is available
    api_key_to_use = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key_to_use:
        print("ERROR: OpenAI API key is required. Provide it via --api-key or OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # CRITICAL: Set environment variable BEFORE any direct openai.Client() calls (e.g., in file_upload.py)
    os.environ["OPENAI_API_KEY"] = api_key_to_use
    # ALSO set the SDK default key for agents.set_default_... functions
    set_default_openai_key(api_key_to_use)
    # If using a separate tracing key:
    # set_tracing_export_api_key("YOUR_TRACING_KEY")
    print("Set default OpenAI API key for SDK and environment.")
    
    # Ensure we're using the Responses API for o4-mini model
    set_default_openai_api("responses")
    # If needed, configure a custom client:
    # from openai import AsyncOpenAI
    # set_default_openai_client(AsyncOpenAI(api_key=api_key_to_use, ...))

    print("Precision control now handled by selective rounding when needed, global JSON patch removed.")

    if args.command == 'tutor':
        # Define a default location for output log
        output_path = os.path.join(os.getcwd(), f"ai_tutor_session_{os.path.basename(args.files[0])}.log")
        # But allow command-line arg to override
        if args.output:
            output_path = args.output
        
        # Import and run the main async function from cli.py
        from ai_tutor.cli import main as cli_main
        asyncio.run(cli_main(args))
    elif args.command == 'explain':
        from ai_tutor.explain import explain_code
        explain_code(args.files[0], args.verbose)
    # The 'analyze' command is handled similarly via cli.py's main function
    elif args.command == 'analyze':
        from ai_tutor.cli import main as cli_main
        asyncio.run(cli_main(args))
    else: # Should ideally be caught by argparse, but good to have a fallback
        print(f"Unknown command received in main.py: {args.command}")
        sys.exit(1) 