import os
import sys
import logging # Import standard logging
from typing import List, Optional

# Add this directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SDK configuration functions directly from 'agents'
from agents import (
    set_default_openai_key, set_default_openai_api, set_default_openai_client
)
from ai_tutor.output_logger import get_logger

# Keep a reference to the original methods
import json as _stdlib_json


if __name__ == "__main__":
    # Import the CLI's parser
    from ai_tutor.cli_parser import parse_args
    
    # Parse command-line args
    args = parse_args()
    
    # Set logger level
    for logger_name in ["openai", "openai_agents", "httpx"]:
        # Use standard logging configuration
        logging.getLogger(logger_name).setLevel(args.log_level.upper())
    
    # SDK Tracing is configured via RunConfig or globally via agents.set_tracing_... functions
    
    # Set default API key to use for all calls (or tries env vars)
    set_default_openai_key(args.api_key)
    # If using a separate tracing key:
    # set_tracing_export_api_key("YOUR_TRACING_KEY")
    print("Set default OpenAI API key for SDK.")
    
    # Ensure we're using the Responses API for o3-mini model
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
        
        # Start tracing
        # set_tracing_enabled(True)
        from ai_tutor.tutor import start_ai_tutoring
        # Start the AI tutoring session (will run until completion)
        start_ai_tutoring(
            filename=args.files[0],
            output_log_file=output_path,
            topic=args.topic,
            learning_level=args.learning_level,
            students_per_group=args.students,
            max_quiz_questions=args.max_quiz_questions,
            verbose=args.verbose
        )
    elif args.command == 'explain':
        from ai_tutor.explain import explain_code
        explain_code(args.files[0], args.verbose)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1) 