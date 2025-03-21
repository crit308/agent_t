import argparse
import asyncio
import json
import os
import sys
from typing import List

from ai_tutor.manager import AITutorManager

# Create parser outside of the __main__ block so it can be imported
parser = argparse.ArgumentParser(description="AI Tutor System")
parser.add_argument(
    "files", 
    nargs="+", 
    help="Paths to the files to upload for the lesson"
)
parser.add_argument(
    "--output", 
    "-o", 
    help="Path to save the lesson content to (JSON format)",
    default=None
)
parser.add_argument(
    "--api-key", 
    help="OpenAI API key", 
    default=os.environ.get("OPENAI_API_KEY")
)

async def main(file_paths: List[str], api_key: str, output_file: str = None):
    """Run the AI tutor with the provided files."""
    # Set API key in environment variables if provided
    if api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
        print(f"Set OPENAI_API_KEY environment variable from CLI argument")
    elif not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set!")
    
    # Create the AI tutor manager
    manager = AITutorManager(api_key)
    
    print("AI Tutor System")
    print("==============")
    
    # Run the full workflow
    print("\n1. Uploading documents...")
    try:
        upload_results = await manager.upload_documents(file_paths)
        print(upload_results)
    except Exception as e:
        print(f"Error uploading documents: {str(e)}")
        return
    
    print("\n2. Generating lesson plan...")
    try:
        lesson_plan = await manager.generate_lesson_plan()
        print(f"✓ Generated lesson plan: {lesson_plan.title}")
        print(f"   Description: {lesson_plan.description}")
        print(f"   Target audience: {lesson_plan.target_audience}")
        print(f"   Total duration: {lesson_plan.total_estimated_duration_minutes} minutes")
        print(f"   Sections: {len(lesson_plan.sections)}")
    except Exception as e:
        print(f"Error generating lesson plan: {str(e)}")
        return
    
    print("\n3. Creating lesson content...")
    try:
        lesson_content = await manager.generate_lesson()
        print(f"✓ Generated lesson content: {lesson_content.title}")
        print(f"   Sections: {len(lesson_content.sections)}")
        
        # Save the lesson content to a file if requested
        if output_file:
            with open(output_file, "w") as f:
                f.write(json.dumps(lesson_content.dict(), indent=2))
            print(f"\nLesson content saved to {output_file}")
        
        print("\nLesson content preview:")
        print(f"Title: {lesson_content.title}")
        print(f"Introduction: {lesson_content.introduction[:200]}...")
        print("\nSections:")
        for i, section in enumerate(lesson_content.sections):
            print(f"  {i+1}. {section.title}")
            
        print("\nView lesson trace: https://platform.openai.com/traces")
        
    except Exception as e:
        print(f"Error generating lesson content: {str(e)}")
        return


if __name__ == "__main__":
    args = parser.parse_args()
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: OpenAI API key is required. Provide it with --api-key or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Set API key in environment variables
    os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Check if files exist
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            sys.exit(1)
    
    # Run the main function
    asyncio.run(main(args.files, args.api_key, args.output)) 