import argparse
import asyncio
import json
import os
import sys
from typing import List

from ai_tutor.manager import AITutorManager
from ai_tutor.agents.analyzer_agent import analyze_documents

# Create parser outside of the __main__ block so it can be imported
parser = argparse.ArgumentParser(description="AI Tutor System")
subparsers = parser.add_subparsers(dest="command", help="Command to run")

# Full AI tutor workflow command
tutor_parser = subparsers.add_parser("tutor", help="Run the full AI tutor workflow")
tutor_parser.add_argument(
    "files", 
    nargs="+", 
    help="Paths to the files to upload for the lesson"
)
tutor_parser.add_argument(
    "--output", 
    "-o", 
    help="Path to save the lesson content to (JSON format)",
    default=None
)
tutor_parser.add_argument(
    "--api-key", 
    help="OpenAI API key", 
    default=os.environ.get("OPENAI_API_KEY")
)
tutor_parser.add_argument(
    "--analyze",
    "-a",
    action="store_true",
    help="Run document analysis as part of the workflow"
)
tutor_parser.add_argument(
    "--auto-analyze",
    action="store_true",
    help="Automatically run document analysis when documents are uploaded"
)
tutor_parser.add_argument(
    "--run-analyzer-sync",
    action="store_true",
    help="Run the analyzer synchronously (waits for completion) instead of in the background"
)

# Document analyzer command
analyzer_parser = subparsers.add_parser("analyze", help="Run only the document analyzer")
analyzer_parser.add_argument(
    "--vector-store-id",
    help="Vector store ID to analyze. If not provided, will wait for a vector store to be created"
)
analyzer_parser.add_argument(
    "--output", 
    "-o", 
    help="Path to save the analysis to (JSON format)",
    default=None
)
analyzer_parser.add_argument(
    "--api-key", 
    help="OpenAI API key", 
    default=os.environ.get("OPENAI_API_KEY")
)
analyzer_parser.add_argument(
    "--watch",
    "-w",
    action="store_true",
    help="Watch for vector store creation if none exists"
)
analyzer_parser.add_argument(
    "files", 
    nargs="*", 
    help="Paths to the files to upload (if not using an existing vector store)"
)

async def run_tutor(args):
    """Run the AI tutor with the provided files."""
    # Set API key in environment variables if provided
    if args.api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = args.api_key
        print(f"Set OPENAI_API_KEY environment variable from CLI argument")
    elif not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set!")
    
    # Create the AI tutor manager with auto-analyze option
    manager = AITutorManager(args.api_key, auto_analyze=args.auto_analyze)
    
    print("AI Tutor System")
    print("==============")
    
    # Run the full workflow
    print("\n1. Uploading documents...")
    try:
        upload_results = await manager.upload_documents(args.files)
        print(upload_results)
    except Exception as e:
        print(f"Error uploading documents: {str(e)}")
        return
    
    # Run the analyzer if requested and not already run via auto-analyze
    run_synchronously = args.run_analyzer_sync
    if args.analyze and not manager.document_analysis:
        print("\n2. Analyzing documents...")
        try:
            if run_synchronously:
                # Run analyzer and wait for completion
                analysis = await manager.analyze_documents(run_in_background=False)
                if analysis:
                    print(f"✓ Document analysis complete")
                    # Extract metadata if possible
                    file_count = len(getattr(analysis, "file_names", []))
                    concept_count = len(getattr(analysis, "key_concepts", []))
                    term_count = len(getattr(analysis, "key_terms", {}))
                    print(f"   Files analyzed: {file_count}")
                    print(f"   Key concepts: {concept_count}")
                    print(f"   Key terms: {term_count}")
                    
                    # Save analysis to file if output specified
                    if args.output:
                        analysis_output = f"{os.path.splitext(args.output)[0]}_analysis.txt"
                        try:
                            with open(analysis_output, "w", encoding="utf-8") as f:
                                f.write(analysis if isinstance(analysis, str) else str(analysis))
                            print(f"   Analysis saved to {analysis_output}")
                        except Exception as e:
                            print(f"   Error saving analysis: {e}")
                            # Try with fallback encoding
                            try:
                                with open(analysis_output, "w", encoding="ascii", errors="ignore") as f:
                                    f.write(analysis if isinstance(analysis, str) else str(analysis))
                                print(f"   Analysis saved to {analysis_output} (with encoding fallback)")
                            except Exception as e2:
                                print(f"   Could not save analysis: {e2}")
                else:
                    print("✗ Document analysis failed")
            else:
                # Start analyzer in background
                await manager.analyze_documents(run_in_background=True)
                print(f"✓ Document analysis started in background")
                print(f"   Analysis will run in parallel with lesson generation")
        except Exception as e:
            print(f"Error analyzing documents: {str(e)}")
    
    # Generate lesson plan with step number adjusted based on whether analysis was run
    step_num = 3 if (args.analyze or args.auto_analyze) else 2
    print(f"\n{step_num}. Generating lesson plan...")
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
    
    # Create lesson content with step number adjusted
    step_num += 1
    print(f"\n{step_num}. Creating lesson content...")
    try:
        lesson_content = await manager.generate_lesson()
        print(f"✓ Generated lesson content: {lesson_content.title}")
        print(f"   Sections: {len(lesson_content.sections)}")
        
        # Save the lesson content to a file if requested
        if args.output:
            with open(args.output, "w") as f:
                f.write(json.dumps(lesson_content.model_dump(), indent=2))
            print(f"\nLesson content saved to {args.output}")
        
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

    # Try to wait for analysis results if they weren't available earlier
    if args.analyze and not run_synchronously and not manager.document_analysis:
        analysis = await manager.wait_for_analysis(timeout=0.1)
        if analysis:
            print("\nBackground analysis results are now available:")
            # Extract metadata if possible
            file_count = len(getattr(analysis, "file_names", []))
            concept_count = len(getattr(analysis, "key_concepts", []))
            term_count = len(getattr(analysis, "key_terms", {}))
            print(f"   Files analyzed: {file_count}")
            print(f"   Key concepts: {concept_count}")
            print(f"   Key terms: {term_count}")
            
            # Save analysis to file if output specified
            if args.output:
                analysis_output = f"{os.path.splitext(args.output)[0]}_analysis.txt"
                try:
                    with open(analysis_output, "w", encoding="utf-8") as f:
                        f.write(analysis if isinstance(analysis, str) else str(analysis))
                    print(f"   Analysis saved to {analysis_output}")
                except Exception as e:
                    print(f"   Error saving analysis: {e}")
                    # Try with fallback encoding
                    try:
                        with open(analysis_output, "w", encoding="ascii", errors="ignore") as f:
                            f.write(analysis if isinstance(analysis, str) else str(analysis))
                        print(f"   Analysis saved to {analysis_output} (with encoding fallback)")
                    except Exception as e2:
                        print(f"   Could not save analysis: {e2}")

async def run_analyzer(args):
    """Run only the document analyzer."""
    # Set API key in environment variables if provided
    if args.api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = args.api_key
        print(f"Set OPENAI_API_KEY environment variable from CLI argument")
    elif not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set!")
        return
    
    print("Document Analyzer")
    print("================")
    
    vector_store_id = args.vector_store_id
    
    # If files were provided but no vector store ID, upload the files first
    if not vector_store_id and args.files:
        print("\n1. Uploading documents to create vector store...")
        manager = AITutorManager(args.api_key)
        try:
            upload_results = await manager.upload_documents(args.files)
            print(upload_results)
            vector_store_id = manager.vector_store_id
        except Exception as e:
            print(f"Error uploading documents: {str(e)}")
            return
    
    # If watch mode and no vector store ID, wait for one to be created
    if not vector_store_id and args.watch:
        print("\nWatching for vector store creation...")
        # Import and use the vector store watcher
        from ai_tutor.run_analyzer import VectorStoreWatcher
        watcher = VectorStoreWatcher(args.api_key)
        await watcher.analyze_on_vector_store_creation()
        return
    
    # If we have a vector store ID, analyze it
    if vector_store_id:
        print(f"\nAnalyzing vector store: {vector_store_id}")
        try:
            analysis = await analyze_documents(vector_store_id, args.api_key)
            if analysis:
                print(f"\n✓ Document analysis complete")
                # Extract metadata if possible
                file_count = len(getattr(analysis, "file_names", []))
                concept_count = len(getattr(analysis, "key_concepts", []))
                term_count = len(getattr(analysis, "key_terms", {}))
                key_concepts = getattr(analysis, "key_concepts", [])
                
                print(f"   Files analyzed: {file_count}")
                print(f"   Key concepts identified: {concept_count}")
                if key_concepts and len(key_concepts) > 0:
                    print(f"   Top concepts: {', '.join(key_concepts[:5])}")
                print(f"   Vector store ID: {vector_store_id}")
                
                # Save analysis to file if output specified
                output_file = args.output or f"document_analysis_{vector_store_id}.txt"
                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(analysis if isinstance(analysis, str) else str(analysis))
                    print(f"\nAnalysis saved to {output_file}")
                except Exception as e:
                    print(f"\nError saving analysis: {e}")
                    # Try with fallback encoding
                    try:
                        with open(output_file, "w", encoding="ascii", errors="ignore") as f:
                            f.write(analysis if isinstance(analysis, str) else str(analysis))
                        print(f"Analysis saved to {output_file} (with encoding fallback)")
                    except Exception as e2:
                        print(f"Could not save analysis: {e2}")
            else:
                print("✗ Document analysis failed")
        except Exception as e:
            print(f"Error analyzing documents: {str(e)}")
    else:
        print("ERROR: No vector store ID provided or created. Use --vector-store-id or provide files to upload.")

async def main(args):
    """Run the appropriate command based on arguments."""
    if args.command == "analyze":
        await run_analyzer(args)
    else:  # Default to tutor command
        await run_tutor(args)


if __name__ == "__main__":
    args = parser.parse_args()
    
    # Default to tutor command if none specified
    if not args.command:
        args.command = "tutor"
    
    # Check if API key is provided
    if not args.api_key:
        print("Error: OpenAI API key is required. Provide it with --api-key or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Set API key in environment variables
    os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Check if files exist when required
    if args.command == "tutor" or (args.command == "analyze" and args.files):
        files_to_check = args.files
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist.")
                sys.exit(1)
    
    # Run the main function
    asyncio.run(main(args)) 