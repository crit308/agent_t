import argparse
import asyncio
import json
import os
import sys
import time
from typing import List
import uuid

from ai_tutor.manager import AITutorManager
from ai_tutor.agents.analyzer_agent import analyze_documents
from ai_tutor.agents.models import Quiz, LessonPlan, QuizUserAnswer, QuizUserAnswers, SessionAnalysis, QuizFeedback
from ai_tutor.output_logger import get_logger

# Create parser outside of the __main__ block so it can be imported
parser = argparse.ArgumentParser(description="AI Tutor System")
# Add log level argument
parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level for SDK and related libraries."
)
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
tutor_parser.add_argument(
    "--skip-quiz",
    action="store_true",
    help="Skip the interactive quiz-taking and evaluation steps"
)
tutor_parser.add_argument(
    "--no-session-analysis",
    action="store_true",
    help="Skip running session analysis after the workflow is complete",
    default=False
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
    
    # Get the logger instance first
    logger = get_logger()
    
    # Ensure analyzer runs at the beginning
    args.auto_analyze = True
    print("Auto-analyze enabled: Analyzer agent will run at the start and generate Knowledge Base")
    
    # Create the AI tutor manager with auto-analyze option and the logger
    manager = AITutorManager(auto_analyze=False, output_logger=logger)  # Set auto_analyze to False to control the flow manually
    
    print("AI Tutor System")
    print("==============")

    # Run the full workflow
    print("\n1. Uploading documents...")
    try:
        # Record session start time
        session_start_time = time.time()
        
        try:
            upload_results = await manager.upload_documents(args.files)
            print(upload_results)
        except Exception as e:
            print(f"ERROR: Document upload failed in CLI: {e}")
            raise  # Re-raise to the outer try-except
    except Exception as e:
        print(f"Error uploading documents: {str(e)}")
        return
    
    # Always run the analyzer synchronously and wait for completion
    print("\n2. Analyzing documents...")
    try:
        # Run analyzer and wait for completion
        try:
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
        except Exception as e:
            print(f"ERROR: Document analysis failed in CLI: {e}")
            raise  # Re-raise to the outer try-except
    except Exception as e:
        print(f"Error analyzing documents: {str(e)}")
        return  # Exit if document analysis fails
    
    # Generate lesson plan with step number adjusted based on whether analysis was run
    step_num = 3
    print(f"\n{step_num}. Generating lesson plan...")
    try:
        try:
            lesson_plan = await manager.generate_lesson_plan()
            # Log the planner agent output
            logger.log_planner_output(lesson_plan)
            
            print(f"✓ Generated lesson plan: {lesson_plan.title}")
            print(f"   Description: {lesson_plan.description}")
            
            # Check if lesson_plan is actually a Quiz object
            if isinstance(lesson_plan, Quiz):
                print(f"   Quiz with {len(lesson_plan.questions)} questions")
                print(f"   Passing score: {lesson_plan.passing_score}/{lesson_plan.total_points}")
            # Only try to access target_audience if it's a LessonPlan
            elif isinstance(lesson_plan, LessonPlan) and hasattr(lesson_plan, 'target_audience'):
                print(f"   Target audience: {lesson_plan.target_audience}")
                print(f"   Total duration: {lesson_plan.total_estimated_duration_minutes} minutes")
                print(f"   Sections: {len(lesson_plan.sections)}")
            else:
                # Generic handling for other types
                if hasattr(lesson_plan, 'sections'):
                    print(f"   Sections: {len(lesson_plan.sections)}")
        except Exception as e:
            print(f"ERROR: Lesson plan generation failed in CLI: {e}")
            raise  # Re-raise to the outer try-except
    except Exception as e:
        print(f"Error generating lesson plan: {str(e)}")
        return

    # Create lesson content (simplified)
    step_num += 1
    print(f"\n{step_num}. Creating lesson content...")
    try:
        # generate_lesson_content now returns the simplified model
        lesson_content = await manager.generate_lesson_content()
        logger.log_teacher_output(lesson_content) # Log the simplified content

        print(f"✓ Generated lesson content: {lesson_content.title}")

        if args.output:
            # Saving remains the same, just outputs simpler JSON
            with open(args.output, "w") as f:
                f.write(json.dumps(lesson_content.model_dump(), indent=2))
            print(f"\nLesson content saved to {args.output}")

        # --- SIMPLIFIED PREVIEW ---
        print("\nLesson content preview:")
        print(f"Title: {lesson_content.title}")
        print(f"Text (first 300 chars): {lesson_content.text[:300]}...")
        print("\nView lesson trace: https://platform.openai.com/traces")
        # --- End SIMPLIFIED PREVIEW ---

    except Exception as e:
        print(f"Error creating lesson content: {str(e)}")
        return

    # --- REMOVE/SIMPLIFY INTERACTIVE LESSON DELIVERY ---
    step_num += 1
    print(f"\n{step_num}. Displaying Lesson...")
    print("="*60)
    print(f"LESSON: {lesson_content.title}")
    print("="*60)
    print("\nFull Lesson Text:")
    print(lesson_content.text) # Print the full text directly
    print("\n" + "="*60)
    print("Lesson display complete.")
    print("="*60)
    # --- End REMOVED/SIMPLIFIED INTERACTIVE LESSON DELIVERY ---

    # Skip quiz if requested
    if args.skip_quiz:
        print("\nSkipping interactive quiz...")
        logger.save()
        return

    # Quiz creation/taking/feedback/analysis logic remains largely the same,
    # as it depends on the Quiz and QuizFeedback models which haven't changed.
    # ... (Quiz steps) ...

    print("\nAI Tutor workflow complete!")
    print(f"Detailed logs have been saved to: {logger.output_file}")
    return 0

async def run_analyzer(args):
    """Run only the document analyzer."""
    print("Document Analyzer")
    print("================")
    
    # API key is expected to be set globally via main.py

    vector_store_id = args.vector_store_id
    
    # If files were provided but no vector store ID, upload the files first
    if args.files and not vector_store_id:
        print("\n1. Uploading documents to create vector store...")
        manager = AITutorManager()
        try:
            upload_results = await manager.upload_documents(args.files)
            print(upload_results)
            vector_store_id = manager.vector_store_id
        except Exception as e:
            print(f"Error uploading documents: {str(e)}")
            return
    
    # If no vector store ID, and watching is enabled, watch for new vector stores
    if not vector_store_id and args.watch:
        print("\nWatching for vector store creation...")
        # Import and use the vector store watcher
        from ai_tutor.run_analyzer import VectorStoreWatcher
        watcher = VectorStoreWatcher()
        await watcher.analyze_on_vector_store_creation()
        return
    
    # If vector store ID is available, analyze the documents
    if vector_store_id:
        print(f"\nAnalyzing vector store: {vector_store_id}")
        try:
            try:
                analysis = await analyze_documents(vector_store_id)
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
                print(f"ERROR: Analyzer agent failed in CLI: {e}")
                # Since we're already in a try-except, no need to re-raise
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
    
    # Check if files exist when required
    if args.command == "tutor" or (args.command == "analyze" and args.files):
        files_to_check = args.files
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist.")
                sys.exit(1)
    
    # Run the main function
    asyncio.run(main(args)) 