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
from ai_tutor.agents.models import Quiz, LessonPlan, QuizUserAnswer, QuizUserAnswers
from ai_tutor.output_logger import get_logger

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
tutor_parser.add_argument(
    "--skip-quiz",
    action="store_true",
    help="Skip the interactive quiz-taking and evaluation steps"
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
    
    # Get the logger instance
    logger = get_logger()
    
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
        print(f"Error generating lesson plan: {str(e)}")
        return
    
    # Create lesson content with step number adjusted
    step_num += 1
    print(f"\n{step_num}. Creating lesson content...")
    try:
        lesson_content = await manager.generate_lesson_content()
        # Log the teacher agent output
        logger.log_teacher_output(lesson_content)
        
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
        print(f"Error creating lesson content: {str(e)}")
        return
    
    # Skip quiz if requested
    if args.skip_quiz:
        print("\nSkipping interactive quiz...")
        # Save the session log before exiting
        logger.save()
        return
    
    # Create and take quiz
    step_num += 1
    print(f"\n{step_num}. Creating quiz...")
    try:
        # Quiz may already be generated if lesson plan was a Quiz
        if isinstance(lesson_plan, Quiz):
            quiz = lesson_plan
            print(f"✓ Quiz already generated as part of lesson plan")
        else:
            quiz = await manager.generate_quiz()
            print(f"✓ Generated quiz: {quiz.title}")
        
        # Log the quiz creator output
        logger.log_quiz_creator_output(quiz)
        
        print(f"   Questions: {len(quiz.questions)}")
        print(f"   Passing score: {quiz.passing_score}/{quiz.total_points}")
        
        # Interactive quiz section
        print("\n" + "="*60)
        print(f"QUIZ: {quiz.title}")
        print(f"Description: {quiz.description}")
        print(f"Total Questions: {len(quiz.questions)}")
        print(f"Passing Score: {quiz.passing_score}/{quiz.total_points}")
        print(f"Estimated Time: {quiz.estimated_completion_time_minutes} minutes")
        print("="*60)
        
        print("\nYou will now take the quiz. For each question, select the option number (1, 2, 3, etc.)")
        print("Press Enter to start the quiz...\n")
        input()
        
        # Record quiz answers
        user_answers = []
        quiz_start_time = time.time()
        
        for i, question in enumerate(quiz.questions):
            print(f"\nQuestion {i+1}: {question.question}")
            print(f"Difficulty: {question.difficulty}")
            print("\nOptions:")
            for j, option in enumerate(question.options):
                print(f"{j+1}. {option}")
            
            # Get user's answer with input validation
            question_start_time = time.time()
            valid_answer = False
            while not valid_answer:
                try:
                    answer_str = input(f"\nYour answer (1-{len(question.options)}): ")
                    selected_option = int(answer_str) - 1  # Convert to 0-based index
                    if 0 <= selected_option < len(question.options):
                        valid_answer = True
                    else:
                        print(f"Please enter a number between 1 and {len(question.options)}")
                except ValueError:
                    print("Please enter a valid number")
            
            # Calculate time taken for this question
            question_end_time = time.time()
            time_taken = int(question_end_time - question_start_time)
            
            # Record the answer
            user_answers.append(QuizUserAnswer(
                question_index=i,
                selected_option_index=selected_option,
                time_taken_seconds=time_taken
            ))
            
            # Log the user answer
            logger.log_quiz_user_answer(
                question=question.question,
                options=question.options,
                selected_idx=selected_option,
                correct_idx=question.correct_answer_index
            )
            
            print(f"Answer recorded: Option {selected_option + 1}. Time taken: {time_taken} seconds")
        
        # Calculate total quiz time
        quiz_end_time = time.time()
        total_time = int(quiz_end_time - quiz_start_time)
        
        # Create the user answers object
        quiz_user_answers = QuizUserAnswers(
            quiz_title=quiz.title,
            user_answers=user_answers,
            total_time_taken_seconds=total_time
        )
        
        # Submit answers for evaluation
        step_num += 1
        print(f"\n{step_num}. Evaluating quiz answers...")
        quiz_feedback = await manager.submit_quiz_answers(quiz_user_answers)
        
        # Log the quiz teacher output
        logger.log_quiz_teacher_output(quiz_feedback)
        
        # Display the feedback
        print("\n" + "="*60)
        print(f"QUIZ RESULTS: {quiz_feedback.quiz_title}")
        print(f"Score: {quiz_feedback.correct_answers}/{quiz_feedback.total_questions} ({quiz_feedback.score_percentage:.1f}%)")
        print(f"Result: {'PASS' if quiz_feedback.passed else 'FAIL'}")
        print("="*60)
        
        print("\nFeedback by Question:")
        for i, feedback in enumerate(quiz_feedback.feedback_items):
            print(f"\nQuestion {i+1}: {feedback.question_text}")
            print(f"Your answer: {feedback.user_selected_option}")
            print(f"Correct answer: {feedback.correct_option}")
            print(f"Result: {'✓ Correct' if feedback.is_correct else '✗ Incorrect'}")
            if not feedback.is_correct:
                print(f"Explanation: {feedback.explanation}")
        
        print("\nOverall Feedback:")
        print(quiz_feedback.overall_feedback)
        
        if quiz_feedback.suggested_study_topics:
            print("\nAreas for Improvement/Suggested Study Topics:")
            for topic in quiz_feedback.suggested_study_topics:
                print(f"- {topic}")
        
        if quiz_feedback.next_steps:
            print("\nRecommended Next Steps:")
            for step in quiz_feedback.next_steps:
                print(f"- {step}")
                
        # Save the session log
        output_file = logger.save()
        print(f"\nAI Tutor session log saved to: {output_file}")
        
    except Exception as e:
        print(f"Error in quiz workflow: {str(e)}")
        # Save the session log even if there was an error
        logger.save()
        return

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