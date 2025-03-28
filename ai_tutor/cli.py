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
    
    # Create lesson content with step number adjusted
    step_num += 1
    print(f"\n{step_num}. Creating lesson content...")
    try:
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
            
            # Show a brief preview instead of full content
            print("\nLesson content preview:")
            print(f"Title: {lesson_content.title}")
            print(f"Introduction: {lesson_content.introduction[:200]}...")
            print("\nSections:")
            for i, section in enumerate(lesson_content.sections):
                print(f"  {i+1}. {section.title}")
            
            print("\nView lesson trace: https://platform.openai.com/traces")
        except Exception as e:
            print(f"ERROR: Lesson content generation failed in CLI: {e}")
            raise  # Re-raise to the outer try-except
    except Exception as e:
        print(f"Error creating lesson content: {str(e)}")
        return
    
    # Skip quiz if requested
    if args.skip_quiz:
        print("\nSkipping interactive quiz...")
        # Save the session log before exiting
        logger.save()
        return
    
    # --- NEW INTERACTIVE LESSON DELIVERY STEP ---
    step_num += 1
    print(f"\n{step_num}. Starting Interactive Lesson...")
    print("="*60)
    print(f"LESSON: {lesson_content.title}")
    print("="*60)
    print(f"\nIntroduction: {lesson_content.introduction}\n")
    input("Press Enter to begin the first section...")

    for i, section in enumerate(lesson_content.sections):
        print(f"\n--- Section {i+1}: {section.title} ---")
        print(f"\n{section.introduction}\n")

        if section.explanations:
            for j, explanation in enumerate(section.explanations):
                print(f"\n>> Concept: {explanation.topic}")
                print(f"\n{explanation.explanation}")

                if explanation.examples:
                    print("\nExamples:")
                    for ex in explanation.examples:
                        print(f"- {ex}")

                # --- MINI-QUIZ LOGIC ---
                if explanation.mini_quiz:
                    print("\n\n>>> Quick Check! <<<")
                    for k, mini_q in enumerate(explanation.mini_quiz):
                        print(f"\nMini-Question {k+1}: {mini_q.question}")
                        for opt_idx, option in enumerate(mini_q.options):
                            print(f"  {opt_idx+1}. {option}")

                        valid_mini_answer = False
                        while not valid_mini_answer:
                            try:
                                mini_answer_str = input(f"  Your answer (1-{len(mini_q.options)}): ")
                                selected_mini_option = int(mini_answer_str) - 1
                                if 0 <= selected_mini_option < len(mini_q.options):
                                    valid_mini_answer = True
                                else:
                                    print(f"  Please enter a number between 1 and {len(mini_q.options)}.")
                            except ValueError:
                                print("  Please enter a valid number.")

                        is_correct = (selected_mini_option == mini_q.correct_answer_index)
                        user_choice = mini_q.options[selected_mini_option]
                        correct_choice = mini_q.options[mini_q.correct_answer_index]

                        if is_correct:
                            print(f"  ✓ Correct!")
                        else:
                            print(f"  ✗ Incorrect. The correct answer was: {correct_choice}")
                        print(f"  Explanation: {mini_q.explanation}")
                        logger.log_mini_quiz_attempt(mini_q.question, user_choice, correct_choice, is_correct)
                    print("\n>>> End Quick Check <<<\n")
                # --- END MINI-QUIZ LOGIC ---

                # Prompt to continue after explanation/mini-quiz
                input("Press Enter to continue...")

        # Optionally, add interaction for exercises later
        if section.exercises:
            print("\nExercises for this section (solutions provided by agent):")
            for ex in section.exercises:
                print(f"- {ex.question} ({ex.difficulty_level})")

        print(f"\nSection Summary: {section.summary}")
        print(f"\n--- End of Section {i+1} ---")
        if i < len(lesson_content.sections) - 1:
            input("Press Enter for the next section...")

    print(f"\nLesson Conclusion: {lesson_content.conclusion}")
    print("\n" + "="*60)
    print("Lesson delivery complete.")
    print("="*60)
    # --- END INTERACTIVE LESSON DELIVERY STEP ---
    
    # Create and take quiz
    step_num += 1
    print(f"\n{step_num}. Creating quiz...")
    try:
        # Check if quiz may already be generated from handoff chain
        if hasattr(manager, 'quiz') and manager.quiz and hasattr(manager.quiz, 'questions') and len(manager.quiz.questions) > 0:
            quiz = manager.quiz
            print(f"✓ Using quiz that was already generated via handoff chain: {quiz.title}")
        # Quiz may already be generated if lesson plan was a Quiz
        elif isinstance(lesson_plan, Quiz):
            quiz = lesson_plan
            print(f"✓ Quiz already generated as part of lesson plan")
        else:
            try:
                print("\nCreating quiz...")
                quiz = await manager.generate_quiz()
                print(f"✓ Generated quiz: {quiz.title}")
            except Exception as e:
                print(f"ERROR: Quiz generation failed in CLI: {e}")
                raise  # Re-raise to the outer try-except
        
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
        try:
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
                    if topic.strip():  # Only print non-empty topics
                        print(f"- {topic}")
            
            if quiz_feedback.next_steps:
                print("\nRecommended Next Steps:")
                for step in quiz_feedback.next_steps:
                    print(f"- {step}")
            
            # Run session analysis if requested
            if not args.no_session_analysis and manager.lesson_content and manager.quiz and manager.quiz_feedback:
                step_num += 1
                print(f"\n{step_num}. Running session analysis...")
                try:
                    # Calculate session duration
                    session_duration = int(time.time() - session_start_time)
                    
                    # Run session analysis
                    try:
                        session_analysis = await manager.analyze_session(session_duration)
                        
                        # Log the session analysis output only if it's not None
                        if session_analysis:
                            logger.log_session_analysis_output(session_analysis)
                            
                            print(f"✓ Session analysis complete")
                            print(f"   Overall effectiveness: {session_analysis.overall_effectiveness:.2f}/5.0")
                            print(f"   Identified {len(session_analysis.strengths)} strengths and {len(session_analysis.improvement_areas)} areas for improvement")
                            print(f"   Session analysis has been added to the Knowledge Base")
                        else:
                            print(f"✓ Session analysis process ran but no results were generated")
                    except Exception as e:
                        print(f"ERROR: Session analysis failed in CLI: {e}")
                        raise  # Re-raise to the outer try-except
                except Exception as e:
                    print(f"Error running session analysis: {str(e)}")
                
                # Save the session log
                output_file = logger.save()
                print(f"\nAI Tutor session log saved to: {output_file}")
        except Exception as e:
            print(f"ERROR: Quiz feedback generation failed in CLI: {e}")
            raise  # Re-raise to the outer try-except
            
    except Exception as e:
        print(f"Error in quiz workflow: {str(e)}")
        # Save the session log even if there was an error
        logger.save()
        return
    
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