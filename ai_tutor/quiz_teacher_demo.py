#!/usr/bin/env python
import asyncio
import os
import sys
import argparse
import time
from typing import List
import pathlib

from ai_tutor.manager import AITutorManager
from ai_tutor.agents.models import QuizUserAnswers, QuizUserAnswer


async def run_quiz_teacher_demo(file_paths: List[str], api_key: str = None):
    """Run a demonstration of the AI Tutor Quiz Teacher functionality.
    
    This demonstrates the complete workflow from document upload to quiz teaching.
    """
    # Get API key from environment if not provided
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: No OpenAI API key provided. Please set OPENAI_API_KEY environment variable.")
            sys.exit(1)
    
    print(f"=== AI Tutor Quiz Teacher Demo ===")
    
    # Validate file paths
    valid_files = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            # Create a sample file if it doesn't exist (e.g., sample.txt)
            if os.path.basename(file_path).lower() == "sample.txt":
                print(f"Creating a sample text file: {file_path}")
                with open(file_path, "w") as f:
                    f.write("This is a sample text file created for testing the AI Tutor Quiz Teacher.\n\n")
                    f.write("The Quiz Teacher is an agent that evaluates user answers to quiz questions and provides detailed feedback.\n")
                    f.write("It analyzes which answers are correct or incorrect, explains why, and offers suggestions for improvement.\n")
                    f.write("The system generates personalized learning recommendations based on the pattern of errors.\n\n")
                    f.write("Key features of the Quiz Teacher include:\n")
                    f.write("1. Detailed feedback for each question\n")
                    f.write("2. Overall score calculation\n")
                    f.write("3. Pass/fail determination\n")
                    f.write("4. Suggested study topics\n")
                    f.write("5. Recommended next steps for learning\n")
                valid_files.append(file_path)
        else:
            valid_files.append(file_path)
    
    if not valid_files:
        print("ERROR: No valid files to process. Please provide valid file paths.")
        sys.exit(1)
    
    print(f"Using files: {', '.join(valid_files)}")
    print("Set OPENAI_API_KEY environment variable for API and tracing")
    
    # Initialize the AI Tutor manager
    manager = AITutorManager(api_key)
    
    try:
        # Run the workflow until quiz creation
        print("\nRunning full workflow with quiz teacher...")
        
        # Upload the documents
        upload_result = await manager.upload_documents(valid_files)
        print(f"Upload result: {upload_result}")
        
        # Generate the lesson plan
        print("Generating lesson plan and starting potential handoff chain...")
        trace_id = manager.get_trace_id()
        if trace_id:
            print(f"Generating lesson plan with trace ID: {trace_id}")
            print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
        lesson_plan = await manager.generate_lesson_plan()
        
        # Generate the lesson content
        print("Generating lesson content separately since handoff chain didn't complete...")
        trace_id = manager.get_trace_id()
        if trace_id:
            print(f"Generating lesson content with trace ID: {trace_id}")
            print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
        lesson_content = await manager.generate_lesson_content()
        
        # Generate the quiz
        quiz = await manager.generate_quiz(enable_teacher_handoff=True)
        
        # Debug info and safe check for the quiz
        if quiz:
            print(f"Quiz received from manager: {quiz.title}")
            if hasattr(quiz, 'questions'):
                print(f"Quiz questions count: {len(quiz.questions)}")
                
                # Double-check that the quiz really has questions
                if len(quiz.questions) == 0:
                    print("Warning: Quiz has 0 questions despite successful creation")
                    # Create a simple test question for demo purposes
                    from ai_tutor.agents.models import QuizQuestion
                    quiz.questions.append(
                        QuizQuestion(
                            question="What is the main purpose of the Quiz Teacher agent?",
                            options=[
                                "To create lesson plans", 
                                "To evaluate quiz answers and provide feedback", 
                                "To generate quizzes", 
                                "To analyze documents"
                            ],
                            correct_answer_index=1,
                            explanation="The Quiz Teacher agent evaluates user answers and provides detailed feedback on performance.",
                            difficulty="Medium",
                            related_section="Quiz Teacher Features"
                        )
                    )
                    quiz.passing_score = 1
                    quiz.total_points = 1
                    print(f"Added a test question to the quiz. Now has {len(quiz.questions)} questions.")
            else:
                print("Warning: Quiz object doesn't have 'questions' attribute")
                # Create a new Quiz object with a test question
                from ai_tutor.agents.models import Quiz, QuizQuestion
                quiz.questions = [
                    QuizQuestion(
                        question="What is the main purpose of the Quiz Teacher agent?",
                        options=[
                            "To create lesson plans", 
                            "To evaluate quiz answers and provide feedback", 
                            "To generate quizzes", 
                            "To analyze documents"
                        ],
                        correct_answer_index=1,
                        explanation="The Quiz Teacher agent evaluates user answers and provides detailed feedback on performance.",
                        difficulty="Medium",
                        related_section="Quiz Teacher Features"
                    )
                ]
                quiz.passing_score = 1
                quiz.total_points = 1
                print(f"Added questions attribute to quiz with a test question")
        else:
            print("Quiz is None after generation. Creating a minimal quiz.")
            # Create a minimal quiz
            from ai_tutor.agents.models import Quiz, QuizQuestion
            quiz = Quiz(
                title="Quiz Teacher Demo Quiz",
                description="A quiz for demonstrating the quiz teacher functionality",
                lesson_title=lesson_content.title if lesson_content else "Demo Lesson",
                questions=[
                    QuizQuestion(
                        question="What is the main purpose of the Quiz Teacher agent?",
                        options=[
                            "To create lesson plans", 
                            "To evaluate quiz answers and provide feedback", 
                            "To generate quizzes", 
                            "To analyze documents"
                        ],
                        correct_answer_index=1,
                        explanation="The Quiz Teacher agent evaluates user answers and provides detailed feedback on performance.",
                        difficulty="Medium",
                        related_section="Quiz Teacher Features"
                    )
                ],
                passing_score=1,
                total_points=1,
                estimated_completion_time_minutes=5
            )
            # Set the quiz in the manager for future use
            manager.quiz = quiz
            
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
        
        print("\nProcessing your answers...")
        trace_id = manager.get_trace_id()
        if trace_id:
            print(f"Processing quiz answers with trace ID: {trace_id}")
            print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
        
        # Generate feedback on the answers
        quiz_feedback = await manager.submit_quiz_answers(quiz_user_answers)
        
        # Print results
        print("\n" + "="*60)
        print("=== Quiz Results ===")
        print(f"Lesson: {lesson_content.title}")
        print(f"Quiz: {quiz.title} ({len(quiz.questions)} questions)")
        print(f"Your Score: {quiz_feedback.correct_answers}/{quiz_feedback.total_questions} ({quiz_feedback.score_percentage:.1f}%)")
        print(f"Pass/Fail: {'Passed' if quiz_feedback.passed else 'Failed'}")
        print(f"Total Time: {quiz_feedback.total_time_taken_seconds} seconds")
        
        # Print detailed feedback for each question
        print("\n=== Detailed Feedback ===")
        for item in quiz_feedback.feedback_items:
            question_index = item.question_index
            print(f"\nQuestion {question_index + 1}: {item.question_text}")
            print(f"Your answer: {item.user_selected_option}")
            print(f"Correct: {'Yes' if item.is_correct else 'No'}")
            if not item.is_correct:
                print(f"Correct answer: {item.correct_option}")
            print(f"Explanation: {item.explanation}")
            if not item.is_correct and item.improvement_suggestion:
                print(f"Improvement suggestion: {item.improvement_suggestion}")
        
        # Print overall feedback
        print(f"\n=== Overall Feedback ===")
        print(quiz_feedback.overall_feedback)
        
        # Print suggested study topics
        print("\n=== Suggested Study Topics ===")
        if quiz_feedback.suggested_study_topics and quiz_feedback.suggested_study_topics[0] != "":
            for topic in quiz_feedback.suggested_study_topics:
                print(f"- {topic}")
        else:
            print("No specific study topics suggested.")
        
        # Print next steps
        print("\n=== Recommended Next Steps ===")
        for step in quiz_feedback.next_steps:
            print(f"- {step}")
        
        print("\n=== Demo Complete ===")
        result = {
            "lesson_plan": lesson_plan,
            "lesson_content": lesson_content,
            "quiz": quiz,
            "user_answers": quiz_user_answers,
            "quiz_feedback": quiz_feedback
        }
        return result
    
    except Exception as e:
        print(f"\nERROR: An error occurred during the demo: {e}")
        import traceback
        traceback.print_exc()
        print("\nThe demo encountered an error but attempted to complete. Check the error messages above for details.")


def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description="AI Tutor Quiz Teacher Demo")
    parser.add_argument("files", nargs="+", help="Files to process")
    parser.add_argument("--api-key", help="OpenAI API key (optional, can use OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Run the demo
    asyncio.run(run_quiz_teacher_demo(args.files, args.api_key))


if __name__ == "__main__":
    main() 