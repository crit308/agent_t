import asyncio
import os
import sys
from typing import List

from ai_tutor.manager import AITutorManager
from ai_tutor.agents.models import QuizFeedback


async def test_quiz_teacher_workflow(file_paths: List[str], api_key: str):
    """Test the quiz teacher workflow including receiving user answers and providing feedback."""
    print("Initializing AI Tutor Manager...")
    manager = AITutorManager(api_key)
    
    print(f"Starting full workflow with quiz teacher using files: {', '.join(file_paths)}")
    result = await manager.run_full_workflow_with_quiz_teacher(file_paths)
    
    # Print lesson plan and content titles
    lesson_plan = result["lesson_plan"]
    lesson_content = result["lesson_content"]
    quiz = result["quiz"]
    user_answers = result["user_answers"]
    quiz_feedback = result["quiz_feedback"]
    
    print("\n" + "="*50)
    print(f"LESSON PLAN: {lesson_plan.title}")
    print(f"Sections: {len(lesson_plan.sections)}")
    for i, section in enumerate(lesson_plan.sections):
        print(f"  {i+1}. {section.title}")
    
    print("\n" + "="*50)
    print(f"LESSON CONTENT: {lesson_content.title}")
    print(f"Sections: {len(lesson_content.sections)}")
    for i, section in enumerate(lesson_content.sections):
        print(f"  {i+1}. {section.title}")
        print(f"     Explanations: {len(section.explanations)}")
        print(f"     Exercises: {len(section.exercises)}")
    
    # Print quiz information
    print("\n" + "="*50)
    print(f"QUIZ: {quiz.title}")
    print(f"Questions: {len(quiz.questions)}")
    print(f"Passing score: {quiz.passing_score}/{quiz.total_points}")
    print(f"Estimated completion time: {quiz.estimated_completion_time_minutes} minutes")
    
    # Print a sample of quiz questions (first 3 or less)
    sample_count = min(3, len(quiz.questions))
    if sample_count > 0:
        print("\nSample questions:")
        for i in range(sample_count):
            question = quiz.questions[i]
            print(f"  Q{i+1}: {question.question} (Difficulty: {question.difficulty})")
    
    # Print user answers summary
    print("\n" + "="*50)
    print(f"USER ANSWERS:")
    print(f"Total time taken: {user_answers.total_time_taken_seconds} seconds")
    print(f"Answers provided: {len(user_answers.user_answers)}")
    
    # Print a sample of user answers (first 3 or less)
    sample_count = min(3, len(user_answers.user_answers))
    if sample_count > 0:
        print("\nSample answers:")
        for i in range(sample_count):
            answer = user_answers.user_answers[i]
            question_idx = answer.question_index
            if question_idx < len(quiz.questions):
                question = quiz.questions[question_idx]
                selected_option = question.options[answer.selected_option_index]
                print(f"  Q{question_idx+1}: Selected {answer.selected_option_index+1}. {selected_option}")
                print(f"     Time taken: {answer.time_taken_seconds} seconds")
    
    # Print quiz feedback information
    print("\n" + "="*50)
    print(f"QUIZ FEEDBACK:")
    print(f"Score: {quiz_feedback.correct_answers}/{quiz_feedback.total_questions} ({quiz_feedback.score_percentage:.1f}%)")
    print(f"Passed: {'Yes' if quiz_feedback.passed else 'No'}")
    print(f"Total time: {quiz_feedback.total_time_taken_seconds} seconds")
    
    # Print overall feedback and suggested study topics
    print(f"\nOverall feedback: {quiz_feedback.overall_feedback}")
    print("\nSuggested study topics:")
    for topic in quiz_feedback.suggested_study_topics:
        print(f"  - {topic}")
    
    # Print next steps
    print("\nRecommended next steps:")
    for step in quiz_feedback.next_steps:
        print(f"  - {step}")
    
    # Print a sample of feedback items (first 3 or less)
    sample_count = min(3, len(quiz_feedback.feedback_items))
    if sample_count > 0:
        print("\nSample feedback on answers:")
        for i in range(sample_count):
            feedback = quiz_feedback.feedback_items[i]
            print(f"  Q{feedback.question_index+1}: {feedback.question_text} (Correct: {'Yes' if feedback.is_correct else 'No'})")
            print(f"     User selected: {feedback.user_selected_option}")
            print(f"     Correct option: {feedback.correct_option}")
            print(f"     Explanation: {feedback.explanation}")
            if not feedback.is_correct:
                print(f"     Improvement suggestion: {feedback.improvement_suggestion}")
            print("")
    
    print("\n" + "="*50)
    print("QUIZ TEACHER WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == "__main__":
    # Check for API key in environment or as command line argument
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if len(sys.argv) < 2:
        print("Usage: python test_quiz_teacher.py <file_path1> [file_path2 ...]")
        sys.exit(1)
    
    # Get file paths from command line arguments
    file_paths = sys.argv[1:]
    
    # Run the test
    asyncio.run(test_quiz_teacher_workflow(file_paths, api_key)) 