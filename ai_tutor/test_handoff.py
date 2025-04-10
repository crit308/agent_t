import asyncio
import os
import sys
from typing import List

from ai_tutor.manager import AITutorManager


async def test_handoff_workflow(file_paths: List[str], api_key: str):
    """Test the handoff workflow from teacher agent to quiz creator agent."""
    print("Initializing AI Tutor Manager...")
    manager = AITutorManager(api_key)
    
    print(f"Starting full workflow with files: {', '.join(file_paths)}")
    result = await manager.run_full_workflow(file_paths)
    
    # Print lesson plan and content titles
    lesson_plan = result["lesson_plan"]
    lesson_content = result["lesson_content"]
    
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
    
    # Print quiz information if available
    if "quiz" in result:
        quiz = result["quiz"]
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
    else:
        print("\n" + "="*50)
        print("NO QUIZ GENERATED")
    
    print("\n" + "="*50)
    print("HANDOFF RESULT:")
    if "quiz" in result:
        print("Handoff from Teacher Agent to Quiz Creator was successful!")
        print("The Quiz Creator agent generated a quiz based on the lesson content.")
    else:
        print("Handoff to Quiz Creator agent might not have occurred.")
        print("Check the trace URL above for the complete conversation.")
    print("="*50)


if __name__ == "__main__":
    # Check for API key in environment or as command line argument
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if len(sys.argv) < 2:
        print("Usage: python test_handoff.py <file_path1> [file_path2 ...]")
        sys.exit(1)
    
    # Get file paths from command line arguments
    file_paths = sys.argv[1:]
    
    # Run the test
    asyncio.run(test_handoff_workflow(file_paths, api_key)) 