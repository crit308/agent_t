# AI Tutor

A system for creating AI-powered tutors that can teach content from uploaded documents using the OpenAI Agents SDK.

## Overview

The AI Tutor system uses a multi-agent approach to create personalized lessons:

1. **Document Processing**: Upload documents you want to learn about. The system creates a vector store using OpenAI's embeddings.
2. **Lesson Planning**: A planner agent analyzes the documents and creates a structured lesson plan.
3. **Lesson Creation**: A teacher agent takes the lesson plan and creates comprehensive lesson content.
4. **Quiz Creation**: A quiz creator agent generates a quiz based on the lesson content.
5. **Quiz Assessment**: A quiz teacher agent evaluates user answers and provides personalized feedback.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-tutor.git
cd ai-tutor

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

You can use the AI Tutor from the command line:

```bash
# Run the AI Tutor with one or more files
python -m ai_tutor.main file1.pdf file2.pdf

# Save the lesson content to a file
python -m ai_tutor.main file1.pdf --output lesson.json

# Specify your own API key
python -m ai_tutor.main file1.pdf --api-key your-api-key
```

### Python API

You can also use the AI Tutor programmatically:

```python
import asyncio
from ai_tutor.manager import AITutorManager

async def create_lesson(file_paths):
    # Initialize the AI Tutor manager
    manager = AITutorManager(api_key="your-api-key")
    
    # Upload documents
    await manager.upload_documents(file_paths)
    
    # Generate lesson plan
    lesson_plan = await manager.generate_lesson_plan()
    print(f"Generated lesson plan: {lesson_plan.title}")
    
    # Generate lesson content
    lesson_content = await manager.generate_lesson_content()
    print(f"Generated lesson: {lesson_content.title}")
    
    return lesson_content

# Run the function
lesson = asyncio.run(create_lesson(["file1.pdf", "file2.pdf"]))
```

### Working with Quizzes and Feedback

You can create quizzes and get feedback on user answers:

```python
import asyncio
from ai_tutor.manager import AITutorManager
from ai_tutor.agents.models import QuizUserAnswer, QuizUserAnswers

async def create_and_evaluate_quiz(file_paths):
    # Initialize the AI Tutor manager
    manager = AITutorManager(api_key="your-api-key")
    
    # Run the full workflow to generate a lesson and quiz
    await manager.upload_documents(file_paths)
    await manager.generate_lesson_plan()
    await manager.generate_lesson_content()
    quiz = await manager.generate_quiz()
    
    # Create user answers (in a real application, these would come from user input)
    user_answers = QuizUserAnswers(
        quiz_title=quiz.title,
        user_answers=[
            QuizUserAnswer(
                question_index=0,
                selected_option_index=0,  # User selected the first option
                time_taken_seconds=30
            ),
            QuizUserAnswer(
                question_index=1,
                selected_option_index=2,  # User selected the third option
                time_taken_seconds=45
            ),
            # Add more answers as needed
        ],
        total_time_taken_seconds=75
    )
    
    # Get feedback on the answers
    feedback = await manager.submit_quiz_answers(user_answers)
    
    print(f"Score: {feedback.correct_answers}/{feedback.total_questions}")
    print(f"Passed: {feedback.passed}")
    print(f"Overall feedback: {feedback.overall_feedback}")
    
    return feedback

# Run the function
feedback = asyncio.run(create_and_evaluate_quiz(["file1.pdf", "file2.pdf"]))
```

You can also run the complete workflow including quiz assessment:

```python
# Run the complete workflow including quiz assessment
result = asyncio.run(manager.run_full_workflow_with_quiz_teacher(["file1.pdf"]))
lesson_plan = result["lesson_plan"]
lesson_content = result["lesson_content"]
quiz = result["quiz"]
user_answers = result["user_answers"]  # Sample answers for demonstration
quiz_feedback = result["quiz_feedback"]
```

## Supported File Types

The AI Tutor supports many document types including:

- PDF files (.pdf)
- Word documents (.docx, .doc)
- Text files (.txt)
- Markdown files (.md)
- PowerPoint presentations (.pptx, .ppt)

## Viewing Traces

You can view detailed traces of the AI Tutor's execution on the OpenAI platform:

1. Go to [https://platform.openai.com/traces](https://platform.openai.com/traces)
2. Find the trace ID displayed after running the AI Tutor
3. Click on the trace to view detailed information about the execution

## License

MIT 