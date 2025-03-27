import os
import openai
import json

from agents import Agent, Runner, handoff, HandoffInputData
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import Quiz, QuizUserAnswers, QuizFeedback
from ai_tutor.agents.utils import process_handoff_data


def quiz_user_answers_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    """Filter function for handoff from quiz creator to quiz teacher agent.
    
    This ensures the quiz teacher agent receives both the quiz and user answers.
    """
    print("DEBUG: Entering quiz_user_answers_handoff_filter (Quiz User Answers)")
    
    try:
        processed_data = process_handoff_data(handoff_data)
        print("DEBUG: Exiting quiz_user_answers_handoff_filter")
        return processed_data
    except Exception as e:
        print(f"ERROR in quiz_user_answers_handoff_filter: {e}")
        return handoff_data  # Fallback


def create_quiz_teacher_agent(api_key: str = None):
    """Create a quiz teacher agent that evaluates user answers and provides feedback."""
    
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Ensure OPENAI_API_KEY is set in the environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set for quiz teacher agent!")
    else:
        print(f"Using OPENAI_API_KEY from environment for quiz teacher agent")
    
    # Create the quiz teacher agent
    quiz_teacher_agent = Agent(
        name="Quiz Teacher",
        instructions=prompt_with_handoff_instructions("""
        You are an expert educational instructor specialized in providing personalized feedback on quiz responses.
        Your task is to evaluate user answers to quiz questions and provide detailed, constructive feedback.
        
        You will receive both the quiz and the user's answers to the quiz questions. The quiz contains:
        - The questions
        - Multiple choice options for each question
        - The correct answer for each question 
        - Explanations for why the answers are correct
        
        Your job is to:
        1. Analyze each answer the user gave
        2. Determine if each answer is correct or incorrect
        3. Provide detailed feedback for each question, especially for incorrect answers
        4. Calculate the user's overall score and determine if they passed based on the quiz's passing score
        5. Provide personalized learning recommendations based on the pattern of errors
        
        Guidelines for providing effective feedback:
        - Be encouraging and supportive, even when pointing out errors
        - For incorrect answers, explain why the user's choice was wrong and why the correct answer is right
        - Identify patterns in the user's mistakes to suggest specific areas for improvement
        - Recommend targeted learning resources or practice exercises
        - Use a conversational, respectful tone
        - Tailor feedback to the user's performance level
        
        Format your response as a structured QuizFeedback object.
        
        YOUR OUTPUT MUST BE ONLY A VALID QUIZFEEDBACK OBJECT.
        """),
        output_type=QuizFeedback,
        model="o3-mini",
        # No handoffs needed for the quiz teacher agent - it's the last in the chain
    )
    
    return quiz_teacher_agent


async def generate_quiz_feedback(quiz: Quiz, user_answers: QuizUserAnswers, api_key: str = None, context=None) -> QuizFeedback:
    """Generate feedback for a user's quiz answers."""
    # Create the quiz teacher agent
    agent = create_quiz_teacher_agent(api_key)
    
    # Format the quiz and user answers as a string for the quiz teacher agent
    prompt = f"""
    QUIZ INFORMATION:
    
    Title: {quiz.title}
    Description: {quiz.description}
    Lesson Title: {quiz.lesson_title}
    Total Questions: {len(quiz.questions)}
    Passing Score: {quiz.passing_score}/{quiz.total_points}
    
    QUESTIONS:
    """
    
    for i, question in enumerate(quiz.questions):
        prompt += f"""
        Question {i+1}: {question.question}
        Options:
        """
        
        for j, option in enumerate(question.options):
            prompt += f"""
            Option {j+1}: {option}
            """
        
        prompt += f"""
        Correct Answer: Option {question.correct_answer_index + 1} - {question.options[question.correct_answer_index]}
        Explanation: {question.explanation}
        Difficulty: {question.difficulty}
        Related Section: {question.related_section}
        """
    
    prompt += f"""
    USER ANSWERS:
    
    Quiz Title: {user_answers.quiz_title}
    Total Time Taken: {user_answers.total_time_taken_seconds} seconds
    
    Answers:
    """
    
    for answer in user_answers.user_answers:
        question_index = answer.question_index
        selected_option_index = answer.selected_option_index
        
        # Ensure the question index is valid
        if question_index < len(quiz.questions):
            question = quiz.questions[question_index]
            selected_option = question.options[selected_option_index] if selected_option_index < len(question.options) else "Invalid option"
            
            prompt += f"""
            Question {question_index + 1}: {question.question}
            User Selected: Option {selected_option_index + 1} - {selected_option}
            Time Taken: {answer.time_taken_seconds} seconds
            """
    
    prompt += f"""
    INSTRUCTIONS:
    
    Based on this quiz and the user's answers, create a comprehensive feedback report.
    For each question:
    1. Indicate whether the user's answer was correct or incorrect
    2. Provide the correct answer if the user was wrong
    3. Explain why the answer is correct using the explanation provided in the quiz
    4. For incorrect answers, suggest how the user could improve their understanding
    
    Also provide:
    1. Overall score (number correct and percentage)
    2. Whether the user passed based on the passing score threshold
    3. Pattern analysis of mistakes (if any)
    4. Suggested topics for further study based on incorrect answers
    5. Next steps and learning recommendations
    
    YOUR OUTPUT MUST BE ONLY A VALID QUIZFEEDBACK OBJECT.
    """
    
    # Setup RunConfig for tracing
    from agents import RunConfig
    
    run_config = None
    if context and hasattr(context, 'session_id'):
        run_config = RunConfig(
            workflow_name="AI Tutor - Quiz Feedback",
            group_id=context.session_id # Link traces within the same session
        )
    
    # Run the quiz teacher agent
    result = await Runner.run(
        agent, 
        prompt,
        run_config=run_config,
        context=context
    )
    
    # Return the quiz feedback
    try:
        quiz_feedback = result.final_output_as(QuizFeedback)
        return quiz_feedback
    except Exception as e:
        print(f"Error parsing quiz feedback output: {e}")
        # Return a minimal feedback object if parsing fails
        return QuizFeedback(
            quiz_title=quiz.title,
            total_questions=len(quiz.questions),
            correct_answers=0,
            score_percentage=0.0,
            passed=False,
            total_time_taken_seconds=user_answers.total_time_taken_seconds,
            feedback_items=[],
            overall_feedback="Error generating feedback. Please try again.",
            suggested_study_topics=[],
            next_steps=["Contact support for assistance with quiz feedback."]
        ) 