import os
import json
from typing import Optional

# Use ADK imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner, RunConfig

from ai_tutor.agents.models import Quiz, QuizUserAnswers, QuizFeedback, QuizQuestion, QuizFeedbackItem


def create_quiz_teacher_agent(api_key: str = None):
    """Create a quiz teacher agent that evaluates user answers and provides feedback."""
    
    # Use Gemini model via ADK
    model_identifier = "gemini-1.5-pro"  # Using Pro for better feedback capabilities
    
    # Create the quiz teacher agent
    quiz_teacher_agent = LlmAgent(
        name="quiz_teacher",
        instruction="""
        You are an expert educational feedback provider specializing in analyzing quiz responses and providing detailed, actionable feedback.
        Your task is to analyze the user's quiz answers and provide comprehensive feedback that helps them improve their understanding.
        
        Guidelines for providing effective feedback:
        1. Start with a clear overview of the user's performance
        2. For each question:
           - Explain why the answer was correct or incorrect
           - Provide a clear explanation of the concept
           - Suggest specific improvements if the answer was wrong
        3. Be encouraging and supportive, even when pointing out errors
        4. For incorrect answers, explain why the user's choice was wrong and why the correct answer is right
        5. Identify patterns in the user's mistakes to suggest specific areas for improvement
        6. *** IMPORTANT: When creating 'suggested_study_topics', use the 'related_section' information provided for each incorrectly answered question to guide the user back to the specific section of the lesson they should review. Be specific with section titles. ***
        7. Recommend targeted learning resources or practice exercises if appropriate
        8. Use a conversational, respectful tone
        9. Tailor feedback to the user's performance level
        
        Your feedback should help the user:
        1. Understand what they got right and why
        2. Learn from their mistakes
        3. Know exactly which sections to review
        4. Have a clear path forward for improvement
        
        Remember to:
        1. Be specific about which sections to review
        2. Use the exact section titles from the lesson
        3. Reference specific concepts within sections
        4. Provide actionable next steps
        5. Keep the tone encouraging and constructive
        
        YOUR OUTPUT MUST BE A VALID QUIZ FEEDBACK OBJECT.
        """,
        output_type=QuizFeedback,
        model=model_identifier
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
    run_config = None
    if context and hasattr(context, 'session_id'):
        run_config = RunConfig(
            workflow_name="AI Tutor - Quiz Feedback",
            group_id=str(context.session_id)
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


async def evaluate_single_answer(
    question: QuizQuestion,
    user_answer_index: int,
    api_key: str = None,
    context = None
) -> Optional[QuizFeedbackItem]:
    """
    Evaluates a single user answer against a given question using the Quiz Teacher agent.

    Args:
        question: The QuizQuestion object.
        user_answer_index: The 0-based index of the option selected by the user.
        api_key: Optional OpenAI API key.
        context: Optional context object (e.g., TutorContext).

    Returns:
        A QuizFeedbackItem object or None if evaluation fails.
    """
    # Input validation
    if not question or user_answer_index < 0 or user_answer_index >= len(question.options):
        print(f"Error: Invalid input for single answer evaluation. Question: {question}, Answer Index: {user_answer_index}")
        return None

    # Create the quiz teacher agent
    agent = create_quiz_teacher_agent(api_key)

    # Construct a focused prompt for single answer evaluation
    user_selected_option_text = question.options[user_answer_index]
    correct_option_text = question.options[question.correct_answer_index]
    is_correct = user_answer_index == question.correct_answer_index

    prompt = f"""
    You are evaluating a single quiz answer. Here is the question and the user's response:

    Question: {question.question}
    Options: {question.options}
    Correct Answer Index: {question.correct_answer_index} (Text: '{correct_option_text}')
    Provided Explanation: {question.explanation}
    User Selected Index: {user_answer_index} (Text: '{user_selected_option_text}')
    Was User Correct?: {'Yes' if is_correct else 'No'}

    INSTRUCTIONS:
    Based *only* on the information above, generate feedback for this single answer.
    1. Confirm if the user was correct.
    2. If incorrect, clearly state the correct answer.
    3. Use the provided explanation to explain *why* the correct answer is right.
    4. If incorrect, provide a concise improvement suggestion related *specifically* to this question/concept.
    5. Format your output ONLY as a valid QuizFeedbackItem JSON object.

    YOUR OUTPUT MUST BE ONLY A VALID QuizFeedbackItem OBJECT.
    """

    run_config = None
    if context and hasattr(context, 'session_id'):
        run_config = RunConfig(
            workflow_name="AI Tutor - Single Answer Eval",
            group_id=str(context.session_id)
        )

    result = await Runner.run(agent, prompt, run_config=run_config, context=context)

    try:
        # Parse the full feedback object first, as the agent is configured for it
        full_feedback = result.final_output_as(QuizFeedback)
        
        # Extract the first (and presumably only) feedback item
        if full_feedback and full_feedback.feedback_items:
            feedback_item = full_feedback.feedback_items[0]
            # Ensure the feedback item indices match the input (agent might hallucinate)
            feedback_item.question_index = 0 # Since we only evaluate one, index is 0 relative to this call
            feedback_item.question_text = question.question
            feedback_item.user_selected_option = user_selected_option_text
            feedback_item.is_correct = is_correct
            feedback_item.correct_option = correct_option_text
            # Keep agent's explanation and suggestion
            return feedback_item
        else:
            print(f"Error: Agent returned valid QuizFeedback but without any feedback items.")
            print(f"Agent raw output: {result.final_output}")
            return None
            
    except Exception as e:
        print(f"Error parsing single answer feedback from agent (expected QuizFeedback): {e}")
        print(f"Agent raw output: {result.final_output}")
        return None 