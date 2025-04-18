import os
import openai
from datetime import datetime
from typing import Dict, Any, Optional, List

from agents import Agent, Runner, trace, RunConfig, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from ai_tutor.agents.models import (
    LessonPlan,
    LessonContent, # Simplified
    Quiz,
    QuizUserAnswers,
    QuizFeedback,
    LearningInsight,
    TeachingInsight,
    SessionAnalysis
)
from ai_tutor.agents.utils import RoundingModelWrapper


def create_session_analyzer_agent(api_key: str = None):
    """Create a session analyzer agent that evaluates the entire teaching workflow.
    
    Args:
        api_key: The OpenAI API key to use for the agent
        
    Returns:
        An Agent configured for session analysis
    """
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Ensure OPENAI_API_KEY is set in the environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set for session analyzer agent!")
    else:
        print(f"Using OPENAI_API_KEY from environment for session analyzer agent")
    
    # Instantiate the base model provider and get the base model
    provider: ModelProvider = OpenAIProvider()
    base_model = provider.get_model("gpt-4o")  # Using gpt-4o which supports structured output
    
    # Create the session analyzer agent
    session_analyzer_agent = Agent(
        name="Session Analyzer",
        instructions=prompt_with_handoff_instructions("""
        You are an expert educational analyst specialized in evaluating teaching sessions.
        
        Your task is to analyze the entire AI tutor workflow including:
        1. The lesson plan generated by the planner agent
        2. The teaching content created by the teacher agent
        3. The quiz created by the quiz creator agent
        4. The student's answers to the quiz questions
        5. The feedback provided by the quiz teacher agent
        
        You will have access to ALL raw outputs from the previous agents in the workflow, including:
        - The complete session information
        - The full lesson plan with all sections and objectives
        - The entire lesson content with all explanations and exercises
        - All quiz questions, options, and correct answers
        - Every user answer to each quiz question
        - The comprehensive quiz feedback, including specific feedback for each question
        
        When analyzing, pay particular attention to:
        - Did the student show comprehension of the material?
        - Were the teaching methods effective for the subject matter?
        - Did the quiz accurately assess the key learning objectives?
        - What patterns can be identified in the student's incorrect answers?
        - How well did the overall workflow progress from planning to assessment?
        - What insights can be derived to improve future teaching sessions?
        
        Guidelines for providing effective analysis:
        - Be objective and evidence-based in your assessments
        - Identify both strengths and areas for improvement
        - Provide specific examples from the session to support your insights
        - Make actionable recommendations for improving future sessions
        - Consider the alignment between lesson objectives, content, assessment, and outcomes
        - Evaluate the effectiveness of the teaching methods used
        - Be comprehensive in your analysis - use all the available information
        
        Format your response as a structured SessionAnalysis object.
        
        YOUR OUTPUT MUST BE ONLY A VALID SESSIONANALYSIS OBJECT.
        """),
        output_type=SessionAnalysis,
        model=RoundingModelWrapper(base_model),
    )
    
    return session_analyzer_agent


async def analyze_teaching_session(
    lesson_plan: LessonPlan, 
    lesson_content: LessonContent, # Expects simplified version
    quiz: Quiz, 
    user_answers: QuizUserAnswers, 
    quiz_feedback: QuizFeedback,
    session_duration_seconds: int,
    raw_agent_outputs: Optional[Dict[str, str]] = None,
    api_key: str = None,
    document_analysis = None,
    context = None
) -> SessionAnalysis:
    """Analyze a complete teaching session and generate insights."""
    # Create the session analyzer agent
    agent = create_session_analyzer_agent(api_key)
    
    # Generate a unique session ID
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # --- SIMPLIFIED PROMPT FORMATTING for Lesson Content ---
    prompt = f"""
    SESSION INFORMATION:
    
    Session ID: {session_id}
    Session Duration: {session_duration_seconds} seconds
    
    LESSON PLAN:
    
    Title: {lesson_plan.title}
    Description: {lesson_plan.description}
    Target Audience: {lesson_plan.target_audience}
    Total Estimated Duration: {lesson_plan.total_estimated_duration_minutes} minutes
    
    Prerequisites:
    """
    
    for prereq in lesson_plan.prerequisites:
        prompt += f"- {prereq}\n"
    
    prompt += f"\nSections:\n"
    
    for i, section in enumerate(lesson_plan.sections):
        prompt += f"""
        Section {i+1}: {section.title}
        Estimated Duration: {section.estimated_duration_minutes} minutes
        
        Learning Objectives:
        """
        
        for obj in section.objectives:
            prompt += f"- {obj.title}: {obj.description} (Priority: {obj.priority})\n"
        
        prompt += f"\nConcepts to Cover:\n"
        for concept in section.concepts_to_cover:
            prompt += f"- {concept}\n"
    
    prompt += f"""
    
    LESSON CONTENT:

    Title: {lesson_content.title}

    Text:
    {lesson_content.text}
    --- End of Text ---

    QUIZ:
    
    Title: {quiz.title}
    Description: {quiz.description}
    Passing Score: {quiz.passing_score}/{quiz.total_points}
    Estimated Completion Time: {quiz.estimated_completion_time_minutes} minutes
    
    Questions:
    """
    
    for i, question in enumerate(quiz.questions):
        prompt += f"""
        Question {i+1}: {question.question}
        Difficulty: {question.difficulty}
        Related Section: {question.related_section}
        
        Options:
        """
        
        for j, option in enumerate(question.options):
            prompt += f"Option {j+1}: {option}\n"
        
        prompt += f"""
        Correct Answer: Option {question.correct_answer_index + 1}
        Explanation: {question.explanation}
        """
    
    prompt += f"""
    
    USER QUIZ ANSWERS:
    
    Quiz Title: {user_answers.quiz_title}
    Total Time Taken: {user_answers.total_time_taken_seconds} seconds
    
    Answers:
    """
    
    for answer in user_answers.user_answers:
        prompt += f"""
        Question {answer.question_index + 1}:
        Selected: Option {answer.selected_option_index + 1}
        Time Taken: {answer.time_taken_seconds if answer.time_taken_seconds else 'N/A'} seconds
        """
    
    prompt += f"""
    
    QUIZ FEEDBACK:
    
    Quiz Title: {quiz_feedback.quiz_title}
    Score: {quiz_feedback.correct_answers}/{quiz_feedback.total_questions} ({quiz_feedback.score_percentage}%)
    Passed: {'Yes' if quiz_feedback.passed else 'No'}
    Total Time: {quiz_feedback.total_time_taken_seconds} seconds
    
    Overall Feedback: {quiz_feedback.overall_feedback}
    
    Question Feedback:
    """
    
    for item in quiz_feedback.feedback_items:
        prompt += f"""
        Question {item.question_index + 1}: {item.question_text}
        Selected: {item.user_selected_option}
        Correct: {item.correct_option}
        Correct?: {'Yes' if item.is_correct else 'No'}
        Explanation: {item.explanation}
        Improvement Suggestion: {item.improvement_suggestion}
        """
    
    prompt += f"""
    
    Suggested Study Topics:
    """
    
    for topic in quiz_feedback.suggested_study_topics:
        prompt += f"- {topic}\n"
    
    prompt += f"""
    
    Next Steps:
    """
    
    for step in quiz_feedback.next_steps:
        prompt += f"- {step}\n"
    
    if document_analysis:
        prompt += f"""
        
        DOCUMENT ANALYSIS:
        {document_analysis}
        """
    
    if raw_agent_outputs:
        prompt += f"""
        
        RAW AGENT OUTPUTS:
        """
        
        for agent_name, output in raw_agent_outputs.items():
            prompt += f"""
            {agent_name}:
            {output}
            """
    
    prompt += f"""
    
    INSTRUCTIONS:
    Based on all this information, create a comprehensive analysis of the teaching session.
    Analyze:
    1. Overall effectiveness.
    2. Quality of the lesson plan, the **synthesized lesson text**, and the quiz.
    3. Student's learning and performance (based on quiz).
    4. Teaching methodology effectiveness (more general now, based on text quality and quiz alignment).
    5. Recommendations for improvement.

    YOUR OUTPUT MUST BE ONLY A VALID SESSIONANALYSIS OBJECT.
    """
    
    # Setup RunConfig for tracing
    run_config = None
    if context and hasattr(context, 'session_id'):
        run_config = RunConfig(
            workflow_name="AI Tutor - Session Analysis",
            group_id=str(context.session_id)
        )
    
    # Run the session analyzer agent
    result = await Runner.run(
        agent,
        prompt,
        run_config=run_config,
        context=context
    )
    
    try:
        session_analysis = result.final_output_as(SessionAnalysis)
        print(f"Successfully generated session analysis for {session_id}")
        return session_analysis
    except Exception as e:
        print(f"Error parsing session analysis output: {e}")
        # Return a minimal analysis if parsing fails
        return SessionAnalysis(
            session_id=session_id,
            session_duration_seconds=session_duration_seconds,
            overall_effectiveness=0.0,
            strengths=[],
            improvement_areas=["Error generating analysis"],
            lesson_plan_quality=0.0,
            lesson_plan_insights=[],
            content_quality=0.0,
            content_insights=[],
            quiz_quality=0.0,
            quiz_insights=[],
            student_performance=0.0,
            learning_insights=[],
            teaching_effectiveness=0.0,
            teaching_insights=[],
            recommendations=[],
            recommended_adjustments=[],
            suggested_resources=[]
        ) 