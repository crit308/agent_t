import os
import openai
import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID
from pydantic import ValidationError
from textwrap import shorten

from agents import Agent, Runner, trace, RunConfig, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from ai_tutor.context import TutorContext
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
from ai_tutor.skills.session_analysis_tools import read_interaction_logs

logger = logging.getLogger(__name__)


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
        logger.warning("OPENAI_API_KEY environment variable is not set for session analyzer agent!")
    else:
        logger.debug(f"Using OPENAI_API_KEY from environment for session analyzer agent")
    
    # Instantiate the base model provider and get the base model
    provider: ModelProvider = OpenAIProvider()
    base_model = provider.get_model("gpt-4.1-2025-04-14")  # Updated to gpt-4.1-2025-04-14
    
    # Create the session analyzer agent
    session_analyzer_agent = Agent(
        name="Session Analyzer",
        instructions="""
        You are an expert educational analyst specialized in evaluating tutoring sessions.
        
        Your task is to analyze the tutoring session based on the provided interaction log summary.
        
        Use the `read_interaction_logs` tool to get a summary of the conversation between the student (user) and the AI tutor (agent).
        Then, perform your analysis focusing on:
        - Student comprehension and performance patterns (e.g., areas of struggle, quick grasps).
        - Effectiveness of teaching methods used by the agent (based on log events and responses).
        - Alignment between the session interactions and potential learning objectives (infer if necessary).
        - Actionable recommendations for future sessions or improvements.
        - Any potential issues or successes in the interaction flow.
        
        REQUIRED OUTPUT FORMAT:
        1.  A concise plain-text summary (max 300 words) suitable for appending to a knowledge base. Start this summary *exactly* with the phrase "Session Summary:". Do not include any preamble before this phrase.
        2.  Optionally, after the text summary, include a JSON object conforming to the SessionAnalysis Pydantic model, enclosed in ```json ... ``` marks, for detailed structured data. Make sure the JSON is valid.
        
        Example Text Summary:
        Session Summary: The student grasped evaporation quickly but struggled with condensation, requiring multiple explanations and a targeted question. Overall progress was good. Recommend starting the next session with a brief review of condensation using a different analogy.
        
        Example Text Summary with JSON:
        Session Summary: The student initially confused photosynthesis inputs and outputs but corrected themselves after the agent provided a clarifying example. They answered the subsequent quiz question correctly.
        ```json
        {
            "session_id": "uuid-goes-here",
            "analysis_timestamp": "YYYY-MM-DDTHH:MM:SSZ",
            "overall_effectiveness": 75.0,
            "student_performance_summary": "Student showed initial confusion but demonstrated learning after clarification.",
            "teaching_effectiveness_summary": "Agent effectively used an example to clarify a misconception.",
            "alignment_summary": "Session focused well on the core concept.",
            "learning_insights": [
                {"insight": "Student may confuse related biological processes.", "confidence": 0.7}
            ],
            "teaching_insights": [
                {"insight": "Providing concrete examples is effective for this student.", "confidence": 0.9}
            ],
            "recommendations": [
                "Monitor for confusion between similar processes in future sessions.",
                "Continue using examples to explain concepts."
            ]
        }
        ```
        
        If you cannot generate the structured JSON part for any reason, just provide the plain-text summary starting with "Session Summary:".
        """,
        output_type=SessionAnalysis,
        tools=[read_interaction_logs],
        model=base_model,
    )
    
    return session_analyzer_agent


# -------------------------------
# Helper: Generate plain-text summary from SessionAnalysis object
# -------------------------------
def _generate_text_summary_from_analysis(analysis: 'SessionAnalysis') -> str:
    """Creates a concise text summary string (≤300 words) from a SessionAnalysis object.

    The summary always starts with the required prefix "Session Summary:" so that
    downstream regex logic and KB conventions remain consistent.
    """
    if analysis is None:
        return "Session Summary: (No structured analysis available.)"

    parts: list[str] = [
        "Session Summary:",
        f"Overall Effectiveness: {analysis.overall_effectiveness:.1f}/100."
    ]

    if analysis.strengths:
        parts.append("Strengths: " + ", ".join(analysis.strengths))
    if analysis.improvement_areas:
        parts.append("Areas for Improvement: " + ", ".join(analysis.improvement_areas))

    # Add up to three learning insights
    if analysis.learning_insights:
        parts.append("Key Learning Points:")
        for insight in analysis.learning_insights[:3]:
            flag = "Strength" if insight.strength else "Issue"
            parts.append(f"- {insight.topic} ({flag}): {insight.observation} → {insight.recommendation}")

    if analysis.recommendations:
        parts.append("Recommendation: " + analysis.recommendations[0])

    summary_text = "\n".join(parts)

    # Ensure word limit (300 words) – truncate gracefully if needed
    words = summary_text.split()
    if len(words) > 300:
        summary_text = " ".join(words[:300]) + " …"

    return summary_text


async def analyze_session(session_id: UUID, context: Optional[TutorContext] = None) -> Tuple[Optional[str], Optional[SessionAnalysis]]:
    """Analyzes a session using its ID to fetch logs and KB.

    Args:
        session_id: The UUID of the session to analyze.
        context: Optional TutorContext for run configuration.

    Returns:
        A tuple containing:
        - Optional[str]: The extracted plain-text summary, or None if not found.
        - Optional[SessionAnalysis]: The parsed structured analysis, or None if not found/invalid.
    """
    agent = create_session_analyzer_agent()
    run_config = None
    if context and context.session_id:
        run_config = RunConfig(
            workflow_name="AI Tutor - Session Analysis",
            group_id=str(context.session_id)
        )
        logger.info(f"Running session analysis with RunConfig for group_id: {context.session_id}")
    else:
        logger.info("Running session analysis without RunConfig.")

    # Construct prompt telling agent to use the tool
    prompt = f"Analyze the tutoring session with ID {session_id}. Use the `read_interaction_logs` tool to get the interaction summary. Based *only* on that summary, provide your analysis including a concise text summary starting with 'Session Summary:' and, if possible, the SessionAnalysis JSON object in markdown format."

    logger.info(f"Running Session Analyzer agent for session {session_id}")
    result = await Runner.run(
        agent,
        prompt,
        run_config=run_config,
        max_turns=20,  # Increase turn limit to avoid premature MaxTurnsExceeded
    )

    text_summary: Optional[str] = None
    structured_analysis: Optional[SessionAnalysis] = None

    # ---------------------------------------------
    # Handle RunResult output (string or object)
    # ---------------------------------------------
    if result and hasattr(result, 'final_output'):
        final_output = result.final_output

        # --- Case 1: Agent returned structured object directly ---
        if isinstance(final_output, SessionAnalysis):
            logger.info("Analyzer returned SessionAnalysis object directly. Generating summary.")
            structured_analysis = final_output
            # Ensure essential identifiers are filled
            if not structured_analysis.session_id:
                structured_analysis.session_id = str(session_id)

            text_summary = _generate_text_summary_from_analysis(structured_analysis)

        # --- Case 2: Agent returned string (legacy behaviour) ---
        elif isinstance(final_output, str):
            final_text: str = final_output
            logger.debug(f"Session Analyzer raw output for {session_id}:{final_text}")

            # Extract Text Summary (Must start with "Session Summary:")
            # Use regex that finds "Session Summary:" and captures everything after it
            # until the end of the string OR the start of the json block
            summary_match = re.search(r"Session Summary:(.*?)(?:$|```json)", final_text, re.DOTALL | re.IGNORECASE)
            if summary_match:
                text_summary = summary_match.group(1).strip()
                logger.info(f"Extracted text summary for session {session_id}. Length: {len(text_summary)}")
            else:
                logger.warning(f"Could not find 'Session Summary:' prefix in analyzer output for {session_id}.")
                # Fallback: If no prefix, but JSON is missing, maybe the whole output IS the summary?
                if "```json" not in final_text:
                    logger.warning(f"Assuming entire output is the summary for {session_id} due to missing prefix and JSON.")
                    text_summary = final_text.strip()

            # Extract JSON part
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", final_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                logger.info(f"Found JSON block for session {session_id}. Attempting parse.")
                try:
                    analysis_data = json.loads(json_str)
                    # Add session_id if missing from LLM output (best practice)
                    analysis_data.setdefault('session_id', str(session_id))
                    # TODO: Add duration calculation if needed and not in JSON - requires session start/end times
                    structured_analysis = SessionAnalysis.model_validate(analysis_data)
                    logger.info(f"Successfully parsed and validated SessionAnalysis JSON for {session_id}.")
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse SessionAnalysis JSON for {session_id}: {json_err}. JSON string: {json_str}")
                    # Keep text_summary even if JSON fails
                except ValidationError as val_err:
                    logger.error(f"Failed to validate SessionAnalysis JSON for {session_id}: {val_err}. Parsed data: {analysis_data}")
                    # Keep text_summary even if JSON fails
            else:
                logger.info(f"No valid SessionAnalysis JSON found in ```json ... ``` block for session {session_id}.")

            # Handle case where only summary was expected/provided
            if text_summary and not structured_analysis and not json_match:
                logger.info(f"Analyzer provided only text summary for session {session_id}, as expected.")
            elif not text_summary and not structured_analysis:
                logger.error(f"Session Analyzer for {session_id} failed to produce usable output (no summary prefix, no JSON). Raw output: {final_text}")

    else:
        logger.error(f"Session Analyzer for {session_id} did not return usable output. Result: {result}")

    return text_summary, structured_analysis


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