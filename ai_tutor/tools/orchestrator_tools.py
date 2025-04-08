from __future__ import annotations
from agents import function_tool, Runner, RunConfig
from agents.run_context import RunContextWrapper
from typing import Any, Optional, Literal, Union, cast, Dict, List
import os
from datetime import datetime
import traceback # Import traceback for error logging

# --- Import TutorContext directly ---
# We need the actual class definition available for get_type_hints called by the decorator.
# This relies on context.py being fully loaded *before* this file attempts to define the tools.
from ai_tutor.context import TutorContext, UserConceptMastery, UserModelState

# --- Import necessary models ---
from ai_tutor.agents.models import LessonPlan, QuizQuestion, QuizFeedbackItem, LessonContent, Quiz, LessonSection, LearningObjective

# Import API response models for potentially formatting tool output
from ai_tutor.api_models import (
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse
)

# --- Orchestrator Tool Implementations ---

@function_tool
async def call_quiz_creator_mini(
    ctx: RunContextWrapper[TutorContext],
    topic: str
) -> Union[QuizQuestion, str]: # Return Question object or error string
    """Generates a single multiple-choice question for the given topic using the Quiz Creator agent."""
    print(f"[Tool] Requesting mini-quiz for topic '{topic}'")
    
    if not topic or not isinstance(topic, str):
        return "Error: Invalid topic provided for quiz."

    try:
        # --- Import and Create Agent *Inside* ---
        from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent # Import here
        # ----------------------------------------
        run_config = RunConfig(workflow_name="Orchestrator_QuizCall", group_id=ctx.context.session_id)

        # Get user's mastery level for difficulty adjustment
        mastery_level = 0.0
        if topic in ctx.context.user_model_state.concepts:
            mastery_level = ctx.context.user_model_state.concepts[topic].mastery_level

        # Get confusion points if any
        confusion_points = []
        if topic in ctx.context.user_model_state.concepts:
            confusion_points = ctx.context.user_model_state.concepts[topic].confusion_points

        # Construct the prompt
        confusion_points_str = "\nFocus on these areas of confusion:\n- " + "\n- ".join(confusion_points) if confusion_points else ""
        difficulty_guidance = "Make the question more challenging." if mastery_level > 0.7 else "Keep the question straightforward." if mastery_level < 0.3 else ""

        quiz_prompt = f"""
        Create a single multiple-choice question to test understanding of: {topic}
        
        Current mastery level: {mastery_level:.2f}
        {difficulty_guidance}
        {confusion_points_str}
        
        Format your response as a Quiz object with a single question that includes:
        - question text
        - multiple choice options
        - correct answer index
        - explanation for the correct answer
        - related_section: '{topic}'
        """

        result = await Runner.run(
            quiz_creator,
            quiz_prompt,
            context=ctx.context,
            run_config=run_config
        )

        quiz_output = result.final_output_as(Quiz)
        if quiz_output and quiz_output.questions:
            first_question = quiz_output.questions[0]
            print(f"[Tool] Got mini-quiz for '{topic}': {first_question.question[:50]}...")
            ctx.context.current_quiz_question = first_question
            return first_question
        else:
            return f"Error: Quiz creator failed to generate question for topic '{topic}'."

    except Exception as e:
        error_msg = f"Error in call_quiz_creator_mini: {str(e)}"
        print(f"[Tool] {error_msg}")
        return error_msg


@function_tool
async def call_quiz_teacher_evaluate(ctx: RunContextWrapper[TutorContext], user_answer_index: int) -> Union[QuizFeedbackItem, str]:
    """Evaluates the user's answer to the current question using the Quiz Teacher logic (via helper function)."""
    print(f"[Tool] Evaluating user answer index '{user_answer_index}' for current question.")
    
    try:
        # --- Import evaluation function *Inside* ---
        from ai_tutor.agents.quiz_teacher_agent import evaluate_single_answer # Import helper here
        # -------------------------------------------
        question_to_evaluate = ctx.context.current_quiz_question
        if not question_to_evaluate:
            return "Error: No current question found in context to evaluate."
        if not isinstance(question_to_evaluate, QuizQuestion):
            # Add type check for safety
            return f"Error: Expected QuizQuestion in context, found {type(question_to_evaluate).__name__}."

        print(f"[Tool] Evaluating answer for question: {question_to_evaluate.question[:50]}...")
        feedback_item = await evaluate_single_answer(
            question=question_to_evaluate,
            user_answer_index=user_answer_index,
            context=ctx.context # Pass context
        )

        if feedback_item:
            print(f"[Tool] Evaluation complete. Feedback: Correct={feedback_item.is_correct}, Explanation: {feedback_item.explanation[:50]}...")
            # Clear pending interaction *after* successful evaluation & getting feedback
            ctx.context.user_model_state.pending_interaction_type = None
            ctx.context.user_model_state.pending_interaction_details = None
            # Clear the question itself from context now that it's evaluated
            ctx.context.current_quiz_question = None
            return feedback_item
        else:
            # The evaluate_single_answer helper should ideally return feedback or raise
            error_msg = f"Error: Evaluation helper function returned None for question on topic '{getattr(question_to_evaluate, 'related_section', 'N/A')}'."
            print(f"[Tool] {error_msg}")
            return error_msg # Return error string
            
    except Exception as e:
        error_msg = f"Error in call_quiz_teacher_evaluate: {str(e)}"
        print(f"[Tool] {error_msg}")
        # Optionally clear pending state even on error? Depends on desired flow.
        # ctx.context.user_model_state.pending_interaction_type = None 
        # ctx.context.user_model_state.pending_interaction_details = None
        return error_msg # Return error string

@function_tool
async def determine_next_learning_step(ctx: RunContextWrapper[TutorContext]) -> Dict[str, Any]:
    """
    Determines the next learning step (topic/section and objectives) based on the lesson plan and user's current progress/mastery.
    Updates the context with the next topic and its objectives.
    Returns the title of the next section/topic to begin, or None if the lesson is complete.
    """
    lesson_plan = ctx.context.lesson_plan
    user_state = ctx.context.user_model_state
    current_topic = ctx.context.current_teaching_topic

    print(f"[Tool determine_next_learning_step] Called. Current topic: '{current_topic}'")

    if not lesson_plan or not lesson_plan.sections:
        print("[Tool determine_next_learning_step] Lesson plan not found or empty.")
        return {"error": "Lesson plan not found or empty.", "next_topic": None}

    current_section_index = -1
    # Get the lesson plan from context
    if not lesson_plan:
        print("[Tool determine_next_learning_step] No lesson plan found in context.")
        return {"error": "Lesson plan not found.", "next_topic": None}

    if current_topic:
        try:
            current_section_index = next(i for i, sec in enumerate(lesson_plan.sections) if sec.title == current_topic)
        except StopIteration:
            print(f"[Tool determine_next_learning_step] Warning: Current topic '{current_topic}' not found in plan. Resetting.")
            current_topic = None # Reset if current topic is invalid

    next_section_index = -1
    next_section = None

    # If there's a current topic, check if objectives are met before moving on
    if current_section_index != -1:
         current_section = lesson_plan.sections[current_section_index]
         # Basic check: Have all objectives for this section been 'mastered'?
         # TODO: Enhance this check - map concept mastery to objectives if possible.
         all_objectives_mastered = len(user_state.mastered_objectives_current_section) >= len(current_section.objectives)

         if all_objectives_mastered:
             print(f"[Tool determine_next_learning_step] Objectives for '{current_topic}' seem complete. Moving to next section.")
             if current_section_index + 1 < len(lesson_plan.sections):
                 next_section_index = current_section_index + 1
             else:
                 next_section_index = -2 # Signal end of lesson
         else:
             print(f"[Tool determine_next_learning_step] Objectives for '{current_topic}' not yet met. Staying on this topic.")
             # Stay on the current topic, return its details again
             next_section_index = current_section_index
    else:
         # No current topic, start from the beginning
         print("[Tool determine_next_learning_step] No current topic. Starting from the first section.")
         next_section_index = 0

    # Determine the next section based on the index
    if 0 <= next_section_index < len(lesson_plan.sections):
         next_section = lesson_plan.sections[next_section_index]
         next_topic_title = next_section.title

         # --- Update Context ---
         ctx.context.current_teaching_topic = next_topic_title
         ctx.context.user_model_state.current_topic_segment_index = 0 # Reset segment index for new/repeated topic
         ctx.context.user_model_state.current_section_objectives = next_section.objectives
         # Reset mastered objectives if it's a *new* topic
         if current_topic != next_topic_title:
             ctx.context.user_model_state.mastered_objectives_current_section = []
         # --- End Update Context ---

         print(f"[Tool determine_next_learning_step] Next step: Topic='{next_topic_title}', Objectives: {[o.title for o in next_section.objectives]}")
         return {"next_topic": next_topic_title, "objectives": [o.model_dump() for o in next_section.objectives]}
    else:
         # End of lesson plan (index was -2 or out of bounds)
         print("[Tool determine_next_learning_step] Reached end of lesson plan.")
         ctx.context.current_teaching_topic = None
         ctx.context.user_model_state.current_section_objectives = []
         return {"next_topic": None, "message": "Lesson complete!"}


@function_tool
async def update_user_model(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained'],
    confusion_point: Optional[str] = None,
    last_accessed: Optional[str] = None,
    mastered_objective_title: Optional[str] = None, # Optional: Mark an objective as mastered
) -> str:
    """Updates the user model state with interaction outcomes and temporal data."""
    print(f"[Tool] Updating user model for topic '{topic}' with outcome '{outcome}'")

    # Ensure context and user model state exist
    if not ctx.context or not ctx.context.user_model_state:
        return "Error: TutorContext or UserModelState not found."

    if not topic or not isinstance(topic, str):
        return "Error: Invalid topic provided for user model update."

    # Initialize concept if needed
    if topic not in ctx.context.user_model_state.concepts:
        ctx.context.user_model_state.concepts[topic] = UserConceptMastery()

    concept_state = ctx.context.user_model_state.concepts[topic]
    concept_state.last_interaction_outcome = outcome

    # Update last_accessed with ISO 8601 timestamp
    concept_state.last_accessed = last_accessed or datetime.now().isoformat()

    # Add confusion point if provided
    if confusion_point and confusion_point not in concept_state.confusion_points:
        concept_state.confusion_points.append(confusion_point)

    # Update attempts and mastery for evaluative outcomes
    if outcome in ['correct', 'incorrect', 'mastered', 'struggled']:
        concept_state.attempts += 1

        # Adjust mastery level based on outcome
        if outcome in ['correct', 'mastered']:
            concept_state.mastery_level = min(1.0, concept_state.mastery_level + 0.2)
        elif outcome in ['incorrect', 'struggled']:
            concept_state.mastery_level = max(0.0, concept_state.mastery_level - 0.1)
            
            # Adjust learning pace if struggling
            if len(concept_state.confusion_points) > 2:
                ctx.context.user_model_state.learning_pace_factor = max(0.5, 
                    ctx.context.user_model_state.learning_pace_factor - 0.1)

    # Update mastered objectives if provided
    if mastered_objective_title and mastered_objective_title not in ctx.context.user_model_state.mastered_objectives_current_section:
         ctx.context.user_model_state.mastered_objectives_current_section.append(mastered_objective_title)
         print(f"[Tool] Marked objective '{mastered_objective_title}' as mastered for current section.")

    print(f"[Tool] Updated '{topic}' - Mastery: {concept_state.mastery_level:.2f}, "
          f"Pace: {ctx.context.user_model_state.learning_pace_factor:.2f}")
    return f"User model updated for {topic}."

@function_tool
async def update_explanation_progress(ctx: RunContextWrapper[TutorContext], segment_index: int) -> str:
    """Updates the current explanation segment index in the user model state."""
    print(f"[Tool] Updating explanation segment index to {segment_index}")
    if not isinstance(segment_index, int) or segment_index < 0:
        return "Error: Invalid segment_index provided."
    try:
        ctx.context.user_model_state.current_topic_segment_index = segment_index # CORRECT
    except AttributeError:
        return "Error: There was an issue updating the explanation progress. The user model doesn't have the expected fields." # Return error message
    ctx.context.last_interaction_summary = f"Delivered explanation segment {segment_index}"
    return f"Explanation progress updated to segment {segment_index}."

@function_tool
async def get_user_model_status(ctx: RunContextWrapper[TutorContext], topic: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves detailed user model state, optionally for a specific topic."""
    print(f"[Tool] Retrieving user model status for topic '{topic}'")

    if not ctx.context.user_model_state:
        return {"error": "No user model state found in context."}

    state = ctx.context.user_model_state

    if topic:
        if topic not in state.concepts:
            return {
                "topic": topic,
                "exists": False,
                "message": "Topic not found in user model."
            }
            
        concept = state.concepts[topic]
        return {
            "topic": topic,
            "exists": True,
            "mastery_level": concept.mastery_level,
            "attempts": concept.attempts,
            "last_outcome": concept.last_interaction_outcome,
            "confusion_points": concept.confusion_points,
            "last_accessed": concept.last_accessed
        }
    
    return {
        "overall_progress": state.overall_progress,
        "current_topic": state.current_topic,
        "learning_pace": state.learning_pace_factor,
        "preferred_style": state.preferred_interaction_style,
        "topics": {
            topic: {
                "mastery": concept.mastery_level,
                "attempts": concept.attempts,
                "confusion_points": len(concept.confusion_points)
            }
            for topic, concept in state.concepts.items()
        }
    } 

@function_tool
async def reflect_on_interaction(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    interaction_summary: str, # e.g., "User answered checking question incorrectly."
    user_response: Optional[str] = None, # The actual user answer/input
    feedback_provided: Optional[QuizFeedbackItem] = None # Feedback from teacher tool if available
) -> Dict[str, Any]:
    """
    Analyzes the last interaction for a given topic, identifies potential reasons for user difficulty,
    and suggests adaptive next steps for the Orchestrator.
    """
    print(f"[Tool reflect_on_interaction] Called for topic '{topic}'. Summary: {interaction_summary}")

    # Basic reflection logic (can be enhanced, e.g., calling another LLM for deeper analysis)
    suggestions = []
    analysis = f"Reflection on interaction regarding '{topic}': {interaction_summary}. "

    if feedback_provided and not feedback_provided.is_correct:
        analysis += f"User incorrectly selected '{feedback_provided.user_selected_option}' when the correct answer was '{feedback_provided.correct_option}'. "
        analysis += f"Explanation: {feedback_provided.explanation}. "
        suggestions.append(f"Re-explain the core concept using the provided explanation: '{feedback_provided.explanation}'.")
        if feedback_provided.improvement_suggestion:
            suggestions.append(f"Focus on the improvement suggestion: '{feedback_provided.improvement_suggestion}'.")
        suggestions.append(f"Try asking a slightly different checking question on the same concept.")
    elif "incorrect" in interaction_summary.lower() or "struggled" in interaction_summary.lower():
        suggestions.append(f"Consider re-explaining the last segment of '{topic}' using a different approach or analogy.")
        suggestions.append(f"Ask a simpler checking question focused on the specific confusion points for '{topic}'.")

    print(f"[Tool reflect_on_interaction] Analysis: {analysis}. Suggestions: {suggestions}")
    return {"analysis": analysis, "suggested_next_steps": suggestions}

# Ensure all tools intended for the orchestrator are exported or available
__all__ = [
    'call_quiz_creator_mini',
    'call_quiz_teacher_evaluate',
    'reflect_on_interaction',
    'determine_next_learning_step',
    'update_user_model',
    'get_user_model_status',
    'update_explanation_progress',
] 