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
async def call_planner_get_next_topic(ctx: RunContextWrapper[TutorContext], current_topic: Optional[str] = None) -> Optional[str]:
    """Determines the next linearly planned topic based on the current topic."""
    print(f"[Tool] Getting next topic linearly (current is '{current_topic}')")

    # Get the lesson plan from context
    lesson_plan = ctx.context.lesson_plan
    if not lesson_plan:
        print("[Tool] No lesson plan found in context.")
        return None

    # Extract all concepts from the lesson plan
    all_concepts: List[str] = [
        section.title for section in lesson_plan.sections
        if section.title  # Ensure the title exists
    ]

    if not all_concepts:
        print("[Tool] No concepts found in lesson plan.")
        return None

    if not current_topic: # If no current topic provided, return the first one
        next_topic = all_concepts[0]
        print(f"[Tool] Starting with first topic: '{next_topic}'")
        ctx.context.current_teaching_topic = next_topic  # Update context
        return next_topic

    try:
        current_idx = all_concepts.index(current_topic)
        if current_idx + 1 < len(all_concepts):
            next_topic = all_concepts[current_idx + 1]
            print(f"[Tool] Moving to next topic: '{next_topic}'")
            ctx.context.current_teaching_topic = next_topic  # Update context
            return next_topic
        else:
            print("[Tool] Reached end of lesson plan.")
            ctx.context.current_teaching_topic = None # Clear topic if at end
            return None
    except ValueError:
        # current_topic not found in the list, default to the first topic
        print(f"[Tool] Current topic '{current_topic}' not found in plan, starting from beginning.")
        next_topic = all_concepts[0]
        ctx.context.current_teaching_topic = next_topic  # Update context
        return next_topic

@function_tool
async def update_user_model(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained'],
    confusion_point: Optional[str] = None,
    last_accessed: Optional[str] = None
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

    print(f"[Tool] Updated '{topic}' - Mastery: {concept_state.mastery_level:.2f}, "
          f"Pace: {ctx.context.user_model_state.learning_pace_factor:.2f}")
    return f"User model updated for {topic}."

@function_tool
async def update_explanation_progress(ctx: RunContextWrapper[TutorContext], segment_index: int) -> str:
    """Updates the current explanation segment index in the user model state."""
    print(f"[Tool] Updating explanation segment index to {segment_index}")
    if not isinstance(segment_index, int) or segment_index < 0:
        return "Error: Invalid segment_index provided."
    ctx.context.user_model_state.current_explanation_segment = segment_index
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

# Ensure all tools intended for the orchestrator are exported or available
__all__ = [
    'call_quiz_creator_mini',
    'call_quiz_teacher_evaluate',
    'call_planner_get_next_topic',
    'update_user_model',
    'get_user_model_status',
    'update_explanation_progress',
] 