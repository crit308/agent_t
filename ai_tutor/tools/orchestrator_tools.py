from __future__ import annotations
from agents import function_tool, Runner, RunConfig
from agents.run_context import RunContextWrapper
from typing import Any, Optional, Literal, Union, cast
import os

# --- Import TutorContext directly ---
# We need the actual class definition available for get_type_hints called by the decorator.
# This relies on context.py being fully loaded *before* this file attempts to define the tools.
from ai_tutor.context import TutorContext

# --- Import agent creation functions DIRECTLY from their modules ---
from ai_tutor.agents.teacher_agent import create_teacher_agent_without_handoffs
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
from ai_tutor.agents.quiz_teacher_agent import create_quiz_teacher_agent, evaluate_single_answer # Import evaluate_single_answer here

# --- Import necessary models ---
from ai_tutor.agents.models import LessonPlan, QuizQuestion, QuizFeedbackItem, LessonContent, Quiz, LessonSection, LearningObjective

# Import API response models for potentially formatting tool output
from ai_tutor.api_models import (
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse
)

# --- Orchestrator Tool Implementations ---

@function_tool
async def call_teacher_explain(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    segment_index: int
) -> Union[LessonContent, str]: # Return LessonContent or error string
    """Generates an explanation for a specific topic and segment using the Teacher Agent."""
    print(f"Orchestrator tool: Requesting explanation for topic '{topic}', segment {segment_index}")
    
    # Input validation
    if not topic or not isinstance(topic, str):
        return "Error: Invalid topic provided."
    if not isinstance(segment_index, int) or segment_index < 0:
        return "Error: Invalid segment index provided."
    
    if not ctx.context.lesson_plan:
         return "Error: Lesson plan not found in context. Cannot generate explanation without plan context."

    try:
        # Get the teacher agent instance using the corrected import
        teacher_agent = create_teacher_agent_without_handoffs(ctx.context.vector_store_id)
        run_config = RunConfig(workflow_name="Orchestrator_TeacherCall", group_id=ctx.context.session_id)

        # We now construct the prompt *here* for the teacher agent
        lesson_plan_title = ctx.context.lesson_plan.title
        lesson_plan_context_str = f"Lesson Plan Context:\nTitle: {lesson_plan_title}\n..." # Simplified for brevity
        analysis_context_str = f"Analysis Context:\nKey Concepts: {ctx.context.analysis_result.key_concepts[:5]}...\n..." if ctx.context.analysis_result else ""

        teacher_prompt = f"""
        {lesson_plan_context_str}
        {analysis_context_str}
        TASK:
        Explain segment {segment_index} of the topic: '{topic}'.
        Use the context above and `file_search` if needed for details.
        Keep the explanation focused and concise (1-3 paragraphs ideally).
        If segment_index is 0, provide an introduction.
        Format your output as a valid LessonContent JSON object including fields 'title', 'topic', 'segment_index', 'is_last_segment', and 'text'.
        'title' should be '{lesson_plan_title}'.
        'topic' should be '{topic}'.
        'segment_index' must be {segment_index}.
        Estimate 'is_last_segment' based on whether this explanation likely concludes the topic.
        'text' should contain ONLY the explanation for segment {segment_index}.
        """
        
        result = await Runner.run(teacher_agent, teacher_prompt, context=ctx.context, run_config=run_config)
        content_output = result.final_output_as(LessonContent)
        
        if content_output and content_output.text and content_output.topic == topic:
             print(f"Orchestrator tool: Got explanation segment {segment_index} for '{topic}'")
             # Add validation for segment index if needed
             # content_output.segment_index = segment_index # Force it if agent fails
             return content_output # Return the full LessonContent object for this segment
        else:
             return f"Error: Teacher agent failed to generate explanation for {topic}."
    except Exception as e:
        print(f"Error in call_teacher_explain: {str(e)}")
        return f"Error: Failed to generate explanation - {str(e)}"


@function_tool
async def call_quiz_creator_mini(ctx: RunContextWrapper[TutorContext], topic: str) -> Union[QuizQuestion, str]: # Return question or error string
    """Generates a single multiple-choice question for the given topic using the Quiz Creator Agent."""
    print(f"Orchestrator tool: Requesting mini-quiz for topic '{topic}'")
    # Input validation
    if not topic or not isinstance(topic, str):
        return "Error: Invalid topic provided."

    try:
        # Get the quiz creator agent
        from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
        quiz_creator = create_quiz_creator_agent()

        # Setup run config for tracing
        run_config = RunConfig(workflow_name="Orchestrator_QuizCreatorCall", group_id=ctx.context.session_id)

        # Call the quiz creator agent to generate a mini-quiz
        quiz_result = await Runner.run(
            quiz_creator,
            f"Create a single multiple-choice question to test understanding of: {topic}",
            context=ctx.context,
            run_config=run_config
        )

        if quiz_result and quiz_result.questions:
             print(f"Orchestrator tool: Got mini-quiz for '{topic}'")
             # Need to handle this state update carefully. Let Orchestrator call another tool?
             current_q = quiz_result.questions[0]
             ctx.context.current_quiz_question = current_q
             # ------------------------------------------
             return current_q
        else:
            return f"Error: Quiz creator failed to generate question for {topic}."

    except Exception as e:
        error_msg = f"Error in call_quiz_creator_mini: {str(e)}"
        print(f"Orchestrator tool: {error_msg}")
        return error_msg


@function_tool
async def call_quiz_teacher_evaluate(ctx: RunContextWrapper[TutorContext], user_answer_index: int) -> Union[QuizFeedbackItem, str]: # Return feedback item or error string
    """Calls the Quiz Teacher agent to evaluate the user's answer to the question currently stored in context."""
    print(f"Orchestrator tool: Evaluating user answer index '{user_answer_index}'")

    try:
        # Get the current question from context
        question_to_evaluate = ctx.context.current_quiz_question
        if not question_to_evaluate:
            return "Error: No current question found in context."

        # Get the quiz teacher agent and evaluate
        from ai_tutor.agents.quiz_teacher_agent import create_quiz_teacher_agent, evaluate_answer
        quiz_teacher = create_quiz_teacher_agent()
        feedback_item = await evaluate_answer(
            quiz_teacher,
            question=question_to_evaluate,
            user_answer_index=user_answer_index,
            context=ctx.context # Pass context for tracing etc.
        )

        if feedback_item:
            return feedback_item
        else:
            error_msg = f"Error: Evaluation failed for question on topic '{question_to_evaluate.related_section}'."
        print(f"Orchestrator tool: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error in call_quiz_teacher_evaluate: {str(e)}"
        print(f"Orchestrator tool: {error_msg}")
        return error_msg

@function_tool
async def call_planner_get_next_topic(ctx: RunContextWrapper[TutorContext]) -> Optional[str]:
    """Determines the next topic to cover based on the lesson plan and user model."""
    print("Orchestrator tool: Getting next topic from lesson plan")

    # Get the lesson plan from context
    lesson_plan = ctx.context.lesson_plan
    if not lesson_plan:
        print("Orchestrator tool: No lesson plan found in context.")
        return None

    # Simple linear progression through concepts_to_cover for now
    all_concepts = [
        concept for section in lesson_plan.sections for concept in (section.concepts_to_cover or [])
    ]

    if not all_concepts:
        print("Orchestrator tool: No concepts found in lesson plan.")
        return None

    if not ctx.context.user_model_state.current_topic:
        next_topic = all_concepts[0]
        print(f"Orchestrator tool: Returning first topic: '{next_topic}'")
        ctx.context.user_model_state.current_topic = next_topic # Update context
        return next_topic

    try:
        current_idx = all_concepts.index(ctx.context.user_model_state.current_topic)
        if current_idx + 1 < len(all_concepts):
            next_topic = all_concepts[current_idx + 1]
            print(f"Orchestrator tool: Returning next topic: '{next_topic}'")
            ctx.context.user_model_state.current_topic = next_topic # Update context
            return next_topic
        else:
            print("Orchestrator tool: Reached end of concepts in plan.")
            ctx.context.user_model_state.current_topic = None # Update context
            return None # End of plan
    except ValueError:
        # current_topic not found in the list, default to the first topic
        print(f"Orchestrator tool: Current topic '{ctx.context.user_model_state.current_topic}' not in plan, returning first topic.")
        next_topic = all_concepts[0]
        ctx.context.user_model_state.current_topic = next_topic # Update context
        return next_topic

@function_tool
def update_user_model(ctx: RunContextWrapper[TutorContext], topic: str, outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained']) -> str:
    """Updates the user model state in the context based on an interaction outcome for a specific topic."""
    print(f"Orchestrator tool: Updating user model for topic '{topic}' with outcome '{outcome}'")

    if not topic or not isinstance(topic, str):
        return "Error: Invalid topic provided for user model update."

    # Ensure the concept exists in the user model state
    # We need the actual UserConceptMastery class here at runtime.
    # If it's defined in context.py, this *should* work fine if context.py loaded correctly first.
    # Let's import it directly here to be safe, assuming context.py can now load fully.
    from ai_tutor.context import UserConceptMastery
    if topic not in ctx.context.user_model_state.concepts:
        ctx.context.user_model_state.concepts[topic] = UserConceptMastery()

    concept_state = ctx.context.user_model_state.concepts[topic]
    concept_state.last_interaction_outcome = outcome

    # Update attempts only for evaluative outcomes
    if outcome in ['correct', 'incorrect', 'mastered', 'struggled']:
        concept_state.attempts += 1

    # Basic mastery update logic (can be refined significantly)
    if outcome == 'correct' or outcome == 'mastered':
        # Increase mastery, more boost for 'mastered'
        increment = 0.3 if outcome == 'mastered' else 0.15
        concept_state.mastery_level = min(1.0, concept_state.mastery_level + increment)
    elif outcome == 'incorrect' or outcome == 'struggled':
        # Decrease mastery, more reduction for 'struggled'
        decrement = 0.25 if outcome == 'struggled' else 0.1
        concept_state.mastery_level = max(0.0, concept_state.mastery_level - decrement)
    # 'explained' outcome doesn't directly change mastery but logs the interaction

    # Note: This modifies the context object passed by reference.
    # The updated context needs to be saved by the caller (API layer via SessionManager).
    print(f"Orchestrator tool: User model updated for '{topic}'. New mastery: {concept_state.mastery_level:.2f}")
    return f"User model updated for {topic}."

@function_tool
def update_explanation_progress(ctx: RunContextWrapper[TutorContext], segment_index: int) -> str:
    """Updates the current explanation segment index in the user model state."""
    print(f"Orchestrator tool: Updating explanation segment index to {segment_index}")
    if not isinstance(segment_index, int) or segment_index < 0:
        return "Error: Invalid segment_index provided."
    ctx.context.user_model_state.current_explanation_segment = segment_index
    ctx.context.last_interaction_summary = f"Delivered explanation segment {segment_index}"
    return f"Explanation progress updated to segment {segment_index}."

@function_tool
def get_user_model_status(ctx: RunContextWrapper[TutorContext], topic: Optional[str] = None) -> Any:
    """Retrieves the current state of the user model, optionally for a specific topic."""
    print(f"Orchestrator tool: Retrieving user model status for topic '{topic}'")

    if not ctx.context.user_model_state:
        return "Error: No user model state found in context."

    if topic:
        # Return the specific concept's state or a default if not found
        concept_state = ctx.context.user_model_state.concepts.get(topic, UserConceptMastery())
        return concept_state.model_dump(mode='json') # Return as dict
    else:
        # Return the entire user model state
        return ctx.context.user_model_state.model_dump(mode='json') # Return as dict 