from __future__ import annotations
from agents import function_tool, Runner, RunConfig
from agents.run_context import RunContextWrapper
from typing import Any, Optional, Literal, Union
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

# Import the specific generate_lesson_content function
from ai_tutor.agents.teacher_agent import generate_lesson_content # Import specific function if needed

# Import API response models for potentially formatting tool output
from ai_tutor.api_models import (
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse
)

# --- Orchestrator Tool Implementations ---

@function_tool
async def call_teacher_explain(ctx: RunContextWrapper[TutorContext], topic: str) -> Union[str, ErrorResponse]:
    """Generates an explanation for a specific topic using the Teacher Agent."""
    print(f"Orchestrator tool: Requesting explanation for topic '{topic}'")

    # Input validation
    if not topic or not isinstance(topic, str):
        return "Error: Invalid topic provided."
    if not ctx.context.vector_store_id:
         return "Error: Vector store ID not found in context. Cannot research topic."
    if not ctx.context.lesson_plan:
         return "Error: Lesson plan not found in context. Cannot generate explanation without plan context."

    try:
        # Use the non-handoff teacher agent to generate content JUST for this topic plan
        teacher_agent = create_teacher_agent_without_handoffs(ctx.context.vector_store_id)
        run_config = RunConfig(workflow_name="Orchestrator_TeacherCall", group_id=ctx.context.session_id)

        # generate_lesson_content expects an Agent and LessonPlan
        # It internally calls Runner.run
        lesson_content_result : LessonContent = await generate_lesson_content(
            teacher_agent=teacher_agent,
            lesson_plan=ctx.context.lesson_plan, # Pass the *original* full plan for context
            topic_to_explain=topic, # Pass the specific topic
            context=ctx.context
        )

        if lesson_content_result and lesson_content_result.text:
             print(f"Orchestrator tool: Got explanation for '{topic}'")
             # Update context - mark topic as 'explained' or similar?
             ctx.context.last_interaction_summary = f"Explained topic: {topic}"
             return lesson_content_result.text
        else:
             return f"Error: Teacher agent failed to generate explanation for {topic}."

    except Exception as e:
        print(f"ERROR in call_teacher_explain for '{topic}': {e}")
        return f"An error occurred while generating the explanation for {topic}."


@function_tool
async def call_quiz_creator_mini(ctx: RunContextWrapper[TutorContext], topic: str) -> Union[QuizQuestion, str]:
    """Generates a single multiple-choice question for the given topic using the Quiz Creator Agent."""
    print(f"Orchestrator tool: Requesting mini-quiz for topic '{topic}'")
    # Input validation
    if not topic or not isinstance(topic, str):
        # Return a dummy error question or raise? Let's return dummy.
        return "Error: Invalid topic provided for quiz."

    try:
        quiz_creator_agent = create_quiz_creator_agent() # Non-handoff version
        run_config = RunConfig(workflow_name="Orchestrator_QuizMiniCall", group_id=ctx.context.session_id)

        # Construct a very specific quiz_prompt telling the QuizCreatorAgent to create exactly one question
        quiz_prompt = (
            f"Based on the topic '{topic}', create ONLY ONE simple multiple-choice question "
            f"with 4 options. The question should test basic understanding of '{topic}'. "
            "Your output MUST be a valid Quiz object containing EXACTLY ONE question in its 'questions' list. "
            f"Set the 'related_section' for the question to '{topic}'."
        )
        result = await Runner.run(quiz_creator_agent, quiz_prompt, context=ctx.context, run_config=run_config)
        quiz_result = result.final_output_as(Quiz) # Expecting Quiz object

        if quiz_result and quiz_result.questions:
             print(f"Orchestrator tool: Got mini-quiz for '{topic}'")
             ctx.context.last_interaction_summary = f"Asked mini-quiz for topic: {topic}"
             # --- Store the asked question in context ---
             current_q = quiz_result.questions[0]
             ctx.context.current_quiz_question = current_q
             # ------------------------------------------
             # Return only the first (and hopefully only) question
             return current_q
        else:
             error_msg = f"Error: Could not generate quiz question for {topic}."
             print(f"Orchestrator tool: {error_msg} Result: {result.final_output}")
             return error_msg

    except Exception as e:
        error_msg = f"An error occurred while generating the quiz question for {topic}."
        print(f"ERROR in call_quiz_creator_mini: {error_msg} - {e}")
        return error_msg


@function_tool
async def call_quiz_teacher_evaluate(ctx: RunContextWrapper[TutorContext], user_answer_index: int) -> Union[QuizFeedbackItem, str]:
    """Calls the Quiz Teacher agent to evaluate the user's answer to the question currently stored in context."""
    print(f"Orchestrator tool: Evaluating user answer index '{user_answer_index}'")

    question_to_evaluate = ctx.context.current_quiz_question

    if not question_to_evaluate:
        error_msg = "Error: No current quiz question found in context to evaluate."
        print(f"Orchestrator tool: {error_msg}")
        return error_msg

    try:
        feedback_item = await evaluate_single_answer(
            question=question_to_evaluate,
            user_answer_index=user_answer_index,
            context=ctx.context # Pass context for tracing etc.
        )
        ctx.context.last_interaction_summary = f"Evaluated answer for question: {question_to_evaluate.question[:50]}..."
        return feedback_item
    except ValueError:
        error_msg = f"Error: Invalid answer index '{user_answer_index}' provided."
        print(f"Orchestrator tool: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"An error occurred while evaluating the answer: {e}"
        print(f"ERROR in call_quiz_teacher_evaluate: {error_msg}")
        return error_msg

@function_tool
async def call_planner_get_next_topic(ctx: RunContextWrapper[TutorContext], current_topic: Optional[str] = None) -> Optional[str]:
    """Consults the LessonPlan in the context to determine the next topic to cover."""
    print(f"Orchestrator tool: Getting next topic after '{current_topic}'")
    lesson_plan = ctx.context.lesson_plan
    if not lesson_plan or not lesson_plan.sections:
        print("Orchestrator tool: No lesson plan or sections found.")
        return None

    # Simple linear progression through concepts_to_cover for now
    all_concepts = [concept for section in lesson_plan.sections for concept in section.concepts_to_cover]

    if not all_concepts:
        print("Orchestrator tool: No concepts found in lesson plan.")
        return None

    if not current_topic:
        next_topic = all_concepts[0]
        print(f"Orchestrator tool: Returning first topic: '{next_topic}'")
        ctx.context.user_model_state.current_topic = next_topic # Update context
        return next_topic

    try:
        current_idx = all_concepts.index(current_topic)
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
        print(f"Orchestrator tool: Current topic '{current_topic}' not in plan, returning first topic.")
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
def get_user_model_status(ctx: RunContextWrapper[TutorContext], topic: Optional[str] = None) -> Any:
    """Retrieves the current state of the user model, optionally for a specific topic."""
    print(f"Orchestrator tool: Retrieving user model status for topic '{topic}'")

    # Need UserConceptMastery at runtime here too.
    from ai_tutor.context import UserConceptMastery

    if topic:
        # Return the specific concept's state or a default if not found
        concept_state = ctx.context.user_model_state.concepts.get(topic, UserConceptMastery())
        return concept_state.model_dump(mode='json')
    else:
        # Return the entire user model state
        return ctx.context.user_model_state.model_dump(mode='json') 