"""Tools for the AI Tutor system.""" 

from __future__ import annotations
from agents import function_tool, Runner, RunConfig
from agents.run_context import RunContextWrapper
from typing import Any, Optional, Literal, Union, cast, Dict, List
import os
from datetime import datetime
import traceback
from ai_tutor.dependencies import SUPABASE_CLIENT
from ai_tutor.utils.tool_helpers import invoke
from ai_tutor.context import TutorContext, UserConceptMastery, UserModelState, is_mastered
from ai_tutor.agents.models import FocusObjective, QuizQuestion, QuizFeedbackItem, ExplanationResult, QuizCreationResult, PlannerOutput
from ai_tutor.api_models import (
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse
)
from ai_tutor.errors import ToolExecutionError
import json
from ai_tutor.telemetry import log_tool
from pydantic import BaseModel
from ai_tutor.utils.decorators import function_tool_logged

_exports: List[str] = []
def _export(obj):
    # Support both function and FunctionTool objects
    name = getattr(obj, '__name__', None)
    if name is None and hasattr(obj, '__original_func__'):
        name = getattr(obj.__original_func__, '__name__', None)
    if name:
        _exports.append(name)
    return obj

# --- Orchestrator Tool Implementations ---

@_export
@function_tool_logged()
async def healthz(ctx) -> str:
    """Ping‑pong tool so orchestrator traces always include a baseline call."""
    return "ok"

@_export
@function_tool_logged()
def update_explanation_progress(ctx: RunContextWrapper[TutorContext], segment_index: int) -> str:
    """DEPRECATED: The Orchestrator manages micro-steps directly."""
    return "Error: This tool is deprecated. Orchestrator manages micro-steps."

@_export
@function_tool_logged()
async def call_quiz_teacher_evaluate(ctx: RunContextWrapper[TutorContext], user_answer_index: int) -> Union[QuizFeedbackItem, str]:
    """Evaluates the user's answer to the current question using the Quiz Teacher logic (via helper function)."""
    print(f"[Tool call_quiz_teacher_evaluate] Evaluating user answer index '{user_answer_index}'.")
    try:
        from ai_tutor.agents.quiz_teacher_agent import evaluate_single_answer
        question_to_evaluate = ctx.context.current_quiz_question
        if not question_to_evaluate:
            raise ToolExecutionError("No current question found in context to evaluate.", code="missing_question")
        if not isinstance(question_to_evaluate, QuizQuestion):
            raise ToolExecutionError(f"Expected QuizQuestion in context, found {type(question_to_evaluate).__name__}.", code="invalid_question_type")
        print(f"[Tool call_quiz_teacher_evaluate] Evaluating answer for question: {question_to_evaluate.question[:50]}...")
        feedback_item = await evaluate_single_answer(
            question=question_to_evaluate,
            user_answer_index=user_answer_index,
            context=ctx.context
        )
        if feedback_item:
            print(f"[Tool call_quiz_teacher_evaluate] Evaluation complete. Feedback: Correct={feedback_item.is_correct}, Explanation: {feedback_item.explanation[:50]}...")
            ctx.context.user_model_state.pending_interaction_type = None
            ctx.context.user_model_state.pending_interaction_details = None
            ctx.context.current_quiz_question = None
            return feedback_item
        else:
            error_msg = f"Evaluation failed for question on topic '{getattr(question_to_evaluate, 'related_section', 'N/A')}'."
            print(f"[Tool] {error_msg}")
            raise ToolExecutionError(error_msg, code="evaluation_failed")
    except Exception as e:
        error_msg = f"Exception in call_quiz_teacher_evaluate: {str(e)}"
        print(f"[Tool] {error_msg}")
        raise ToolExecutionError(error_msg, code="exception")

@_export
@function_tool_logged()
async def update_user_model(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained'],
    confusion_point: Optional[str] = None,
    last_accessed: Optional[str] = None,
    mastered_objective_title: Optional[str] = None,
) -> str:
    """Updates the user model state with interaction outcomes and temporal data."""
    print(f"[Tool update_user_model] Updating '{topic}' with outcome '{outcome}'")
    if not ctx.context or not ctx.context.user_model_state:
        raise ToolExecutionError("TutorContext or UserModelState not found.", code="missing_context")
    if not topic or not isinstance(topic, str):
        raise ToolExecutionError("Invalid topic provided for user model update.", code="invalid_topic")
    if topic not in ctx.context.user_model_state.concepts:
        ctx.context.user_model_state.concepts[topic] = UserConceptMastery()
    concept_state = ctx.context.user_model_state.concepts[topic]
    old_mastery = concept_state.mastery
    previously_mastered = is_mastered(concept_state)
    concept_state.last_interaction_outcome = outcome
    concept_state.last_accessed = last_accessed or datetime.now().isoformat()
    if confusion_point and confusion_point not in concept_state.confusion_points:
        concept_state.confusion_points.append(confusion_point)
    if outcome in ['correct', 'incorrect', 'mastered', 'struggled']:
        concept_state.attempts += 1
        if outcome in ['correct', 'mastered']:
            concept_state.alpha += 1
        else:
            concept_state.beta += 1
            if len(concept_state.confusion_points) > 2:
                ctx.context.user_model_state.learning_pace_factor = max(
                    0.5, ctx.context.user_model_state.learning_pace_factor - 0.1
                )
    if mastered_objective_title and mastered_objective_title not in ctx.context.user_model_state.mastered_objectives_current_section:
        ctx.context.user_model_state.mastered_objectives_current_section.append(mastered_objective_title)
        print(f"[Tool] Marked objective '{mastered_objective_title}' as mastered for current section.")
    print(f"[Tool] Updated '{topic}' - Mastery: {concept_state.mastery:.2f}, "
          f"Confidence: {concept_state.confidence}, "
          f"Pace: {ctx.context.user_model_state.learning_pace_factor:.2f}")
    if not previously_mastered and is_mastered(concept_state):
        print(f"[Tool update_user_model] Mastery achieved for '{topic}'. Triggering planner agent.")
        from ai_tutor.tools import call_planner_agent
        if ctx.context.current_focus_objective is None:
            await invoke(call_planner_agent, ctx)
    mastery = concept_state.mastery
    confidence = concept_state.confidence
    if hasattr(ctx.context, "ws"):
        try:
            await ctx.context.ws.send_json({"type": "mastery_update", "topic": topic, "mastery": mastery, "confidence": confidence})
        except Exception:
            pass
    new_mastery = mastery
    delta_mastery = new_mastery - old_mastery
    if SUPABASE_CLIENT:
        try:
            SUPABASE_CLIENT.table("concept_events").insert({
                "session_id": str(ctx.context.session_id),
                "user_id": str(ctx.context.user_id),
                "concept": topic,
                "outcome": outcome,
                "delta_mastery": delta_mastery
            }).execute()
        except Exception as e:
            print(f"[Tool update_user_model] Failed to log concept event: {e}")
    return f"User model updated for {topic}."

@_export
@function_tool_logged()
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
            "mastery_level": concept.mastery,
            "attempts": concept.attempts,
            "last_outcome": concept.last_interaction_outcome,
            "confusion_points": concept.confusion_points,
            "last_accessed": concept.last_accessed
        }
    return state.model_dump(mode='json')

@_export
@function_tool_logged()
async def reflect_on_interaction(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    interaction_summary: str,
    user_response: Optional[str] = None,
    feedback_provided: Optional[QuizFeedbackItem] = None
) -> Dict[str, Any]:
    class _Params(BaseModel):
        topic: str
        interaction_summary: str
        user_response: Optional[str] = None
        feedback_provided: Optional[QuizFeedbackItem] = None
        model_config = {"extra": "forbid"}
    print(f"[Tool reflect_on_interaction] Called for topic '{topic}'. Summary: {interaction_summary}")
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
    else:
        analysis += "Interaction seems positive or neutral."
        suggestions.append("Proceed with the next logical step in the micro-plan (e.g., next segment, checking question).")
    print(f"[Tool reflect_on_interaction] Analysis: {analysis}. Suggestions: {suggestions}")
    return {"analysis": analysis, "suggested_next_steps": suggestions}

@_export
@function_tool_logged()
async def call_planner_agent(
    ctx: RunContextWrapper[TutorContext],
    user_state_summary: Optional[str] = None
) -> Union[PlannerOutput, str]:
    """Calls the Planner Agent to determine the next learning focus objective. Returns PlannerOutput on success, or an error string on failure."""
    print("[Tool call_planner_agent] Calling Planner Agent...")
    try:
        from ai_tutor.agents.planner_agent import create_planner_agent
        if not ctx.context.vector_store_id:
            raise ToolExecutionError("Vector store ID not found in context for Planner.", code="missing_vector_store")
        planner_agent = create_planner_agent(ctx.context.vector_store_id)
        run_config = RunConfig(workflow_name="Orchestrator_PlannerCall", group_id=ctx.context.session_id)
        mastered_topics = [t for t, c in ctx.context.user_model_state.concepts.items()
                          if c.mastery >= 0.8 and c.confidence >= 5]
        payload = json.dumps({"exclude_topics": mastered_topics, "goal": ctx.context.session_goal})
        planner_prompt = f"""
        Determine the next learning focus for the user.
        First, call `read_knowledge_base` to understand the material's structure and concepts.
        Analyze the knowledge base.
        {f'Consider the user state: {user_state_summary}' if user_state_summary else 'Assume the user is starting or has just completed the previous focus.'}
        Exclude these topics from consideration: {mastered_topics}
        Identify the single most important topic or concept for the user to focus on next.
        Output your decision ONLY as a PlannerOutput object.
        """
        # Run the planner agent; output will be raw JSON string since no output_type is enforced
        result = await Runner.run(
            planner_agent,
            planner_prompt,
            context=ctx.context,
            run_config=run_config
        )
        raw_output = result.final_output
        # Parse into PlannerOutput model, extracting JSON if embedded in prose or code fences
        if isinstance(raw_output, PlannerOutput):
            planner_output = raw_output
        elif isinstance(raw_output, str):
            raw_text = raw_output.strip()
            # Try to extract full JSON block between code fences
            import re
            json_str = None
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text, re.DOTALL)
            if fence_match:
                json_str = fence_match.group(1).strip()
            else:
                # Fallback: find first '{' and last '}'
                start = raw_text.find('{')
                end = raw_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = raw_text[start:end+1]
            if not json_str:
                error_msg = f"PLANNER_PARSING_ERROR: Could not locate JSON object in planner output. Raw output: {raw_text}"  
                print(f"[Tool call_planner_agent] {error_msg}")
                raise ToolExecutionError(error_msg, code="planner_output_parse_error")
            try:
                planner_output = PlannerOutput.parse_raw(json_str)
            except Exception as e:
                error_msg = f"PLANNER_PARSING_ERROR: Failed to parse PlannerOutput JSON: {e}. JSON snippet: {json_str}"  
                print(f"[Tool call_planner_agent] {error_msg}")
                raise ToolExecutionError(error_msg, code="planner_output_parse_error")
        else:
            error_msg = f"Planner agent returned unexpected type: {type(raw_output).__name__}. Raw output: {raw_output}"
            print(f"[Tool call_planner_agent] {error_msg}")
            raise ToolExecutionError(error_msg, code="planner_output_error")
        # Success: store and return
        print(f"[Tool call_planner_agent] Planner returned objective: {planner_output.objective.topic}")
        ctx.context.current_focus_objective = planner_output.objective
        return planner_output
    except Exception as e:
        error_msg = f"EXCEPTION calling Planner Agent: {str(e)}\n{traceback.format_exc()}"
        print(f"[Tool] {error_msg}")
        raise ToolExecutionError("PLANNER_EXECUTION_ERROR: An exception occurred while running the planner.", code="planner_exception")

@_export
@function_tool_logged()
async def call_teacher_agent(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    explanation_details: str
) -> Union[ExplanationResult, str]:
    """Calls the Teacher Agent to provide an explanation for a specific topic/detail."""
    print(f"[Tool call_teacher_agent] Requesting explanation for '{topic}': {explanation_details}")
    try:
        from ai_tutor.agents.teacher_agent import create_interactive_teacher_agent
        if not ctx.context.vector_store_id:
            raise ToolExecutionError("Vector store ID not found in context for Teacher.", code="missing_vector_store")
        teacher_agent = create_interactive_teacher_agent(ctx.context.vector_store_id)
        run_config = RunConfig(workflow_name="Orchestrator_TeacherCall", group_id=ctx.context.session_id)
        teacher_prompt = f"""
        Explain the topic: '{topic}'.
        Specific instructions for this explanation: {explanation_details}.
        Use the file_search tool if needed to find specific information or examples from the documents.
        Format your response ONLY as an ExplanationResult object containing the explanation text in the 'details' field.
        """
        from ai_tutor.agents.agent_runner import AgentRunner
        runner = AgentRunner(teacher_agent, ctx.context, ctx.context.vector_store_id)
        return await runner.run_teacher(teacher_prompt, section=explanation_details)
    except Exception as e:
        error_msg = f"Error calling Teacher Agent: {str(e)}\n{traceback.format_exc()}"
        print(f"[Tool] {error_msg}")
        raise ToolExecutionError(error_msg, code="teacher_exception")

@_export
@function_tool_logged()
async def call_quiz_creator_agent(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    instructions: str
) -> Union[QuizCreationResult, str]:
    """Calls the Quiz Creator Agent to generate one or more quiz questions."""
    print(f"[Tool call_quiz_creator_agent] Requesting quiz creation for '{topic}': {instructions}")
    try:
        from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
        quiz_creator_agent = create_quiz_creator_agent()
        run_config = RunConfig(workflow_name="Orchestrator_QuizCreatorCall", group_id=ctx.context.session_id)
        quiz_creator_prompt = f"""
        Create quiz questions based on the following instructions:
        Topic: '{topic}'
        Instructions: {instructions}
        Format your response ONLY as a QuizCreationResult object. Include the created question(s) in the appropriate field ('question' or 'quiz').
        """
        from ai_tutor.agents.agent_runner import AgentRunner
        runner = AgentRunner(quiz_creator_agent, ctx.context, ctx.context.vector_store_id)
        mastered_sections = ctx.context.user_model_state.mastered_objectives_current_section or []
        quiz_creation_result = await runner.run_quiz_creator(quiz_creator_prompt, mastered_sections)
            return quiz_creation_result
    except Exception as e:
        error_msg = f"Error calling Quiz Creator Agent: {str(e)}\n{traceback.format_exc()}"
        print(f"[Tool] {error_msg}")
        raise ToolExecutionError(error_msg, code="quiz_creator_exception")

__all__ = _exports 