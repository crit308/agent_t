# ai_tutor/skills/update_user_model.py
# from agents import function_tool # No longer used
from ai_tutor.skills import skill # Import correct decorator
from typing import Optional, Literal
from ai_tutor.context import TutorContext, UserConceptMastery, UserModelState
from agents.run_context import RunContextWrapper
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

@skill
async def update_user_model(
    ctx: RunContextWrapper[TutorContext],
    topic: str,
    outcome: Literal['correct', 'incorrect', 'unsure', 'clarification_needed', 'explained'],
    details: Optional[str] = None
) -> UserModelState:
    """Updates the user's mastery model for a given topic based on the interaction outcome.
    
    Modifies ctx.user_model_state.concepts[topic] by updating alpha/beta
    based on 'correct'/'incorrect' outcomes, increments attempts, and updates timestamps.
    Returns the modified UserModelState object.
    """
    logger.info(f"Updating user model for topic: '{topic}', outcome: '{outcome}'")
    
    # Ensure user model state and concepts dictionary exist
    if not ctx.context.user_model_state:
        logger.warning("User model state not found in context. Initializing.")
        ctx.context.user_model_state = UserModelState()
    if not isinstance(ctx.context.user_model_state.concepts, dict):
         logger.warning("User model concepts dictionary not found. Initializing.")
         ctx.context.user_model_state.concepts = {}

    # Get or create the concept state
    if topic not in ctx.context.user_model_state.concepts:
        logger.info(f"First interaction with topic '{topic}'. Creating new concept state.")
        ctx.context.user_model_state.concepts[topic] = UserConceptMastery()
        
    concept_state = ctx.context.user_model_state.concepts[topic]
    
    # Update based on outcome
    if outcome == 'correct':
        concept_state.alpha += 1
        concept_state.attempts += 1
        logger.info(f"Incremented alpha for '{topic}'. New alpha={concept_state.alpha}")
    elif outcome == 'incorrect':
        concept_state.beta += 1
        concept_state.attempts += 1
        if details:
            # Add confusion point if provided and not already present
            if details not in concept_state.confusion_points:
                 concept_state.confusion_points.append(details)
                 logger.info(f"Added confusion point for '{topic}': {details}")
            else:
                 logger.info(f"Confusion point '{details}' already recorded for '{topic}'.")
        logger.info(f"Incremented beta for '{topic}'. New beta={concept_state.beta}")
    elif outcome == 'explained' or outcome == 'clarification_needed' or outcome == 'unsure':
        # These outcomes don't directly update alpha/beta but mark interaction
        concept_state.attempts += 1 # Still counts as an interaction attempt
        logger.info(f"Recorded interaction ('{outcome}') for topic '{topic}'. No alpha/beta change.")
    else:
        logger.warning(f"Unhandled outcome type '{outcome}' for topic '{topic}'. No update performed.")
        # Do not increment attempts for unhandled outcomes

    # Update common fields
    concept_state.last_interaction_outcome = outcome
    # Update the last accessed timestamp with an ISO string
    concept_state.last_accessed = datetime.now(timezone.utc).isoformat()

    # Update current_topic in overall state
    ctx.context.user_model_state.current_topic = topic

    logger.info(f"User model for '{topic}' updated: Mastery={concept_state.mastery:.3f}, Confidence={concept_state.confidence}, Attempts={concept_state.attempts}")
    return ctx.context.user_model_state # Return the modified state object 