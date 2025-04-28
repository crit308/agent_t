import json
import inspect # Need inspect module
from agents.run_context import RunContextWrapper
from typing import Any, TYPE_CHECKING
from agents.tool import FunctionTool # Import the class
import logging

# Conditional import for type checking
if TYPE_CHECKING:
    from ai_tutor.context import TutorContext

logger = logging.getLogger(__name__)


async def invoke(tool_or_func: Any, ctx: 'TutorContext', **kwargs) -> Any:
    """
    Uniformly call an async skill/function, handling whether it's a
    direct function or an ADK FunctionTool object.
    Returns the raw Python object returned by the skill.
    Ensures context (raw or wrapped) is passed correctly based on signature.
    """
    target_func = tool_or_func
    tool_name = "Unknown Function"
    is_adk_tool = isinstance(tool_or_func, FunctionTool)

    if is_adk_tool:
        tool_name = getattr(tool_or_func, 'name', tool_name)
        # Try to get the original function attached by the @function_tool decorator
        if hasattr(tool_or_func, '__original_func__'):
            target_func = tool_or_func.__original_func__
            logger.debug(f"Invoke: Found original function for ADK tool {tool_name}")
        else:
            # Fallback or error if original func isn't found
            raise TypeError(f"ADK FunctionTool {tool_name} does not have __original_func__")
    else:
        tool_name = getattr(target_func, '__name__', tool_name)

    if not callable(target_func):
        raise TypeError(f"Target '{tool_name}' ({target_func}) is not callable.")

    if not inspect.iscoroutinefunction(target_func):
         raise TypeError(f"Target function '{tool_name}' is not async.")

    # --- Determine how to pass context --- 
    sig = inspect.signature(target_func)
    pass_wrapper = False
    context_arg_name = None
    takes_context = False

    # Check parameters for context argument (wrapper or raw)
    for param_name, param in sig.parameters.items():
        # Check for RunContextWrapper annotation (direct class reference)
        if (inspect.isclass(param.annotation) and issubclass(param.annotation, RunContextWrapper)):
            pass_wrapper = True
            context_arg_name = param_name
            takes_context = True
            logger.debug(f"Invoke: Target '{tool_name}' expects RunContextWrapper for arg '{context_arg_name}' (direct class)")
            break  # Highest-priority match

        # --- NEW: Handle typing generics e.g. RunContextWrapper[TutorContext] ---
        if not takes_context:
            anno_str = str(param.annotation)
            # In Python 3.9+, generic alias string looks like 'agents.run_context.RunContextWrapper[ai_tutor.context.TutorContext]'
            if anno_str.startswith('agents.run_context.RunContextWrapper') or 'RunContextWrapper' in anno_str:
                pass_wrapper = True
                context_arg_name = param_name
                takes_context = True
                logger.debug(
                    f"Invoke: Target '{tool_name}' expects RunContextWrapper for arg '{context_arg_name}' (string match)"
                )
                break

        # Check for raw TutorContext annotation
        # Note: This requires TutorContext to be imported or handled via TYPE_CHECKING
        elif inspect.isclass(param.annotation) and 'TutorContext' in str(param.annotation):  # Check if TutorContext is the type
            context_arg_name = param_name
            takes_context = True
            logger.debug(f"Invoke: Target '{tool_name}' expects raw TutorContext for arg '{context_arg_name}'")
            break
        # Fallback check for parameter named 'ctx'
        elif param_name == 'ctx':
             context_arg_name = param_name
             takes_context = True
             logger.debug(f"Invoke: Target '{tool_name}' has parameter named 'ctx', assuming raw context.")
             # Don't break here, prefer explicit type annotation if found later

    # Call the function
    logger.debug(f"Invoke: Calling '{tool_name}' with kwargs: {list(kwargs.keys())}")
    try:
        if takes_context and context_arg_name:
             if pass_wrapper:
                 # Pass RunContextWrapper explicitly using the found argument name
                 call_kwargs = {**{context_arg_name: RunContextWrapper(ctx)}, **kwargs}
                 return await target_func(**call_kwargs)
             else:
                 # Pass raw context using the found argument name
                 call_kwargs = {**{context_arg_name: ctx}, **kwargs}
                 return await target_func(**call_kwargs)
        else: # No context argument detected
             logger.debug(f"Invoke: Target '{tool_name}' does not seem to take context based on signature analysis.")
             return await target_func(**kwargs)

    except Exception as call_err:
         logger.error(f"Invoke: Error calling target function '{tool_name}': {call_err}", exc_info=True)
         raise # Re-raise the original error 