from agents import HandoffInputData
import pprint
import json
# import decimal # Not needed if converting directly
import copy
from decimal import Decimal # Keep Decimal for potential input types
import re
from typing import Any, Dict, List, Union, Tuple, AsyncIterator

# Import additional classes needed for the wrapper
from agents import Model, ModelResponse, ModelSettings, ModelTracing, AgentOutputSchema, Handoff, Tool
from agents.items import TResponseInputItem, TResponseStreamEvent

# --- Helper for recursive rounding ---
def _recursive_round_scores(data: Any, max_decimals: int = 8):
    """
    Recursively traverses data structures (dict, list, tuple) and returns a new
    structure where any float value associated with a key named 'score'
    is rounded to max_decimals. Returns original data if no rounding occurs.
    Handles nested structures and avoids deepcopying non-pickleable objects.

    NOTE: This is necessary because file_search results can contain scores with
    precision exceeding the OpenAI API's limit (currently 16 decimals) when
    included in the input history for subsequent calls. This function aims to
    fix ONLY the 'score' fields. Avoid overly broad recursion or deepcopy.
    """
    # Handle dictionaries
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # If this is a 'score' key with a float value, round it
            if key == 'score' and isinstance(value, float):
                # First round the value properly
                rounded_val = round(value, max_decimals)
                # Format with exact decimal places
                result[key] = float(f"{rounded_val:.{max_decimals}f}")
            elif isinstance(value, (dict, list, tuple)):
                # Recursively process nested structures
                result[key] = _recursive_round_scores(value, max_decimals)
            else:
                # Keep other values as is
                result[key] = value
        return result
    
    # Handle lists and tuples
    elif isinstance(data, (list, tuple)):
        result = []
        for item in data:
            if isinstance(item, (dict, list, tuple)):
                # Recursively process nested structures
                result.append(_recursive_round_scores(item, max_decimals))
            else:
                # Keep primitive values as is
                result.append(item)
        
        # Convert back to tuple if original was a tuple
        if isinstance(data, tuple):
            return tuple(result)
        return result
    
    # Return the original data if it's not a dict/list/tuple
    else:
        return data

# --- Add RoundingModelWrapper Class ---
class RoundingModelWrapper(Model):
    """Wraps another Model to apply score rounding before API calls."""
    def __init__(self, underlying_model: Model, max_decimals: int = 8):
        self._underlying_model = underlying_model
        self._max_decimals = max_decimals
        print(f"Initialized RoundingModelWrapper for {underlying_model.__class__.__name__}")

    def _clean_input(self, input_data: Union[str, List[TResponseInputItem]]) -> Union[str, List[TResponseInputItem]]:
        """Applies recursive rounding to the input history."""
        if isinstance(input_data, str):
            return input_data # No scores in a simple string prompt

        # Process the input without deepcopy, only modifying when needed
        cleaned_input = _recursive_round_scores(input_data, self._max_decimals)
        # print("DEBUG: Cleaned input history for model call.") # Optional Debug
        return cleaned_input

    async def get_response(
        self,
        system_instructions: str | None,
        input: Union[str, List[TResponseInputItem]],
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: List[Handoff],
        tracing: ModelTracing,
    ) -> ModelResponse:
        cleaned_input = self._clean_input(input)
        # print("DEBUG: Calling wrapped get_response") # Optional Debug
        return await self._underlying_model.get_response(
            system_instructions=system_instructions,
            input=cleaned_input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: Union[str, List[TResponseInputItem]],
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: List[Handoff],
        tracing: ModelTracing,
    ) -> AsyncIterator[TResponseStreamEvent]:
        cleaned_input = self._clean_input(input)
        # print("DEBUG: Calling wrapped stream_response") # Optional Debug
        # Use 'async for' correctly for an async iterator
        async for event in self._underlying_model.stream_response(
            system_instructions=system_instructions,
            input=cleaned_input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
        ):
            yield event
# ------------------------------------

def process_handoff_data(handoff_data):
    """
    Process HandoffInputData ensuring float precision for scores and forcing input_history
    to be a tuple of structured items. Avoids deepcopy.

    IMPORTANT: This filter includes rounding for 'score' fields found anywhere
    in the input history to prevent OpenAI API errors related to excessive
    decimal places, particularly from file_search results.
    """
    print("Processing handoff data structure and rounding 'score' fields...")
    MAX_DECIMALS = 8 # Safe limit well below API's 16
    
    # Original method used deepcopy which can fail for non-pickleable objects
    # So we use our targeted recursive_round_scores function instead
    try:
        # Replace with selective recursive rounding just for score fields
        result = HandoffInputData(
            # Round values in input
            input_value=handoff_data.input_value,
            # Process input history carefully (most likely cause of issues)
            input_history=_recursive_round_scores(handoff_data.input_history, MAX_DECIMALS),
            # Agent config is likely minimal, just keep as is
            agent_config=handoff_data.agent_config
        )
        return result
    except Exception as e:
        print(f"Error in process_handoff_data: {e}")
        # If anything fails, return the original unmodified data
        return handoff_data 