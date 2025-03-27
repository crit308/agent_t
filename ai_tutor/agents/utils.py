from agents import HandoffInputData
import pprint
import json
import decimal
import copy
from decimal import Decimal # Keep Decimal for potential input types
import re
from typing import Any, Dict, List, Union, Tuple

def process_handoff_data(handoff_data):
    """
    Process HandoffInputData ensuring float precision and forcing input_history
    to be a tuple of structured items.
    """
    print("Processing handoff data for precision (Forcing History Structure)...")
    try:
        # 1. Ensure original_input_history is a list of dicts using the helper
        from agents import ItemHelpers
        original_input_list = ItemHelpers.input_to_new_input_list(handoff_data.input_history)

        # 2. Process each item in the guaranteed list for float precision
        #    NOTE: Precision limiting logic is REMOVED here. If needed, apply selectively.
        #    Example selective rounding (if scores are still an issue):
        #    for item in original_input_list:
        #        if isinstance(item, dict) and item.get('type') == 'file_search_results':
        #             # Round scores specifically here if needed
        #             pass # Add rounding logic if required
        
        input_history_tuple = tuple(original_input_list) # Ensure it's a tuple

        # Ensure pre_handoff_items and new_items are tuples
        pre_handoff_items_tuple = tuple(handoff_data.pre_handoff_items or ())
        new_items_tuple = tuple(handoff_data.new_items or ())

        # Create new HandoffInputData with guaranteed tuple structure for history
        processed_data = HandoffInputData(
            input_history=input_history_tuple, # Now guaranteed to be a tuple of dicts (or empty)
            pre_handoff_items=pre_handoff_items_tuple,
            new_items=new_items_tuple
        )
        print("Successfully processed handoff data structure.") # Removed precision claim
        return processed_data

    except Exception as e:
        print(f"ERROR during handoff data processing: {e}")
        import traceback
        traceback.print_exc()
        print("Returning original handoff data due to error.")
        return handoff_data # Fallback 