from agents import HandoffInputData
import pprint
import json
import decimal
import copy
from decimal import Decimal, getcontext
import re
from typing import Any, Dict, List, Union, Tuple

def limit_decimal_places(data, max_places=8):
    """
    Recursively ensure no floating point value exceeds the specified number of decimal places
    within standard data structures (dicts, lists, tuples).
    Uses string formatting for robust precision control.

    Args:
        data: Any data structure (dict, list, tuple, primitive).
        max_places: Maximum decimal places allowed (defaulting to 8 for safety).

    Returns:
        Modified data structure (if dict/list/tuple) or processed value.
    """
    if isinstance(data, (float, decimal.Decimal)):
        # Convert to float first (handles Decimal), then format, then convert back
        try:
            # Use string formatting to ensure exactly max_places
            formatted_string = f"{float(data):.{max_places}f}"
            limited_float = float(formatted_string)
            # Double-check string representation in case of rare edge cases
            str_val = str(limited_float)
            if '.' in str_val and len(str_val.split('.')[1]) > max_places:
                # Fallback truncation if formatting somehow failed
                int_part, decimal_part = str_val.split('.', 1)
                corrected_float = float(f"{int_part}.{decimal_part[:max_places]}")
                print(f"DEBUG: Corrected float edge case: {data} -> {corrected_float}")
                return corrected_float
            print(f"DEBUG: Limited float: {data} -> {limited_float}")
            return limited_float
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not limit decimal places for value {data}: {e}")
            # Return original value if conversion fails
            return data
    elif isinstance(data, dict):
        # Recursively process dictionary values
        return {k: limit_decimal_places(v, max_places) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively process list items
        return [limit_decimal_places(item, max_places) for item in data]
    elif isinstance(data, tuple):
        # Recursively process tuple items
        return tuple(limit_decimal_places(item, max_places) for item in data)
    else:
        # Return data unchanged if it's not a float, dict, list, tuple, or known object type
        return data

def process_handoff_data(handoff_data):
    """
    Process HandoffInputData ensuring float precision and forcing input_history
    to be a tuple of structured items.
    """
    print("Processing handoff data for precision (Forcing History Structure)...")
    try:
        # --- Process input_history ---
        processed_history_list = []
        # 1. Ensure original_input_history is a list of dicts using the helper
        from agents import ItemHelpers
        original_input_list = ItemHelpers.input_to_new_input_list(handoff_data.input_history)

        # 2. Process each item in the guaranteed list for float precision
        if original_input_list:
            for item in original_input_list:
                # Apply limit_decimal_places recursively within a deep copy of the item
                processed_item = limit_decimal_places(copy.deepcopy(item), 8)
                processed_history_list.append(processed_item) # Should remain a dict

        # 3. Convert back to tuple
        input_history_tuple = tuple(processed_history_list)

        # --- Process pre_handoff_items and new_items (keep simpler logic) ---
        processed_pre_items_list = []
        if handoff_data.pre_handoff_items:
            for item in handoff_data.pre_handoff_items:
                processed_item = limit_decimal_places(copy.deepcopy(item), 8)
                processed_pre_items_list.append(processed_item)
        pre_handoff_items_tuple = tuple(processed_pre_items_list)

        processed_new_items_list = []
        if handoff_data.new_items:
             for item in handoff_data.new_items:
                processed_item = limit_decimal_places(copy.deepcopy(item), 8)
                processed_new_items_list.append(processed_item)
        new_items_tuple = tuple(processed_new_items_list)
        # --- End pre/new items processing ---

        # *** Debug Print ***
        print("\n--- DEBUG: Force Structure - Processed History (Item 0 Check) ---")
        if input_history_tuple:
            print(f"Type of input_history_tuple[0]: {type(input_history_tuple[0])}")
            import pprint
            pprint.pprint(input_history_tuple[0])
        else:
            print("input_history_tuple is empty.")
        print("--- END DEBUG ---\n")

        # Create new HandoffInputData with guaranteed tuple structure for history
        processed_data = HandoffInputData(
            input_history=input_history_tuple, # Now guaranteed to be a tuple of dicts (or empty)
            pre_handoff_items=pre_handoff_items_tuple,
            new_items=new_items_tuple
        )
        print("Successfully processed handoff data (Forcing History Structure).")
        return processed_data

    except Exception as e:
        print(f"ERROR during handoff data processing (Forcing Structure): {e}")
        import traceback
        traceback.print_exc()
        print("Returning original handoff data due to error.")
        return handoff_data # Fallback

def fix_search_result_scores(data: Any, max_decimal_places: int = 8):
    """Direct fix for search result scores with too many decimal places.
    
    This function directly processes the entire input data structure to find and fix
    any floating point values, especially focusing on search result scores.
    
    Args:
        data: Any data structure (dict, list, tuple, or scalar)
        max_decimal_places: Maximum number of decimal places to keep
        
    Returns:
        The processed data with controlled floating point precision
    """
    # Handle None
    if data is None:
        return None
        
    # Handle primitive types
    if isinstance(data, (int, str, bool)):
        return data
        
    # Handle float values directly
    if isinstance(data, (float, Decimal)):
        # First round the value properly
        rounded_val = round(float(data), max_decimal_places)
        # Then format to exactly max_decimal_places
        return float(f"{rounded_val:.{max_decimal_places}f}")
        
    # Handle dictionaries
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # Special handling for search result scores
            if k == "score" and isinstance(v, (float, Decimal)):
                # Extra careful processing for score fields
                result[k] = float(f"{round(float(v), max_decimal_places):.{max_decimal_places}f}")
            else:
                # Recursive processing for other fields
                result[k] = fix_search_result_scores(v, max_decimal_places)
        return result
        
    # Handle lists
    if isinstance(data, list):
        return [fix_search_result_scores(item, max_decimal_places) for item in data]
        
    # Handle tuples
    if isinstance(data, tuple):
        return tuple(fix_search_result_scores(item, max_decimal_places) for item in data)
    
    # Special handling for HandoffInputData objects
    if str(type(data)) == "<class 'agents.handoffs.HandoffInputData'>":
        print("Processing HandoffInputData object")
        # Process each attribute of HandoffInputData separately
        # We cannot modify the object directly, so we'll create a new one
        
        # Process input_history
        input_history = None
        if hasattr(data, 'input_history') and data.input_history is not None:
            input_history = fix_search_result_scores(data.input_history, max_decimal_places)
            
        # Process pre_handoff_items
        pre_handoff_items = None
        if hasattr(data, 'pre_handoff_items') and data.pre_handoff_items is not None:
            pre_handoff_items = fix_search_result_scores(data.pre_handoff_items, max_decimal_places)
            
        # Process new_items
        new_items = None
        if hasattr(data, 'new_items') and data.new_items is not None:
            new_items = fix_search_result_scores(data.new_items, max_decimal_places)
            
        # Create a new HandoffInputData object with processed attributes
        try:
            return HandoffInputData(
                input_history=input_history if input_history is not None else data.input_history,
                pre_handoff_items=pre_handoff_items if pre_handoff_items is not None else data.pre_handoff_items,
                new_items=new_items if new_items is not None else data.new_items
            )
        except Exception as e:
            print(f"Error creating new HandoffInputData: {e}")
            return data
    
    # For other objects with __dict__ attribute, process their attributes
    if hasattr(data, "__dict__"):
        # We create a shallow copy to avoid modifying the original object
        result = copy.copy(data)
        
        for attr_name, attr_value in data.__dict__.items():
            processed_value = fix_search_result_scores(attr_value, max_decimal_places)
            # Only try to set attributes that are not methods or other non-data attributes
            # Skip attributes of immutable types like str
            if not callable(attr_value) and not isinstance(attr_value, (str, int, bool, float)):
                try:
                    setattr(result, attr_name, processed_value)
                except Exception as e:
                    print(f"Warning: Could not set attribute {attr_name}: {e}")
        return result
    
    # Default case: return the original value for any other type
    return data

def round_search_result_scores(handoff_data: HandoffInputData, max_decimal_places: int = 3) -> HandoffInputData:
    """Process handoff data to ensure all search result scores have at most max_decimal_places decimal places.
    
    Args:
        handoff_data: The handoff data to process
        max_decimal_places: Maximum number of decimal places to keep
        
    Returns:
        The processed handoff data with rounded scores
    """
    try:
        print(f"Processing handoff data type: {type(handoff_data)}")
        print(f"Using max_decimal_places: {max_decimal_places}")
        
        # Implementation that directly targets file search results and scores
        # First, ensure we work with a copy to avoid modifying the original
        data_copy = copy.deepcopy(handoff_data)
        
        # Strictly enforce 8 decimal places maximum (well below OpenAI's 16 limit)
        safe_max_decimals = min(8, max_decimal_places)
        print(f"Using safe_max_decimals: {safe_max_decimals}")
        
        # Special handler for floating point values
        def format_float(value):
            if not isinstance(value, (float, Decimal)):
                return value
            
            # Convert to string with restricted precision
            # Using a safe maximum of 8 decimal places (well below OpenAI's 16 limit)
            safe_decimals = min(8, max_decimal_places)
            
            # Use round() to properly round the value rather than format string truncation
            rounded_value = round(float(value), safe_decimals)
            
            # Format to string with exactly the number of decimals we want
            formatted = f"{{:.{safe_decimals}f}}".format(rounded_value)
            
            # Convert back to float with controlled precision
            return float(formatted)
        
        # Direct aggressive processing of search result scores
        if hasattr(handoff_data, 'input_history') and isinstance(handoff_data.input_history, tuple):
            # Convert to list for easy modification
            input_history_list = list(handoff_data.input_history)
            
            # Go through each item looking for file search results
            for i, item in enumerate(input_history_list):
                if isinstance(item, dict):
                    # Fix all search result scores precisely
                    if 'type' in item and item['type'] in ('file_search_call', 'file_search_results'):
                        if 'results' in item and isinstance(item['results'], list):
                            for result in item['results']:
                                if isinstance(result, dict) and 'score' in result:
                                    # Use multiple techniques to ensure precision is limited
                                    score = float(result['score'])
                                    # Round to 8 places instead of 15
                                    score = round(score, 8)
                                    # Format to string with exactly 8 places then back
                                    score = float(f"{score:.8f}")
                                    # Double check decimal places
                                    str_val = str(score)
                                    if '.' in str_val and len(str_val.split('.')[1]) > 8:
                                        int_part = str_val.split('.')[0]
                                        decimal_part = str_val.split('.')[1][:8]
                                        score = float(f"{int_part}.{decimal_part}")
                                    result['score'] = score
                                    print(f"Aggressive decimal limiting on score: {score}")
        
        # Direct processing of the data attribute
        if hasattr(data_copy, 'data') and data_copy.data is not None:
            data_copy.data = fix_search_result_scores(data_copy.data, safe_max_decimals)
            
        # Direct processing of pre_handoff_items
        if hasattr(data_copy, 'pre_handoff_items') and data_copy.pre_handoff_items is not None:
            data_copy.pre_handoff_items = fix_search_result_scores(data_copy.pre_handoff_items, safe_max_decimals)
            
        # Direct processing of new_items
        if hasattr(data_copy, 'new_items') and data_copy.new_items is not None:
            data_copy.new_items = fix_search_result_scores(data_copy.new_items, safe_max_decimals)
        
        # Legacy processing for compatibility
        if hasattr(data_copy, 'input_history') and isinstance(data_copy.input_history, tuple):
            input_list = list(data_copy.input_history)
            
            # Direct replacement of all file search results with controlled precision
            for i, item in enumerate(input_list):
                # Handle dictionary items in input_history
                if isinstance(item, dict):
                    # Apply recursive processing to the entire dictionary
                    # This ensures we catch all nested values that might be floats
                    process_dict_values(item, format_float)
                    
                    # Additional targeted processing for 'results' which commonly has scores
                    if 'results' in item and isinstance(item['results'], list):
                        for result in item['results']:
                            if isinstance(result, dict):
                                process_dict_values(result, format_float)
                                # Extra direct processing for score field which is known to cause issues
                                if 'score' in result and isinstance(result['score'], (float, Decimal)):
                                    result['score'] = format_float(result['score'])
                elif isinstance(item, list):
                    # Process lists directly
                    process_list_values(item, format_float)
            
            # Create a new HandoffInputData with processed input_history
            try:
                data_copy = HandoffInputData(
                    input_history=tuple(input_list),
                    pre_handoff_items=data_copy.pre_handoff_items,
                    new_items=data_copy.new_items,
                    data=data_copy.data if hasattr(data_copy, 'data') else None
                )
            except Exception as e:
                print(f"Error creating new HandoffInputData: {e}")
        
        # Additional DIRECT string-based processing for any search result scores
        # This is a last-resort approach when the other methods fail
        if hasattr(data_copy, 'input_history') and isinstance(data_copy.input_history, tuple):
            try:
                temp_list = list(data_copy.input_history)
                
                # Process each item in input_history directly through JSON conversion
                for i, item in enumerate(temp_list):
                    if isinstance(item, dict):
                        # Convert to JSON string
                        item_str = json.dumps(item)
                        
                        # Direct regex replacement to limit decimal places in scores
                        # Match "score": followed by a number with 9+ decimal places
                        pattern = r'("score":)(\s*)(\d+\.\d{9,})'
                        
                        # Replace with same score but limited decimal places
                        def replace_score(match):
                            prefix = match.group(1)
                            whitespace = match.group(2)
                            num_str = match.group(3)
                            num = float(num_str)
                            return f'{prefix}{whitespace}{num:.8f}'
                        
                        # Apply the regex replacement
                        fixed_str = re.sub(pattern, replace_score, item_str)
                        
                        # Convert back to dict
                        if fixed_str != item_str:
                            try:
                                temp_list[i] = json.loads(fixed_str)
                                print("Applied direct regex fix to search result scores")
                            except json.JSONDecodeError:
                                # Keep original if JSON parsing fails
                                pass
                
                # Create new HandoffInputData with processed input_history
                try:
                    data_copy = HandoffInputData(
                        input_history=tuple(temp_list),
                        pre_handoff_items=data_copy.pre_handoff_items,
                        new_items=data_copy.new_items,
                        data=data_copy.data if hasattr(data_copy, 'data') else None
                    )
                except Exception as e:
                    print(f"Error in direct string processing: {e}")
            except Exception as e:
                print(f"Error in direct string processing outer block: {e}")
        
        # Process any data attribute that might contain file search results
        if hasattr(data_copy, 'data') and data_copy.data:
            if isinstance(data_copy.data, dict):
                process_dict_values(data_copy.data, format_float)
            elif isinstance(data_copy.data, list):
                process_list_values(data_copy.data, format_float)
            else:
                process_nested_data(data_copy.data, safe_max_decimals)
        
        print("Successfully processed and rounded all scores")
        return data_copy
        
    except Exception as e:
        print(f"Error in score rounding: {e}")
        print("Falling back to emergency decimal places fix")
        
        # Last resort: Direct JSON manipulation to enforce precision
        try:
            # Try to directly manipulate the serialized form to control precision
            data_copy = copy.deepcopy(handoff_data)
            
            # Run emergency direct string replacement on all attributes
            # This is a last-resort measure to fix excessive decimal places
            if hasattr(data_copy, 'input_history') and isinstance(data_copy.input_history, tuple):
                input_list = list(data_copy.input_history)
                
                # EMERGENCY MEASURE: Convert to JSON and back with string replacement
                for i, item in enumerate(input_list):
                    if isinstance(item, dict):
                        try:
                            # Convert to JSON
                            item_json = json.dumps(item)
                            
                            # Use regex to find all floating point numbers with excess precision
                            import re
                            # Find any numbers with more than 6 decimal places (very conservative)
                            excess_precision_pattern = r'(\d+\.\d{6,})'
                            
                            # Replace with truncated versions (5 decimal places)
                            def truncate_match(match):
                                number = float(match.group(0))
                                return f"{number:.5f}"
                            
                            # Apply the regex replacement
                            fixed_json = re.sub(excess_precision_pattern, truncate_match, item_json)
                            
                            # Convert back to dict and replace in the list
                            input_list[i] = json.loads(fixed_json)
                        except Exception as item_err:
                            print(f"Error fixing item {i}: {item_err}")
                
                # Try to create a new HandoffInputData with the fixed input_history
                try:
                    return HandoffInputData(
                        input_history=tuple(input_list),
                        pre_handoff_items=data_copy.pre_handoff_items,
                        new_items=data_copy.new_items,
                        data=data_copy.data if hasattr(data_copy, 'data') else None
                    )
                except Exception as handoff_err:
                    print(f"Error creating new HandoffInputData: {handoff_err}")
            
        except Exception as emergency_e:
            print(f"Emergency fix also failed: {emergency_e}")
        
        # If all else fails, return the original data
        return handoff_data

def process_dict_values(d, value_processor):
    """Recursively process all values in a dictionary.
    
    Args:
        d: The dictionary to process
        value_processor: A function that takes a value and returns a processed value
    """
    if not isinstance(d, dict):
        return
        
    for key, value in list(d.items()):
        if isinstance(value, float) or isinstance(value, Decimal):
            d[key] = value_processor(value)
        elif isinstance(value, dict):
            process_dict_values(value, value_processor)
        elif isinstance(value, list):
            process_list_values(value, value_processor)

def process_list_values(lst, value_processor):
    """Recursively process all values in a list.
    
    Args:
        lst: The list to process
        value_processor: A function that takes a value and returns a processed value
    """
    if not isinstance(lst, list):
        return
        
    for i, value in enumerate(lst):
        if isinstance(value, float) or isinstance(value, Decimal):
            lst[i] = value_processor(value)
        elif isinstance(value, dict):
            process_dict_values(value, value_processor)
        elif isinstance(value, list):
            process_list_values(value, value_processor)

def process_nested_data(data, max_decimal_places):
    """Process nested dictionary or list data to ensure all score values have limited decimal places."""
    if isinstance(data, dict):
        # Process FileSearchTool results
        if 'results' in data and isinstance(data['results'], list):
            for result in data['results']:
                if isinstance(result, dict) and 'score' in result:
                    # Direct string-based precision control
                    result['score'] = float(f"{{:.{max_decimal_places}f}}".format(float(result['score'])))
                    print(f"Fixed result score to {result['score']}")
        
        # Process all keys in the dictionary
        for key, value in list(data.items()):  # Use list() to avoid modification during iteration errors
            if key == 'score' and isinstance(value, (float, decimal.Decimal)):
                # Direct string-based precision control
                data[key] = float(f"{{:.{max_decimal_places}f}}".format(float(value)))
                print(f"Fixed score at key '{key}' to {data[key]}")
            elif isinstance(value, (dict, list)):
                process_nested_data(value, max_decimal_places)
    
    elif isinstance(data, list):
        # Process each item in the list
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                process_nested_data(item, max_decimal_places)
            elif isinstance(item, (float, decimal.Decimal)) and i < len(data):
                # Direct string-based precision control for list items
                data[i] = float(f"{{:.{max_decimal_places}f}}".format(float(item)))
                print(f"Fixed list item at index {i} to {data[i]}") 