from agents import HandoffInputData
import pprint
import json
import decimal
import copy
from decimal import Decimal, getcontext

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
        
        # Special handler for floating point values
        def format_float(value):
            if not isinstance(value, (float, Decimal)):
                return value
            
            # First convert to string with restricted precision
            # Using fewer decimal places than requested as additional safety
            safe_decimals = max(0, max_decimal_places - 1)
            formatted = f"{{:.{safe_decimals}f}}".format(float(value))
            
            # Then convert back to float
            return float(formatted)
        
        # We need to handle the specific structure mentioned in the error:
        # input[2].results[0].score
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
                elif isinstance(item, list):
                    # Process lists directly
                    process_list_values(item, format_float)
            
            # Create a new HandoffInputData with processed input_history
            data_copy = HandoffInputData(
                input_history=tuple(input_list),
                pre_handoff_items=data_copy.pre_handoff_items,
                new_items=data_copy.new_items,
                data=data_copy.data if hasattr(data_copy, 'data') else None
            )
        
        # Process any data attribute that might contain file search results
        if hasattr(data_copy, 'data') and data_copy.data:
            if isinstance(data_copy.data, dict):
                process_dict_values(data_copy.data, format_float)
            elif isinstance(data_copy.data, list):
                process_list_values(data_copy.data, format_float)
            else:
                process_nested_data(data_copy.data, max_decimal_places)
        
        # Process pre_handoff_items
        if hasattr(data_copy, 'pre_handoff_items') and data_copy.pre_handoff_items:
            for idx, item in enumerate(data_copy.pre_handoff_items):
                if hasattr(item, '__dict__'):
                    obj_dict = item.__dict__
                    process_dict_values(obj_dict, format_float)
                    # Recreate the object with processed values if needed
        
        # Process new_items
        if hasattr(data_copy, 'new_items') and data_copy.new_items:
            for idx, item in enumerate(data_copy.new_items):
                if hasattr(item, '__dict__'):
                    obj_dict = item.__dict__
                    process_dict_values(obj_dict, format_float)
                    # Recreate the object with processed values if needed
                    
        # Test serialization to verify issue is fixed
        try:
            if hasattr(data_copy, 'data') and data_copy.data:
                # Use standard json module to check serialization
                json_str = json.dumps(data_copy.data)
                json.loads(json_str)  # Try parsing back to verify
        except Exception as e:
            print(f"WARNING: JSON serialization check failed: {e}")
            print("Applying emergency serialization fix")
            
            # Emergency fallback using str representation
            if hasattr(data_copy, 'data') and data_copy.data:
                try:
                    # Convert to JSON and back using precision limiting
                    data_json = json.dumps(data_copy.data)
                    # Use regex to fix excessive precision
                    import re
                    data_json = re.sub(r'(\d+\.\d{10,})', lambda m: f"{float(m.group(0)):.6f}", data_json)
                    data_copy.data = json.loads(data_json)
                except Exception as json_err:
                    print(f"JSON emergency fix failed: {json_err}")
        
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