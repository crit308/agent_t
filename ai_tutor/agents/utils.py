from agents import HandoffInputData
import pprint
import json
import decimal
import copy

def round_search_result_scores(data):
    """Round search result scores to avoid issues with decimal precision.
    
    This function is particularly important because the OpenAI APIs have 
    restrictions on the number of decimal places allowed in numeric values.
    
    Args:
        data: Data object that might contain search result scores
        
    Returns:
        The original data with any scores rounded to a safe precision
    """
    if isinstance(data, dict):
        # Handle dict objects
        for key, value in data.items():
            if key == "score" and isinstance(value, (float, int)):
                # Round scores to 10 decimal places (well below OpenAI's 16 decimal limit)
                data[key] = round(value, 10)
            elif isinstance(value, (dict, list)):
                # Recursively process nested objects
                data[key] = round_search_result_scores(value)
    elif isinstance(data, list):
        # Handle list objects
        for i, item in enumerate(data):
            data[i] = round_search_result_scores(item)
    
    return data

def round_search_result_scores(handoff_data: HandoffInputData, max_decimal_places: int = 10) -> HandoffInputData:
    """Process handoff data to ensure all search result scores have at most max_decimal_places decimal places.
    
    Args:
        handoff_data: The handoff data to process
        max_decimal_places: Maximum number of decimal places to keep (defaults to 10, well below OpenAI's 16 decimal limit)
        
    Returns:
        The processed handoff data with rounded scores
    """
    # Create a deep copy to avoid modifying the original during conversion
    try:
        handoff_copy = copy.deepcopy(handoff_data)
        
        # Define a recursive function to directly process the object structure
        def process_object(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "score" and isinstance(value, (float, decimal.Decimal)):
                        # Ensure we don't exceed max_decimal_places
                        obj[key] = round(float(value), max_decimal_places)
                    elif isinstance(value, (dict, list)):
                        process_object(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        process_object(item)
                    elif isinstance(item, (float, decimal.Decimal)) and i < len(obj):
                        obj[i] = round(float(item), max_decimal_places)
        
        # Process the data attribute if it exists
        if hasattr(handoff_copy, 'data') and handoff_copy.data:
            process_object(handoff_copy.data)
        
        # Process input_history if it's a tuple
        if hasattr(handoff_copy, 'input_history') and isinstance(handoff_copy.input_history, tuple):
            # We need to be careful with immutable tuples
            input_list = list(handoff_copy.input_history)
            for item in input_list:
                if isinstance(item, dict):
                    process_object(item)
            # Return a new HandoffInputData object
            return HandoffInputData(
                input_history=tuple(input_list),
                pre_handoff_items=handoff_copy.pre_handoff_items,
                new_items=handoff_copy.new_items,
                data=handoff_copy.data if hasattr(handoff_copy, 'data') else None
            )
        
        # Process pre_handoff_items
        if hasattr(handoff_copy, 'pre_handoff_items'):
            for item in handoff_copy.pre_handoff_items:
                if hasattr(item, '__dict__'):
                    process_object(item.__dict__)
        
        # Process new_items
        if hasattr(handoff_copy, 'new_items'):
            for item in handoff_copy.new_items:
                if hasattr(item, '__dict__'):
                    process_object(item.__dict__)
        
        print("Successfully processed and rounded all scores")
        return handoff_copy
        
    except Exception as e:
        print(f"Error in score rounding: {e}")
        print("Applied score rounding to handoff data")
        # Return the original data if processing fails
        return handoff_data 