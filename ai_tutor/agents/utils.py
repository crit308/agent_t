from agents import HandoffInputData
import pprint
import json

def round_search_result_scores(handoff_data: HandoffInputData, max_decimal_places: int = 16) -> HandoffInputData:
    """Process handoff data to ensure all search result scores have at most max_decimal_places decimal places.
    
    Args:
        handoff_data: The handoff data to process
        max_decimal_places: Maximum number of decimal places to allow (default: 16)
        
    Returns:
        The processed handoff data (modified in place)
    """
    # Process the input data by converting to JSON and back
    # This allows us to access all nested structures
    try:
        # Convert to dict first
        if hasattr(handoff_data, '__dict__'):
            data_dict = handoff_data.__dict__
        else:
            data_dict = dict(handoff_data)
            
        # Convert to JSON string
        json_str = json.dumps(data_dict)
        
        # Define a recursive function to find and round all float values
        def process_json(json_obj):
            if isinstance(json_obj, dict):
                for key, value in json_obj.items():
                    if key == "score" and isinstance(value, float):
                        json_obj[key] = round(value, max_decimal_places)
                    elif isinstance(value, (dict, list)):
                        process_json(value)
            elif isinstance(json_obj, list):
                for i, item in enumerate(json_obj):
                    if isinstance(item, (dict, list)):
                        process_json(item)
        
        # Parse JSON back to Python objects
        parsed_data = json.loads(json_str)
        
        # Process all nested structures to round scores
        process_json(parsed_data)
        
        # Update the original object with processed values
        for key, value in parsed_data.items():
            if hasattr(handoff_data, key):
                setattr(handoff_data, key, value)
        
        print("Successfully processed and rounded all scores")
        
    except Exception as e:
        print(f"Error processing scores: {e}")
        
        # Fallback method if the JSON approach fails
        try:
            # Try to round scores in the raw dictionary
            for key, value in handoff_data.__dict__.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "results" in item:
                            for result in item["results"]:
                                if "score" in result and isinstance(result["score"], float):
                                    result["score"] = round(result["score"], max_decimal_places)
        except Exception as fallback_error:
            print(f"Fallback processing also failed: {fallback_error}")
    
    # Return the modified handoff data
    return handoff_data 