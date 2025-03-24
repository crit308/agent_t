from agents import HandoffInputData
import pprint

def round_search_result_scores(handoff_data: HandoffInputData, max_decimal_places: int = 16) -> HandoffInputData:
    """Process handoff data to ensure all search result scores have at most max_decimal_places decimal places.
    
    Args:
        handoff_data: The handoff data to process
        max_decimal_places: Maximum number of decimal places to allow (default: 16)
        
    Returns:
        The processed handoff data (modified in place)
    """
    # Print debug information about the structure
    print("HandoffInputData attributes:", dir(handoff_data))
    
    # The HandoffInputData object doesn't have 'all_items' but likely has other properties
    # Let's check for common properties like 'items', 'input_items', or 'history'
    items = []
    
    # Try to access different potential properties of HandoffInputData
    if hasattr(handoff_data, 'items'):
        items = handoff_data.items
        print("Found 'items' property")
    elif hasattr(handoff_data, 'input_items'):
        items = handoff_data.input_items
        print("Found 'input_items' property")
    elif hasattr(handoff_data, 'history'):
        items = handoff_data.history
        print("Found 'history' property")
    elif hasattr(handoff_data, 'data'):
        items = handoff_data.data
        print("Found 'data' property")
    elif hasattr(handoff_data, 'content'):
        items = handoff_data.content
        print("Found 'content' property")
    elif hasattr(handoff_data, 'messages'):
        items = handoff_data.messages
        print("Found 'messages' property")
    else:
        print("No standard item list properties found")
        
    # Process the items to ensure all decimal values have at most the specified decimal places
    for item in items:
        # Check if this is a file search result item
        if isinstance(item, dict) and item.get("type") == "file_search_call" and "results" in item:
            # Process each result
            for result in item.get("results", []):
                if "score" in result and isinstance(result["score"], float):
                    # Round the score to the specified decimal places to avoid API errors
                    result["score"] = round(result["score"], max_decimal_places)
                    print(f"Rounded score to {result['score']}")
    
    # Access raw dictionary if all else fails
    if not items and hasattr(handoff_data, '__dict__'):
        print("Falling back to __dict__ exploration")
        for key, value in handoff_data.__dict__.items():
            print(f"Found attribute: {key}, type: {type(value)}")
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and item.get("type") == "file_search_call" and "results" in item:
                        for result in item.get("results", []):
                            if "score" in result and isinstance(result["score"], float):
                                result["score"] = round(result["score"], max_decimal_places)
                                print(f"Rounded score in __dict__ to {result['score']}")
    
    # Return the modified handoff data
    return handoff_data 