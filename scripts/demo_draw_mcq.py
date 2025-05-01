import json
from typing import List, Dict, Any
from ai_tutor.agents.models import QuizQuestion # Assuming this is the correct path

# Define a sample QuizQuestion
sample_mcq = QuizQuestion(
    question="What is the primary mechanism by which water returns to the atmosphere in the water cycle?",
    options=[
        "Condensation",
        "Evaporation",
        "Precipitation",
        "Transpiration"
    ],
    correct_index=1,
    topic="Water Cycle"
)

# Define layout constants (adjust as needed)
QUESTION_X = 50
QUESTION_Y = 50
QUESTION_WIDTH = 700
OPTION_START_Y = 100
OPTION_SPACING = 40
OPTION_X_OFFSET = 20
OPTION_RADIO_RADIUS = 8
OPTION_TEXT_X_OFFSET = 25 # Offset from radio center

def generate_mcq_actions(question: QuizQuestion, question_id: str = "q1") -> List[Dict[str, Any]]:
    """
    Generates a list of CanvasObjectSpec dictionaries to draw an MCQ.
    """
    actions: List[Dict[str, Any]] = []
    
    # 1. Question Text
    actions.append({
        "id": f"mcq-{question_id}-text",
        "kind": "text",
        "x": QUESTION_X,
        "y": QUESTION_Y,
        "text": question.question,
        "fontSize": 18,
        "fill": "#000000",
        "width": QUESTION_WIDTH, # Optional: for potential wrapping
        "metadata": {
            "source": "assistant",
            "role": "question",
            "question_id": question_id
        }
    })

    # 2. Options (Radio button + Text)
    current_y = OPTION_START_Y
    for i, option_text in enumerate(question.options):
        option_id = i # Use index as option ID

        # Radio button (using a circle)
        actions.append({
            "id": f"mcq-{question_id}-opt-{option_id}-radio",
            "kind": "circle",
            "x": QUESTION_X + OPTION_X_OFFSET, # Center X
            "y": current_y + OPTION_RADIO_RADIUS, # Center Y
            "radius": OPTION_RADIO_RADIUS,
            "stroke": "#555555",
            "strokeWidth": 1,
            "fill": "#FFFFFF", # Unselected appearance
            "metadata": {
                "source": "assistant",
                "role": "option_selector", # Specific role for the interactive part
                "question_id": question_id,
                "option_id": option_id
            }
        })
        
        # Option Text Label
        actions.append({
            "id": f"mcq-{question_id}-opt-{option_id}-text",
            "kind": "text",
            "x": QUESTION_X + OPTION_X_OFFSET + OPTION_TEXT_X_OFFSET,
            "y": current_y + OPTION_RADIO_RADIUS, # Align text baseline near circle center
            "text": f"{chr(65+i)}. {option_text}", # Add A., B., etc.
            "fontSize": 16,
            "fill": "#333333",
            "metadata": {
                "source": "assistant",
                "role": "option_label", # Specific role for the text part
                "question_id": question_id,
                "option_id": option_id
            }
        })
        
        current_y += OPTION_SPACING
        
    return actions

if __name__ == "__main__":
    mcq_drawing_actions = generate_mcq_actions(sample_mcq)
    print(json.dumps(mcq_drawing_actions, indent=2)) 