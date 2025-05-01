from ai_tutor.skills import skill
from ai_tutor.agents.models import QuizQuestion
from typing import List, Dict, Any

@skill
async def draw_mcq_actions(question: QuizQuestion, question_id: str = "q1") -> List[Dict[str, Any]]:
    """
    Generates a list of CanvasObjectSpec dictionaries to draw an MCQ.
    """
    QUESTION_X = 50
    QUESTION_Y = 50
    QUESTION_WIDTH = 700
    OPTION_START_Y = 100
    OPTION_SPACING = 40
    OPTION_X_OFFSET = 20
    OPTION_RADIO_RADIUS = 8
    OPTION_TEXT_X_OFFSET = 25

    actions: List[Dict[str, Any]] = []
    # Question Text
    actions.append({
        "id": f"mcq-{question_id}-text",
        "kind": "text",
        "x": QUESTION_X,
        "y": QUESTION_Y,
        "text": question.question,
        "fontSize": 18,
        "fill": "#000000",
        "width": QUESTION_WIDTH,
        "metadata": {
            "source": "assistant",
            "role": "question",
            "question_id": question_id
        }
    })
    current_y = OPTION_START_Y
    for i, option_text in enumerate(question.options):
        option_id = i
        actions.append({
            "id": f"mcq-{question_id}-opt-{option_id}-radio",
            "kind": "circle",
            "x": QUESTION_X + OPTION_X_OFFSET,
            "y": current_y + OPTION_RADIO_RADIUS,
            "radius": OPTION_RADIO_RADIUS,
            "stroke": "#555555",
            "strokeWidth": 1,
            "fill": "#FFFFFF",
            "metadata": {
                "source": "assistant",
                "role": "option_selector",
                "question_id": question_id,
                "option_id": option_id
            }
        })
        actions.append({
            "id": f"mcq-{question_id}-opt-{option_id}-text",
            "kind": "text",
            "x": QUESTION_X + OPTION_X_OFFSET + OPTION_TEXT_X_OFFSET,
            "y": current_y + OPTION_RADIO_RADIUS,
            "text": f"{chr(65+i)}. {option_text}",
            "fontSize": 16,
            "fill": "#333333",
            "metadata": {
                "source": "assistant",
                "role": "option_label",
                "question_id": question_id,
                "option_id": option_id
            }
        })
        current_y += OPTION_SPACING
    return actions 