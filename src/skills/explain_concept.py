from ai_tutor.core.llm import LLMClient
from ai_tutor.context import TutorContext as SessionContext
from loguru import logger

async def explain_concept(context: SessionContext, topic: str, detail_level: str = "beginner") -> str:
    """
    Generates an explanation for a given topic using an LLM.

    Args:
        context: The current session context (potentially used for KB lookup later).
        topic: The concept or topic to explain.
        detail_level: The desired level of detail (e.g., "beginner", "intermediate").

    Returns:
        A string containing the explanation.
    """
    logger.debug(f"[explain_concept] Called with topic='{topic}', detail_level='{detail_level}', session_id='{getattr(context, 'session_id', None)}'")
    llm_client = LLMClient()
    learning_goal = context.learning_goal or "learn about the topic" # Default if not set

    # TODO: Potentially add logic here to fetch relevant snippets from the KB/vector store
    # using context.vector_store_id if available.

    prompt = f"""
    Explain the concept of "{topic}" clearly and concisely.
    The overall learning goal is: "{learning_goal}".
    Tailor the explanation for a "{detail_level}" level.
    Focus on the core ideas and provide simple examples if possible.
    Do not ask follow-up questions, just provide the explanation.
    """

    try:
        response = await llm_client.chat(prompt)
        explanation = response.message.content
        logger.debug(f"[explain_concept] Returning explanation: {explanation}")
        return explanation if explanation else "I couldn't generate an explanation for that topic right now."
    except Exception as e:
        logger.error(f"[explain_concept] Error calling LLM for explanation: {e}")
        return "Sorry, I encountered an error while trying to generate the explanation."

# Example usage (for testing purposes)
# if __name__ == "__main__":
#     import asyncio
#     async def main():
#         # Mock context for testing
#         class MockContext:
#             learning_goal = "Understand the basics of photosynthesis"
#             vector_store_id = None
#             # Add other necessary attributes if needed
#         mock_context = MockContext()
#         explanation = await explain_concept(mock_context, "Photosynthesis", "beginner")
#         print(explanation)
#     asyncio.run(main()) 