#!/usr/bin/env python
import asyncio
from uuid import uuid4
from ai_tutor.context import TutorContext
from ai_tutor.fsm import TutorFSM

"""
Interactive CLI demo for the AI Tutor using the Planner→Executor→FSM loop.
Type user messages and see tutor responses; enter 'exit' to quit.
"""

async def main():
    # Initialize a fresh TutorContext
    ctx = TutorContext(session_id=uuid4(), user_id="demo_user")
    fsm = TutorFSM(ctx)
    print("=== AI Tutor Interactive CLI Demo ===")
    print("Type your message and press enter. Type 'exit' to quit.")
    while True:
        text = input("> ")
        if text.strip().lower() in {"exit", "quit"}:
            print("Session ended.")
            break
        # Send user message through the FSM
        result = await fsm.on_user_message({"event_type": "user_message", "data": {"text": text}})
        # Print the result
        try:
            # If the result is a dict or Pydantic model, print it neatly
            print(result)
        except Exception:
            print(str(result))

if __name__ == "__main__":
    asyncio.run(main()) 