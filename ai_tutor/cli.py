#!/usr/bin/env python
import asyncio
from uuid import uuid4
from ai_tutor.context import TutorContext
from ai_tutor.fsm import TutorFSM

"""
Interactive CLI for the AI Tutor using the new Planner→Executor→FSM loop.
Type your messages and see tutor responses; enter 'exit' to quit.
"""

async def main():
    # Create a fresh TutorContext and FSM
    ctx = TutorContext(session_id=uuid4(), user_id="cli_user")
    fsm = TutorFSM(ctx)
    print("=== AI Tutor Interactive CLI ===")
    print("Type your message and press enter. Type 'exit' to quit.")
    while True:
        text = input("> ")
        if text.strip().lower() in {"exit", "quit"}:
            print("Session ended.")
            break
        # Send the input to the FSM
        result = await fsm.on_user_message({"event_type": "user_message", "data": {"text": text}})
        # Print the response
        try:
            print(result)
        except Exception:
            print(str(result))

if __name__ == "__main__":
    asyncio.run(main())
# End of CLI demo 