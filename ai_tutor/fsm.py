"""
Finite-State Machine orchestrator that wraps the legacy OrchestratorAgent for Phase 1.
"""

from ai_tutor.agents.planner_agent import run_planner
from ai_tutor.agents.executor_agent import ExecutorAgent
from ai_tutor.core.enums import ExecutorStatus
from ai_tutor.context import TutorContext

class TutorFSM:
    """Finite-state orchestrator using Planner and Executor agents."""
    def __init__(self, ctx: TutorContext):
        self.ctx = ctx
        # FSM states: 'idle', 'planning', 'executing', 'awaiting_user'
        self.state = ctx.state or 'idle'
        # Persist initial state back into context
        self.ctx.state = self.state

    async def on_user_message(self, event: dict):
        """
        Entry point for user messages. Routes through planning or execution based on FSM state.
        """
        # Normalize event type: support both 'type' and 'event_type'
        evt_type = event.get('type') or event.get('event_type') or 'unknown'
        # Persist the raw event
        self.ctx.last_event = event
        print(f"[FSM] Received event: {evt_type}, Current State: {self.state}")

        # Determine the action based on the current state
        if self.state == 'idle':
            # Start by planning
            action_result = await self._plan()
        elif self.state == 'planning':
             # If somehow called while planning, just re-plan (or handle error)
             print("[FSM] Warning: Called on_user_message while already planning. Re-planning.")
             action_result = await self._plan()
        elif self.state == 'executing':
             # If ready to execute, run the executor
             print("[FSM] State is 'executing', calling _execute().")
             action_result = await self._execute()
        elif self.state == 'awaiting_user':
             # If awaiting user, assume the event is the input and execute
             print("[FSM] State is 'awaiting_user', calling _execute().")
             action_result = await self._execute()
        else:
            # Fallback for unknown state: plan
            print(f"[FSM] Unknown state '{self.state}', defaulting to _plan().")
            action_result = await self._plan()

        # Return the result from the executed action (_plan or _execute)
        print(f"[FSM] Action Result Type: {type(action_result)}, State AFTER action: {self.state}")
        return action_result

    async def _plan(self):
        """Invoke the planner to pick the next focus objective and update state."""
        self.state = 'planning'
        self.ctx.state = self.state
        print("[FSM] State -> planning")

        try:
            planner_output = await run_planner(self.ctx)
        except Exception as e:
            self.state = 'idle' # Revert state on error
            self.ctx.state = self.state
            print(f"[FSM] Planner Error: {e}")
            # Ensure error is serializable
            return {"response_type": "error", "message": f"Planner failed: {str(e)}"}

        if not planner_output or not planner_output.objectives:
            self.state = 'idle' # Revert state if no objectives
            self.ctx.state = self.state
            print("[FSM] Planner produced no objectives.")
            # Maybe send a completion message? Use MessageResponse format if possible
            return {"response_type": "message", "text": "Looks like we've covered all the planned topics!"}

        # Successfully planned
        self.ctx.current_focus_objective = planner_output.objectives[0]
        self.state = 'executing' # Set state for the *next* step
        self.ctx.state = self.state
        # Use .topic instead of .title for logging
        print(f"[FSM] State -> executing (Objective Topic: {self.ctx.current_focus_objective.topic})")
        # --- Execute immediately after planning ---
        # This maintains the original flow where planning leads directly to the first execution step,
        # but the recursion is broken because _execute will now return instead of calling _plan back.
        return await self._execute()

    async def _execute(self):
        """Run one execution step, update state, and return the result."""
        # Check if there is a current objective before executing
        if not self.ctx.current_focus_objective:
             print("[FSM] Error: _execute called without a current_focus_objective. Re-planning.")
             self.state = 'planning'
             self.ctx.state = self.state
             # Maybe return an error or trigger _plan differently? For now, return error message.
             return {"response_type": "error", "message": "Tutor state error: Missing current objective. Trying to recover."}

        # Use .topic instead of .title for logging
        print(f"[FSM] Executing objective topic: {self.ctx.current_focus_objective.topic}")
        # Ensure we are in 'executing' state before running (or 'awaiting_user' which leads here)
        # No longer strictly necessary with the new on_user_message logic, but good for sanity check
        if self.state not in ['executing', 'awaiting_user']:
             print(f"[FSM] Warning: _execute called while not in 'executing' or 'awaiting_user' state ({self.state}). Proceeding...")

        # Set state to executing before running the agent
        self.state = 'executing'
        self.ctx.state = self.state

        status, result = await ExecutorAgent.run(self.ctx.current_focus_objective, self.ctx)
        print(f"[FSM] Executor Result: Status={status}, Result Type={type(result)}") # Add logging

        # --- State transition based on executor status ---
        if status == ExecutorStatus.COMPLETED:
            print("[FSM] Objective COMPLETED.") # Add logging
            self.state = 'planning' # Ready to plan the *next* objective on the *next* call
            self.ctx.state = self.state
            # Clear the completed objective
            completed_objective_topic = self.ctx.current_focus_objective.topic
            self.ctx.current_focus_objective = None
            # Return a message indicating completion, frontend might trigger next step or user action will trigger _plan
            # Use a standard response model structure if available (like MessageResponse)
            return {"response_type": "message", "text": f"Finished objective: {completed_objective_topic}. Ready for the next topic."}

        elif status == ExecutorStatus.STUCK:
            print("[FSM] Objective STUCK.") # Add logging
            self.state = 'planning' # Ready to plan the *next* objective on the *next* call
            self.ctx.state = self.state
            # Clear the stuck objective
            stuck_objective_topic = self.ctx.current_focus_objective.topic
            self.ctx.current_focus_objective = None
            # Return a message indicating stuck
            return {"response_type": "message", "text": f"Got stuck on objective: {stuck_objective_topic}. Let's try planning again."}

        elif status == ExecutorStatus.CONTINUE:
            print("[FSM] Objective CONTINUE. Awaiting user.") # Add logging
            self.state = 'awaiting_user' # Wait for next user message to trigger execution again
            self.ctx.state = self.state
            # Result should be the actual data (e.g., question, explanation) from the executor agent
            # Ensure 'result' is serializable and fits one of the InteractionResponseData.data types
            # If result is already a Pydantic model like ExplanationResponse, QuestionResponse, etc., it's fine.
            # If it's a plain dict, ensure it matches one of those structures or is handled correctly.
            return result

        else:
            print(f"[FSM] Unexpected Executor Status: {status}. Defaulting to re-plan.") # Add logging
            self.state = 'planning' # Default to planning again
            self.ctx.state = self.state
            self.ctx.current_focus_objective = None # Clear objective
            # Use ErrorResponse structure
            return {"response_type": "error", "message": f"Unexpected execution status: {status}"} 