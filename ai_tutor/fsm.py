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
        # Persist last event for session resume
        self.ctx.last_event = event

        if self.state in ('idle', 'planning'):
            return await self._plan()
        elif self.state == 'awaiting_user':
            # Resume execution after user input
            return await self._execute()
        else:
            # Fallback to planning
            return await self._plan()

    async def _plan(self):
        """Invoke the planner to pick the next focus objective and start execution."""
        self.state = 'planning'
        self.ctx.state = self.state
        planner_output = await run_planner(self.ctx)
        # Use first objective
        self.ctx.current_focus_objective = planner_output.objectives[0]
        self.state = 'executing'
        self.ctx.state = self.state
        return await self._execute()

    async def _execute(self):
        """Run one execution step for the current objective, handling COMPLETED, STUCK, and CONTINUE statuses."""
        status, result = await ExecutorAgent.run(self.ctx.current_focus_objective, self.ctx)
        if status == ExecutorStatus.COMPLETED:
            # On completion, plan next objective
            self.state = 'planning'
            self.ctx.state = self.state
            return await self._plan()
        elif status == ExecutorStatus.STUCK:
            # On stuck, plan next objective
            self.state = 'planning'
            self.ctx.state = self.state
            return await self._plan()
        elif status == ExecutorStatus.CONTINUE:
            # Continue current objective, await user input
            self.state = 'awaiting_user'
            self.ctx.state = self.state
            return result
        else:
            # Unexpected status, fallback to planning
            self.state = 'planning'
            self.ctx.state = self.state
            return await self._plan() 