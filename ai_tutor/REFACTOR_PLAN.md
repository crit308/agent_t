# AI Tutor Refactor Plan – Planner + Executor Architecture

_Last updated: {{DATE}}_

## 0. Purpose
Migrate the current multi‑agent design (Orchestrator ➜ Planner ➜ Teacher/QuizCreator/QuizTeacher/…) into a leaner, cheaper architecture consisting of:

1. **PlannerAgent** – decides *what* learning objective comes next.
2. **ExecutorAgent** – decides *how* to fulfil that objective using atomic **skills**.
3. **Skill Registry** – self‑registering functions that encapsulate individual teaching tactics.
4. **Python Finite‑State Machine (FSM)** – deterministic orchestrator that glues Planner, Executor, user input and DB writes.

The AnalyzerAgent remains unchanged for offline heavy processing.

---

## 1. Directory / Package Layout Changes

### New/Changed Packages
| Path | Action | Reason |
|------|--------|--------|
| `ai_tutor/core/` | **ADD** | Common primitives: `llm.py`, `memory.py`, `schema.py` |
| `ai_tutor/skills/` | **ADD** | All atomic tool functions live here; decorated for registry |
| `ai_tutor/agents/planner_agent.py` | **KEEP/ADAPT** | Keep but simplify prompt + remove direct tool imports |
| `ai_tutor/agents/executor_agent.py` | **ADD** | New agent handling objective execution via skills |
| `ai_tutor/fsm.py` | **ADD** | Deterministic state‑machine orchestrator |
| `ai_tutor/agents/orchestrator_agent.py` | **DEPRECATE** | Replace by `fsm.py`; keep temporarily as `legacy_orchestrator.py` |
| `ai_tutor/policy.py` | **DELETE** | Logic folded into `fsm.py` + `executor_agent.py` |
| `ai_tutor/utils/` | **MOVE/ADAPT** | Validate what helpers become skills vs. core utilities |

### File Removals / Renames
* `agents/quiz_teacher_agent.py`, `teacher_agent.py`, `quiz_creator_agent.py` … ➜ **retire** once their functionality is implemented as skills.
* Rename `ai_tutor/agents/orchestrator_agent.py` ➜ `ai_tutor/agents/legacy_orchestrator.py` (until removed).

---

## 2. Skill Registry Implementation

1. **Create** `ai_tutor/skills/__init__.py`:
   ```python
   _REGISTRY = {}

   def tool(name: str | None = None):
       def decorator(fn):
           _REGISTRY[name or fn.__name__] = fn
           return fn
       return decorator

   def get_tool(name):
       return _REGISTRY[name]

   def list_tools():
       return list(_REGISTRY)
   ```
2. **Migrate existing tool‑like functions** (e.g. `call_teacher_agent`, `call_quiz_creator_agent`, `update_user_model`, grading helpers) into `ai_tutor/skills/…` modules and decorate with `@tool()`.
3. **Remove** nested `Runner.run()` inside these functions; replace with direct Python logic or *cheap* LLM calls (`gpt‑3.5‑turbo`).
4. **Cost flag (optional)**: allow decorator to accept `cost="high"` to mark skills that call GPT‑4.

---

## 3. PlannerAgent Adjustments

* **Prompt**: emphasise that output must be a single `ObjectiveSpec` JSON + optional stack of objectives.
* **Tools**: only list meta‑tools (`read_knowledge_base`, `dag_query`). No direct teacher/quiz tools.
* **Output Schema** (`ai_tutor/core/schema.py`):
  ```python
  class Objective(BaseModel):
      topic: str
      learning_goal: str
      target_mastery: float
      priority: int = 5

  class PlannerOutput(BaseModel):
      objectives: list[Objective]  # planner may output >1 objectives
  ```
* **Caching**: retain current lru_cache pattern for `get_planner()`.

---

## 4. ExecutorAgent – Design

```
ExecutorAgent.receive(objective: Objective, ctx: TutorContext):
    while True:
        tactic = choose_tactic(ctx, objective)
        result = await skills.get_tool(tactic)(ctx, **params)
        update_ctx(ctx, result)

        if objective_completed(ctx, objective):
            return ExecutorStatus.COMPLETED
        if stuck(ctx):
            return ExecutorStatus.STUCK
```

Key files:
* `ai_tutor/agents/executor_agent.py` – contains agent class + helper functions `choose_tactic`, `objective_completed`, `stuck`.
* `ai_tutor/core/enums.py` – define `ExecutorStatus` enum.

Micro‑policy initial rule‑set:
| Condition | Tactic (skill) |
|-----------|----------------|
| No prior explanation | `explain_concept` |
| ≥3 steps since assessment | `create_quiz` |
| Quiz failed | `remediate_concept` |
| Student asked direct question | `answer_question` |

---

## 5. Finite‑State Machine (`fsm.py`)

State diagram:
```
 idle ──user_msg──▶ planning ──Planner OK──▶ executing
 executing ──Executor completed──▶ awaiting_user
 awaiting_user ──user_msg──▶ executing  # if objective ongoing
 executing ──objective_complete──▶ planning  # pick next objective
 executing ──stuck/error──▶ planning
```

Implementation sketch:
```python
class TutorFSM:
    def __init__(self, ctx: TutorContext):
        self.ctx = ctx
        self.state = ctx.state or "idle"

    async def on_user_message(self, text:str):
        if self.state == "idle":
            await self._plan()
        elif self.state == "awaiting_user":
            await ExecutorAgent.feed_user_input(self.ctx, text)
            await self._execute()  # maybe resumes

    async def _plan(self):
        output = await PlannerAgent.run(self.ctx)
        self.ctx.current_objective = output.objectives[0]
        await self._execute()

    async def _execute(self):
        status = await ExecutorAgent.run(self.ctx.current_objective, self.ctx)
        if status == ExecutorStatus.COMPLETED:
            self.state = "planning"
            await self._plan()
        elif status == ExecutorStatus.STUCK:
            self.state = "planning"
            await self._plan()
        else:
            self.state = "awaiting_user"
```

---

## 6. Routing Layer Updates

* **REST / WebSocket** handlers now instantiate `TutorFSM` instead of calling `run_orchestrator`.
* Persist `ctx.state` and `ctx.current_objective` on each transition.
* Remove `InteractionEvent`/`Action` types if no longer used.

---

## 7. Removal / Deprecation Tasks

1. Delete `ai_tutor/policy.py` and references.
2. Mark old agent files as legacy and remove after migration is stable.
3. Remove `agents/utils.py` functions that only served legacy agents.

---

## 8. Testing Plan

1. **Unit tests**
   * Planner returns valid JSON and at least one objective.
   * Each skill returns correct schema.
   * Executor transitions `COMPLETED` when quiz passed.
   * FSM moves from `planning` ➜ `executing` ➜ `awaiting_user`.
2. **Integration test**
   * Simulate chat with scripted student messages; assert DB rows and number of GPT‑4 calls (< N).
3. **Load test**
   * Run 100 concurrent sessions using cheap stubs.

---

## 9. Incremental Roll‑Out Strategy

1. **Phase 0** – create `skills/` registry; migrate one low‑risk skill (`explain_concept`). Keep old orchestrator.
2. **Phase 1** – add `fsm.py` that still calls old OrchestratorAgent; route API to FSM.
3. **Phase 2** – introduce new ExecutorAgent with the single skill; FSM uses it for that skill path.
4. **Phase 3** – migrate remaining skills; retire Teacher/Quiz agents.
5. **Phase 4** – simplify Planner prompt; remove OrchestratorAgent.
6. **Phase 5** – delete legacy code and `policy.py`; run full regression suite.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Hidden reliance on Runner.run inside tools | CI test fails; grep for `Runner.run(` in skills |
| Cost spike due to skills accidentally calling GPT‑4 | Registry allows `cost` tag; Executor enforces budget |
| State loss on websocket reconnect | Persist `ctx.state` and `current_objective` after every transition |
| Planner prompt drift | Unit test golden output; re‑train with fixed examples |

---

## 11. Owner Matrix (RACI)

| Task | Dev Lead | Reviewer | Status |
|------|----------|----------|--------|
| Skill Registry scaffolding | @dev1 | @lead | ⬜ |
| Planner prompt rewrite | @dev2 | @pedagogy | ⬜ |
| FSM implementation | @dev3 | @lead | ⬜ |
| ExecutorAgent core logic | @dev1 | @dev3 | ⬜ |
| API routing updates | @dev2 | @lead | ⬜ |

_Fill in names once assigned._

---

## 12. Timeline (tentative)

| Week | Milestone |
|------|-----------|
| 1 | Phase 0 complete, CI green |
| 2 | Phase 1–2, MVP session end‑to‑end |
| 3 | Phase 3, 80% of old agents retired |
| 4 | Phase 4, code freeze; pedagogical QA |
| 5 | Phase 5, cut v2.0 release |

---

## 13. Done‑Definition for Migration

* No lingering imports from `ai_tutor.agents.teacher_*`, `quiz_*` in codebase.
* `grep -R "Runner.run("` returns zero matches outside tests.
* `fsm.py` coverage ≥ 90 % lines.
* Average LLM cost per 15‑min session ≤ $0.10.
* All unit + integration tests pass in CI.

---

**End of document** 