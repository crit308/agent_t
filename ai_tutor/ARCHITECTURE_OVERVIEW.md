# AI Tutor — Refactored Architecture Guide

_Last updated: 2025‑04‑24_

This document explains **how the AI Tutor works _after_ the Planner + Executor refactor**.  Read this if you need to debug the runtime flow, add new skills, or integrate the tutor into additional front‑ends.

---

## 1. Bird's‑Eye View

```
┌──────── User/WebSocket/REST ────────┐
│   (event: user_message/json)        │
└──────────────────────────────────────┘
               │
               ▼
        TutorFSM (finite‑state orchestrator)
   ┌───────────────────────────────────┐
   │  States: idle → planning →        │
   │          executing → awaiting     │
   └───────────────────────────────────┘
               │          △
               ▼          │ (COMPLETED / STUCK)
        PlannerAgent      │
   (decides next objective)│
               │          │ (CONTINUE)
               ▼          ▼
         ExecutorAgent  ──────▶ Skill Registry
              (how)             (~10 atomic tactics)
```

• **PlannerAgent** (LLM GPT‑4o) → _WHAT_ to learn (returns `Objective`).  
• **ExecutorAgent** (loop, mostly cheap models) → _HOW_ to teach until mastery.  
• **Skills** are plain `async def` functions registered via `@tool` decorator.  
• **TutorFSM** drives control flow; no business logic in LLMs.

---

## 2. Directory layout (important folders)

| Path | Purpose |
|------|---------|
| `core/` | Thin wrappers and shared primitives (`llm.py`, `schema.py`, `enums.py`, `memory.py`). |
| `skills/` | Self‑registering tactics – each file houses one skill. |
| `agents/` | `planner_agent.py`, `executor_agent.py`, and offline analyzers. |
| `fsm.py` | Deterministic Python state‑machine orchestrator. |
| `context.py` | `TutorContext` model – serialisable runtime state. |
| `routers/` | FastAPI endpoints that feed events into `TutorFSM`. |

---

## 3. Runtime lifecycle

1. **Session Open** (API route)  
   • `TutorContext` row pulled/created → passed to `TutorFSM` (state =idle).
2. **User sends message** (`user_message` event)  
   • `TutorFSM.on_user_message(event)` invoked.
3. **Planning phase** (if state ∈ {`idle`, `planning`})  
   • `run_planner(ctx)` → returns `PlannerOutput` (list of `Objective`).  
   • First objective stored in `ctx.current_focus_objective`.
4. **Executing phase**  
   • `ExecutorAgent.run(objective, ctx)` chooses a skill via `choose_tactic()` and executes it.  
   • Returns `(status, result)` where `status` ∈ {`CONTINUE`, `COMPLETED`, `STUCK`}.
5. **FSM transition rules**
```
if status == CONTINUE  → state = awaiting_user
if status in {COMPLETED, STUCK} → state = planning
```
6. **Result** (LLM explanation, quiz, etc.) is forwarded back over WebSocket / HTTP.
7. Loop until session ends.

---

## 4. Key Models / Schemas

| Model | File | Notes |
|-------|------|-------|
| `TutorContext` | `context.py` | Persists FSM state, objective, user model, last event, high‑cost budgets. |
| `Objective` | `core/schema.py` | `{topic, learning_goal, target_mastery, priority}` |
| `PlannerOutput` | `core/schema.py` | List of Objectives. |
| `ExecutorStatus` | `core/enums.py` | `CONTINUE`, `COMPLETED`, `STUCK`. |

---

## 5. Writing a new skill

1. Create a file in `skills/`, e.g. `summarise_paper.py`.
2. Implement an `async def` with first parameter `ctx: RunContextWrapper[TutorContext]`.
3. Decorate it:
```python
from ai_tutor.skills import tool

@tool(cost="high")  # optional cost tag
async def summarise_paper(ctx, title:str):
    ...
```
4. Because `skills/__init__.py` auto‑imports every module, your skill registers at import time.
5. Update `choose_tactic()` (or Planner prompt) to use the new skill.

---

## 6. Extending the Planner

The planner prompt lives in `agents/planner_agent.py` (`system_msg` var).  
It already describes `read_knowledge_base` & `dag_query` tools; you can add more by:
1. Importing the meta‑tool at top of file.  
2. Extending the tool description string.  
3. Updating validation if output schema changes.

---

## 7. High‑cost skill budget

`TutorContext` tracks `high_cost_calls` and `max_high_cost_calls` (default = 3).  
`ExecutorAgent` checks for `_skill_cost == 'high'` attributes on skills and blocks calls after the budget is consumed.

---

## 8. Persistence & resume (Phase‑3)

Context serialises to Supabase `sessions` table on every state change.  On websocket reconnect the API layer reloads `TutorContext`, recreates `TutorFSM`, and replays the last event so execution can continue.

---

## 9. Removal timeline for legacy modules

| Phase | Action |
|-------|--------|
| 2 | FSM routing live; legacy Orchestrator unreachable from API. |
| 3 | Skills migrated off `ai_tutor/tools`; delete `tools/`. |
| 4 | Remove `agents/legacy_orchestrator.py`, old tests, deprecated utils. |

---

## 10. FAQ

**Q – Why split Planner vs. Executor?**  
A – Reduces GPT‑4 calls per turn and cleanly separates _curriculum_ (what) from _pedagogy_ (how).

**Q – Where is the user‑model stored?**  
A – `TutorContext.user_model_state` (in‑memory during session) and `supabase.table('user_models')` for long‑term storage.

**Q – How do I test?**  
A – See `tests/` folder: `test_fsm_transitions.py`, `test_planner_output.py`, `test_skill_registry.py`.

---

✦ Happy hacking! ✦ 

![AI Tutor data‑flow](docs/assets/architecture_v2.svg) 