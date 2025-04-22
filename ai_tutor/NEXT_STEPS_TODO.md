# AI Tutor Refactor – Remaining Tasks (Phase‑2 → GA)

_Last updated: {{DATE}}_

This document lists every outstanding change required to fully finish the Planner + Executor refactor.  Items are grouped by severity and dependency order.

---

## 0 ▪ Critical / Blockers (must‑fix before merging to `main`)

| ID | Area | Work | Owner |
|----|------|------|-------|
| C‑01 | `core/enums.py` | Add `CONTINUE` member to `ExecutorStatus` (for non‑terminal executor steps). |  |
| C‑02 | `agents/executor_agent.py` | Return `(ExecutorStatus.CONTINUE, result)` when objective not yet mastered; keep `COMPLETED` only on mastery; `STUCK` on fallback. |  |
| C‑03 | `fsm.py` | Handle `CONTINUE` → stay in `awaiting_user` (no Planner call); update transition table. |  |
| C‑04 | API / WebSocket routing | Wire `TutorFSM` into FastAPI routes / `manager.py` so all user messages flow through the FSM (old orchestrator path removed). |  |
| C‑05 | Legacy `ai_tutor/tools` indirection | For Phase‑2, keep; **Phase‑3:** migrate logic in `call_teacher_agent`, `call_quiz_creator_agent`, `call_quiz_teacher_evaluate` directly into skills and delete these wrappers. |  |

---

## 1 ▪ High Priority (before public beta)

| ID | Area | Work | Notes |
|----|------|------|-------|
| H‑01 | `skills/reflect_on_interaction.py` | Remove double hop: implement logic directly; stop importing from legacy `tools`. |  |
| H‑02 | Tests | Update / create unit tests for: PlannerOutput validity, Executor CONTINUE flow, FSM transitions. | Use pytest‑asyncio. |
| H‑03 | Examples / demos | Point demos to new FSM; remove calls to removed agents (`orchestrator_agent`, `teacher_agent`, etc.). |  |
| H‑04 | Lint / type checks | Run `ruff --fix`, `mypy` and update CI config. |  |
| H‑05 | `README.md` | Update architecture diagram + quick‑start instructions. |  |

---

## 2 ▪ Medium Priority (Phase‑3 & Phase‑4)

| ID | Area | Work | Notes |
|----|------|------|-------|
| M‑01 | Skill Cost Budget | Extend `@tool` decorator to accept `cost="high"` flag; Executor enforces max budget per session. |  |
| M‑02 | Objective Completion Heuristics | Replace placeholder `objective_completed()` with rule: _quiz pass ≥ target_mastery_ **and** `update_user_model` mastery probability > threshold. |  |
| M‑03 | Persistence | Store `current_focus_objective` & `state` into DB on every FSM transition (Supabase `sessions` table). |  |
| M‑04 | Resume Logic | On websocket reconnect, load context, feed last event into FSM to resume mid‑objective. |  |
| M‑05 | Skill Docs | Add docstrings + schema for every skill (for auto‑generated Planner tool descriptions). |  |
| M‑06 | Remove Legacy Modules | Delete `ai_tutor/tools/__init__.py` & sub‑modules after skills migration; migrate any remaining utilities to `utils/` or `core/`. |  |

---

## 3 ▪ Low Priority / Nice‑to‑Have

| ID | Area | Work |
|----|------|------|
| L‑01 | Data‑driven micro‑policy | Replace rule‑based `choose_tactic()` with ML‑based policy (Contextual Bandit). |
| L‑02 | Skill Autodiscovery | Load Python modules under `skills/` dynamically using `importlib` to avoid manual import list. |
| L‑03 | Telemetry | Add Prometheus counters: `planner_calls_total`, `executor_steps_total`, `skills_latency_seconds`. |
| L‑04 | Cost Tracking | Track total LLM tokens per session; surface on UI. |

---

## 4 ▪ Clean‑up / Technical Debt

1. Search for `Runner.run(` references **inside `ai_tutor/`** once skills migration is complete – should be zero.  
2. Remove `agents/utils.py` once all helpers are absorbed elsewhere.  
3. Delete `agents/legacy_orchestrator.py` after Phase‑2 launch.

---

## 5 ▪ Timeline Revision (new)

| Week | Milestone |
|------|-----------|
| 0 (now) | Complete Critical list (C‑01 → C‑05) |
| 1 | High‑priority items, updated tests pass in CI |
| 2 | Medium list finished, legacy modules purged, beta tag |
| 3 | Optional low‑priority polish, public release |

---

✦ _End of document_ ✦ 