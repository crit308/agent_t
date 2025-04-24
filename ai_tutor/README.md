[![Docs](https://img.shields.io/badge/docs-online-blue)](https://ai‑tutor.readthedocs.io)

Below is a "white‑box walkthrough" of the **AI Tutor** as it exists after the latest refactor.  
If you read this top‑to‑bottom you will know:

* the full user → UI → backend → OpenAI → UI round‑trip,
* where every important class, file, table and WebSocket message lives,
* how adaptation, persistence and observability are achieved, and  
* how to extend or debug any piece without opening the source tree.

---

## 1 High‑level picture

```
┌───────────────┐   HTTP/WS    ┌─────────────────┐   tool calls   ┌──────────────┐
│ React + Next  │◀────────────▶│  FastAPI router  │◀──────────────▶│ Orchestrator │
│   (browser)   │   JSON &     │  /interact & WS  │   python fn    │ (Runner)     │
└───────────────┘   token deltas└─────────────────┘                └─────▲───────┘
      ▲                                │                               tools
      │ Supabase RPC / Storage         │ ctx.persist()                 ▼
┌───────────┐                         ┌─────────────────┐   OpenAI     ┌───────────┐
│ Edge logs │                         │ FunctionTools   │──────────────▶  GPT‑4o   │
└───────────┘                         │ (teacher, quiz) │  HTTPS JSON  └───────────┘
```

---

## 2 Frontend flow (Next 13 app router)

1. **AuthProvider** (Context) boots, fetches Supabase session once; memoised to stop re‑render loops.  
2. **Home page** lets the learner pick / create a *folder* (content bucket).  
3. `<LearnPage>` uploads materials, starts `/sessions` (POST) and then calls `/plan`.  
4. `useTutorStream(sessionId)` opens **one** WebSocket:  
   `ws://<api>/ws/session/{session_id}?token=<jwt>`  
   * Receives `run_item_stream_event` (token deltas, quiz JSON, feedback) and pushes them into a Zustand `sessionStore`.  
   * Listens for `mastery_update` and `telemetry` to update the side bar & analytics widget.  
5. Components:  
   * **TutorChat** renders streaming Markdown, multiple‑choice UI, feedback bubbles.  
   * **MasteryBar** shows concept‑level progress (colour‑coded pill).  
   * **PaceSlider** writes `learning_pace_factor` via `{event_type:"pace_change", value}` WebSocket message.  
   * "I'm stuck" button sends `{event_type:"help_request", mode:"stuck"}`.

No other components talk directly to the backend; all tutor traffic is WS.

---

## 3 Backend layers

### 3.1 FastAPI routers (`ai_tutor/routers`)

| Endpoint | Purpose | Important lines |
|----------|---------|-----------------|
|`POST /folders`| CRUD for content buckets | 307 → `/folders/` only because of trailing slash; fixed now |
|`POST /sessions`| create tutor session and Supabase row | calls `SessionService.create_session` |
|`POST /sessions/{id}/documents`| upload files, push to OpenAI Vector Store | uses `file‑*.txt` IDs |
|`POST /sessions/{id}/plan`| *Cold start* – single‑turn Runner that asks Planner to set `current_focus_objective`; stores it in DB | |
|`POST /sessions/{id}/interact`| HTTP fallback for older clients (rare now) | just calls `run_orchestrator` once |
|`WS /ws/session/{id}`| **Main transport** | creates / hydrates `TutorContext`, streams `Runner.run_streamed` events back |

### 3.2 `TutorContext` (`ai_tutor/context.py`)

```text
session_id, user_id, vector_store_id
current_focus_objective: FocusObjective
user_model_state:
    - concepts: {topic → UserConceptMastery(alpha,beta)}
    - pending_interaction_type / details
    - learning_pace_factor (0.5–1.5)
    - current_topic_segment_index
```

`context_json` is persisted in the `sessions` table after **every** successful tool call so a browser refresh can resume.

### 3.3 Policy → Orchestrator loop

* `policy.choose_action(ctx, last_event) → Action`  
  pure Python, no LLM.
* `orchestrator_agent.run_orchestrator(ctx, event)`:

```text
while True:
    action = choose_action(...)
    dispatch:
        explain   → call_teacher_agent(...)
        evaluate  → call_quiz_teacher_evaluate(...)
        ask_mcq   → call_quiz_creator_agent(...); set pending_interaction_type; return
        advance   → update_user_model(...); call_planner_agent(...) ; continue
```

The loop streams partial outputs; learner sees explanation tokens, *then* a quiz without another HTTP hit.

### 3.4 FunctionTool barrel (`ai_tutor/tools/__init__.py`)

*All* exported tools live here, decorated:

```python
@_export @function_tool(spec=SPEC) @log_tool async def call_teacher_agent(...)

log_tool → inserts latency & token counts into `edge_logs`.
```

No other modules import each other, so circular‑import risk is gone.

---

## 4 Adaptive mechanics

| Signal | Where updated | How used |
|--------|---------------|----------|
|**mastery (alpha, beta)**|`update_user_model` | `policy.bloom_difficulty` picks `easy/medium/hard` |
|**confidence (α+β)**|same | Planner excludes concepts mastered > 0.8 with confidence ≥ 5 |
|**pace slider**|WS `pace_change` → `learning_pace_factor` | scales mastery before difficulty mapping |
|**help_request: "stuck"**|WS event sets pending flag | Policy forces `Explain` with `style="analogy"` |

---

## 5 Performance & resilience

* **Agent factory** cached with `@lru_cache(1)` so model‑load happens once per worker.  
* OpenAI client is reused singleton.  
* WS handler pipes `Runner.run_streamed` token chunks directly – median server think‑time now 550 ms; total round‑trip ≈ 1.3 s.  
* On every tool completion:  
  * telemetry row inserted (`edge_logs`),  
  * new `context_json` written to `sessions`,  
  * `mastery_update` WS message emitted.  
  If the pod dies, a new one reads `context_json` and continues.

---

## 6 Analytics

*Supabase → Metabase* nightly dashboard:

```
select tool,
       p50(latency_ms) as med_ms,
       sum(prompt_tokens+completion_tokens) as tokens
from edge_logs
where created_at > now() - interval '7 days'
group by tool;
```

Product team filters by `session_id` and sees per‑concept mastery curves.

---

## 7 Testing scaffold

1. **Unit** – `tests/test_orchestrator_dispatch.py` mocks tools, asserts pending state & call counts.  
2. **Schema** – `test_tools_validate.py` runs `build_openai_schema` to ensure all FunctionTools include `"additionalProperties": false`.  
3. **Contract** – Playwright e2e script navigates, uploads a file, ensures the first MCQ appears in ≤ 5 s and mastery bar ticks after answering.

---

## 8 Extending the tutor

* Add a new pedagogy action (e.g. `reteach_with_visual`)  
  1. Decorate tool in `ai_tutor/tools/__init__.py` with @_export.  
  2. Add branch in `policy.choose_action`.  
  3. Implement React component if the output type is new.

* Add voice: swap WS handler for `VoicePipeline` – the rest stays unchanged.

* Add new storage (e.g. Redis cache for Planner): Planner tool already a FunctionTool; just import Redis there.

---

### TL;DR

* **Frontend**: one WS, Zustand store, components render deltas.  
* **Backend**: FastAPI → long‑lived `run_orchestrator` loop with Python policy; FunctionTools barrel; context persisted each call.  
* **Adaptivity**: Bayesian mastery, Bloom difficulty, pace slider, help‑request override.  
* **Ops**: lru‑cached agent, streamed tokens, telemetry rows, session resume.

With this mental model you can trace any bug or add any feature without opening the IDE first.

---

# AI Tutor

AI Tutor is a multi-agent intelligent tutoring system built on top of the OpenAI Agents SDK. It uses a PlannerAgent to decide *what* to teach next, an ExecutorAgent to determine *how* to teach each objective by invoking atomic skills, and a deterministic Finite-State Machine (TutorFSM) to orchestrate planning, execution, and user interactions.

## Architecture Overview

- **PlannerAgent**: Generates a sequence of learning objectives based on user context and knowledge base.
- **ExecutorAgent**: Chooses teaching tactics (skills) to achieve a given objective (e.g., explain concept, create quiz, remediate errors).
- **Skill Registry**: Self-registering, decorated Python functions (`@tool()`) for each atomic teaching tactic.
- **TutorFSM**: Finite-State Machine that transitions between `planning`, `executing`, and `awaiting_user`, persisting state (`ctx.state` and `current_focus_objective`) across messages.

![AI Tutor Architecture](docs/architecture.png)

## Quick Start

1.  Clone this repository and create a virtual environment:

    ```bash
    git clone <repo-url>
    cd agent_t/agent_t
    python -m venv env
    # Linux/Mac
    source env/bin/activate
    # Windows
    .\env\Scripts\activate
    pip install -r requirements.txt
    ```

2.  Run the interactive CLI demo:

    ```bash
    python ai_tutor/cli.py
    ```

    Once launched, type your questions or messages at the prompt. Enter `exit` or `quit` to end the session.

3.  Integrate into your FastAPI app:

    - Use `TutorFSM` in your REST `/interact` endpoint or WebSocket handler (see `ai_tutor/routers` for examples).
    - Persist `TutorContext` after each `on_user_message` call to maintain session state.

## Running Tests & Linting

Make sure you have your virtual environment activated, then:

```bash
make sync        # Installs all dependencies and dev tools
make lint        # Runs ruff lint checks
make mypy        # Runs static type checks
make tests       # Runs pytest for all unit/integration tests
make coverage    # Runs tests with coverage report
```

All new Phase 1 tests are located under `tests/test_phase1_high_priority.py` and cover:

- PlannerOutput schema validation
- ExecutorAgent CONTINUE vs COMPLETED workflows
- FSM transitions into `awaiting_user` on intermediate steps

## Next Steps

- **Phase 2**: Migrate legacy tools off `ai_tutor/tools`, enforce skill budgets, improve objective completion heuristics.  
- **Phase 3+**: Add advanced adaptive skills, data-driven policies, and telemetry.

Happy tutoring!  

## Overview

The AI Tutor system uses a multi-agent approach to create personalized lessons:

1. **Document Processing**: Upload documents you want to learn about. The system creates a vector store using OpenAI's embeddings.
2. **Lesson Planning**: A planner agent analyzes the documents and creates a structured lesson plan.
3. **Lesson Creation**: A teacher agent takes the lesson plan and creates comprehensive lesson content.
4. **Quiz Creation**: A quiz creator agent generates a quiz based on the lesson content.
5. **Quiz Assessment**: A quiz teacher agent evaluates user answers and provides personalized feedback.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-tutor.git
cd ai-tutor

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

You can use the AI Tutor from the command line:

```bash
# Run the AI Tutor with one or more files
python -m ai_tutor.main file1.pdf file2.pdf

# Save the lesson content to a file
python -m ai_tutor.main file1.pdf --output lesson.json

# Specify your own API key
python -m ai_tutor.main file1.pdf --api-key your-api-key
```

### Python API

You can also use the AI Tutor programmatically:

```python
import asyncio
from ai_tutor.manager import AITutorManager

async def create_lesson(file_paths):
    # Initialize the AI Tutor manager
    manager = AITutorManager(api_key="your-api-key")
    
    # Upload documents
    await manager.upload_documents(file_paths)
    
    # Generate lesson plan
    lesson_plan = await manager.generate_lesson_plan()
    print(f"Generated lesson plan: {lesson_plan.title}")
    
    # Generate lesson content
    lesson_content = await manager.generate_lesson_content()
    print(f"Generated lesson: {lesson_content.title}")
    
    return lesson_content

# Run the function
lesson = asyncio.run(create_lesson(["file1.pdf", "file2.pdf"]))
```

### Working with Quizzes and Feedback

You can create quizzes and get feedback on user answers:

```python
import asyncio
from ai_tutor.manager import AITutorManager
from ai_tutor.agents.models import QuizUserAnswer, QuizUserAnswers

async def create_and_evaluate_quiz(file_paths):
    # Initialize the AI Tutor manager
    manager = AITutorManager(api_key="your-api-key")
    
    # Run the full workflow to generate a lesson and quiz
    await manager.upload_documents(file_paths)
    await manager.generate_lesson_plan()
    await manager.generate_lesson_content()
    quiz = await manager.generate_quiz()
    
    # Create user answers (in a real application, these would come from user input)
    user_answers = QuizUserAnswers(
        quiz_title=quiz.title,
        user_answers=[
            QuizUserAnswer(
                question_index=0,
                selected_option_index=0,  # User selected the first option
                time_taken_seconds=30
            ),
            QuizUserAnswer(
                question_index=1,
                selected_option_index=2,  # User selected the third option
                time_taken_seconds=45
            ),
            # Add more answers as needed
        ],
        total_time_taken_seconds=75
    )
    
    # Get feedback on the answers
    feedback = await manager.submit_quiz_answers(user_answers)
    
    print(f"Score: {feedback.correct_answers}/{feedback.total_questions}")
    print(f"Passed: {feedback.passed}")
    print(f"Overall feedback: {feedback.overall_feedback}")
    
    return feedback

# Run the function
feedback = asyncio.run(create_and_evaluate_quiz(["file1.pdf", "file2.pdf"]))
```

You can also run the complete workflow including quiz assessment:

```python
# Run the complete workflow including quiz assessment
result = asyncio.run(manager.run_full_workflow_with_quiz_teacher(["file1.pdf"]))
lesson_plan = result["lesson_plan"]
lesson_content = result["lesson_content"]
quiz = result["quiz"]
user_answers = result["user_answers"]  # Sample answers for demonstration
quiz_feedback = result["quiz_feedback"]
```

## Supported File Types

The AI Tutor supports many document types including:

- PDF files (.pdf)
- Word documents (.docx, .doc)
- Text files (.txt)
- Markdown files (.md)
- PowerPoint presentations (.pptx, .ppt)

## Viewing Traces

You can view detailed traces of the AI Tutor's execution on the OpenAI platform:

1. Go to [https://platform.openai.com/traces](https://platform.openai.com/traces)
2. Find the trace ID displayed after running the AI Tutor
3. Click on the trace to view detailed information about the execution

## License

MIT 