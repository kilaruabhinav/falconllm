# FalconLLM

FalconLLM is a custom AI agent framework built for Track 2, “Build the Brain, Not the Puppet.” It does not use LangChain, CrewAI, AutoGen, LlamaIndex agents, keyword routing, or a ready-made orchestrator.

The runtime flow is:

```text
USER → AgentEngine → LLM → validated AgentAction → ToolRegistry
     → ToolResult observation → State/Trace → repeat or FINAL
```

The LLM chooses tools from their descriptions and JSON schemas. The engine executes a bounded `PLAN → ACT → OBSERVE → REPEAT → FINAL` loop. A plan is only a concise operational summary; hidden chain-of-thought is neither requested nor exposed.

Provider availability is isolated behind the same `BaseLLMClient` interface:

```text
AgentEngine
    ↓
FallbackLLMClient
    ├── Gemini primary
    ├── Gemini fallback
    └── OpenAI
    ↓
AgentAction JSON → AgentEngine continues
```

The router changes transports without restarting the agent run. Existing actions, tool observations, iteration counters, state, and trace remain in the prompt sent to the next provider.

Runs are dispatched asynchronously. `POST /api/runs` returns a `run_id` immediately, while `GET /api/runs/{run_id}/stream` delivers real trace steps over Server-Sent Events as the engine is still executing. Each event is also written to SQLite; reconnecting clients first load persisted steps and resume after the last sequence number without duplicating entries. The stream sends non-persisted heartbeats during idle periods.

## Team contributions

- Abhinav — agent engine, Gemini client, parser, schemas, orchestration, mock LLM
- Rishith — modular tools, safe calculator, search adapter, safe file reader, registry
- Sasidhar — state, trace events, SQLite persistence, recovery, loop guards
- Rohith — FastAPI boundary, React/Vite UI, trace display and frontend contract

## Project layout

```text
backend/
  agent/       LLM-driven loop, prompts, parsing, provider clients and schemas
  tools/       tool contract, production registry and concrete tools
  core/        state, traces, guards, recovery, errors and SQLite persistence
  api/         FastAPI app, routes and AgentService
frontend/      React/Vite application
data/          files explicitly available to the file-reader tool
scripts/       setup diagnostics
tests/         unit, integration, persistence and API tests
run_demo.py    credential-free real-registry multi-tool demo
run_real_agent.py  live Gemini CLI
```

There is one canonical `AgentAction`, `ToolDefinition`, `ToolResult`, and `AgentResult` in `backend/agent/schemas.py`; the canonical persistent `TraceStep` is in `backend/core/trace.py`. `AgentState` is in `backend/core/state.py`.

## Setup (macOS/Linux)

Python 3.11+ and Node.js 20.19+ (or 22.12+) are recommended.

```sh
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
cd frontend && npm ci && cd ..
cp .env.example .env
```

Keep the generated `.env` local and add credentials only when needed. It is ignored by Git.

On Windows PowerShell, replace the virtualenv commands with:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER_MODE` | `fallback` | `fallback`, `gemini`, or `openai` |
| `GEMINI_API_KEY` | empty | Enables Gemini clients |
| `GEMINI_PRIMARY_MODEL` | `gemini-3.8-flash` | First model in fallback mode |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash-lite` | Second model in fallback mode |
| `OPENAI_API_KEY` | empty | Enables the OpenAI client |
| `OPENAI_MODEL` | `gpt-5-mini` | Configurable OpenAI Responses API model |
| `MAX_LLM_RETRIES` | `1` | Retries for transient timeout/service errors; quota failures are never retried |
| `LLM_PROVIDER_COOLDOWN_SECONDS` | `60` | Process cooldown after a provider/model failure |
| `SEARCH_TIMEOUT` | `10` | Search-provider request timeout in seconds |
| `MAX_AGENT_ITERATIONS` | `10` | Hard loop bound |
| `LLM_TIMEOUT` | `30` | LLM request timeout in seconds |
| `SEARCH_PROVIDER` | empty | Set to `tavily` for live search |
| `TAVILY_API_KEY` / `SEARCH_API_KEY` | empty | Optional Tavily credential |
| `DATABASE_URL` | `sqlite:///agent.db` | Run and trace persistence |
| `BACKEND_HOST` | `127.0.0.1` | Documented backend bind host |
| `BACKEND_PORT` | `8000` | Documented backend port |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend API target (set in `frontend/.env`) |

In `fallback` mode, missing provider keys are skipped. If both Gemini and OpenAI keys are missing, startup produces one clean configuration error. Set `LLM_PROVIDER_MODE=openai` for OpenAI only or `LLM_PROVIDER_MODE=gemini` to disable fallback and use Gemini only.

Quota/rate-limit, authentication, timeout, model-unavailable, and service-unavailable failures can advance the router. Quota failures immediately disable that model for the current run and place it in process cooldown; the chain never moves backward. Invalid `AgentAction` JSON, unknown tools, and tool failures remain agent-level recovery events and do not switch providers.

Without a search key, the search tool returns a structured “not configured” failure; it never fabricates production results.

## Run the system

Backend:

```sh
./venv/bin/python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```

Frontend, in another terminal:

```sh
cd frontend
npm run dev
```

The browser displays `RUNNING` immediately and updates PLAN, tool, recovery, provider-fallback, final, and terminal events live. Refreshing during an active run restores its persisted trace and reconnects to the SSE stream.

### Live API contract

```text
POST /api/runs                    -> { run_id, status: "running", dispatch_ms }
GET  /api/runs/{run_id}/stream    -> text/event-stream
GET  /api/runs/{run_id}/trace     -> persisted trace catch-up
GET  /api/runs/{run_id}           -> persisted run status/result
```

SSE trace records use their stable sequence as the event ID. Browsers reconnect with `Last-Event-ID`; callers may also pass `?after=<sequence>`.

Open `http://localhost:5173`. The UI submits to the real backend and renders operational plans, tool calls/results, failures, recovery events, final answers, and persisted run history.

API endpoints:

- `GET /health`
- `POST /api/runs` with `{"prompt":"..."}`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/trace`

## Demos and tests

The offline demo uses the real file-reader and calculator implementations with a deterministic mock LLM, proving a two-tool flow without credentials:

```sh
./venv/bin/python run_demo.py
```

Run a live provider-routed task:

```sh
./venv/bin/python run_real_agent.py --query "What is 40 + 2?"
```

Inspect and simulate the full fallback chain without consuming Gemini quota:

```sh
./venv/bin/python scripts/test_llm_fallback.py --simulate-gemini-429
```

Optionally make one small OpenAI request:

```sh
./venv/bin/python scripts/test_llm_fallback.py --live-openai
```

Validate everything:

```sh
./venv/bin/python scripts/check_setup.py
./venv/bin/python -m pytest -v
./venv/bin/python -m compileall backend tests scripts
cd frontend && npm ci && npm run build && npm run lint
```

## Adding a tool

1. Subclass `BaseTool` and provide `name`, `description`, `input_schema`, and `execute(arguments)`.
2. Return the shared `ToolResult` for success and failure; do not raise routine validation errors.
3. Register the tool in `create_tool_registry()`.
4. Add focused tool and agent-flow tests. The engine requires no tool-specific branches.

## Failure recovery

Malformed LLM output, unknown tools, bad arguments, tool failures, repeated actions, loops, and max-iteration exhaustion become structured state and trace events. Provider switching produces `LLM_PROVIDER_SELECTED`, `LLM_PROVIDER_ERROR`, and `LLM_FALLBACK` infrastructure events. Failures are returned to the next agent iteration as observations when appropriate so it may choose a retry, another tool, or a limited final answer. Guards and iteration limits prevent endless retries. Trace payloads redact credential-shaped fields and SQLite persistence is injectable, keeping tests isolated.

Latency metadata is attached without adding model calls: PLAN steps record LLM duration, TOOL_RESULT records tool/search duration, and RUN_COMPLETED records total run time plus aggregate SQLite-write timing. Search defaults to three results and sends only title, URL, and a bounded snippet to the agent. Planner history is bounded to the eight most recent items with large observations truncated. Provider SDK clients and their HTTP pools are reused across runs while each run gets isolated fallback-router state.
