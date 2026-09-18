# FalconLLM

FalconLLM is a custom AI agent framework built for Track 2, “Build the Brain, Not the Puppet.” It does not use LangChain, CrewAI, AutoGen, LlamaIndex agents, keyword routing, or a ready-made orchestrator.

The runtime flow is:

```text
USER → AgentEngine → LLM → validated AgentAction → ToolRegistry
     → ToolResult observation → State/Trace → repeat or FINAL
```

The LLM chooses tools from their descriptions and JSON schemas. The engine executes a bounded `PLAN → ACT → OBSERVE → REPEAT → FINAL` loop. A plan is only a concise operational summary; hidden chain-of-thought is neither requested nor exposed.

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
| `LLM_PROVIDER` | `gemini` | LLM adapter; Gemini is currently implemented |
| `GEMINI_API_KEY` | empty | Required for the live API and real CLI |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model identifier |
| `MAX_AGENT_ITERATIONS` | `10` | Hard loop bound |
| `LLM_TIMEOUT` | `30` | LLM request timeout in seconds |
| `SEARCH_PROVIDER` | empty | Set to `tavily` for live search |
| `TAVILY_API_KEY` / `SEARCH_API_KEY` | empty | Optional Tavily credential |
| `DATABASE_URL` | `sqlite:///agent.db` | Run and trace persistence |
| `BACKEND_HOST` | `127.0.0.1` | Documented backend bind host |
| `BACKEND_PORT` | `8000` | Documented backend port |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend API target (set in `frontend/.env`) |

Without a search key, the search tool returns a structured “not configured” failure; it never fabricates production results. Without a Gemini key, tests and `run_demo.py` still work, while live API run creation returns a setup error.

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

Run a live Gemini-directed task:

```sh
./venv/bin/python run_real_agent.py --query "What is 40 + 2?"
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

Malformed LLM output, LLM failures, unknown tools, bad arguments, tool failures, repeated actions, loops, and max-iteration exhaustion become structured state and trace events. Failures are returned to the next LLM iteration as observations so it may choose a retry, another tool, or a limited final answer. Guards and iteration limits prevent endless retries. Trace payloads redact credential-shaped fields and SQLite persistence is injectable, keeping tests isolated.
