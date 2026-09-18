# FalconLLM custom agent core

Python 3.11+; development branch: `Abhi-agent-core`.

The decision loop is our own code:

User → AgentEngine → LLM → validated AgentAction → ToolRegistry → observation → LLM → final answer.

There is no agent framework or keyword router. Gemini uses the direct `google-genai`
async SDK, with automatic SDK function calling disabled. Tool schemas are injected
on every iteration. `plan` contains only a short operational summary, not private
chain-of-thought. The engine records actions and observations, allows recovery from
parse/LLM/tool errors, and returns `max_iterations` if it exhausts its budget.

## Setup on macOS

Run from the repository root. Reuse the existing `venv`; if it does not exist,
create it with `python3 -m venv venv`.

```sh
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python scripts/check_setup.py
./venv/bin/python -m pytest -v
./venv/bin/python run_demo.py
./venv/bin/python -m compileall backend tests scripts
```

The global `python3` may not have these dependencies. Either keep using the explicit
venv interpreter or run `source venv/bin/activate` before using `python3`.
Only the five direct runtime/test dependencies are listed in requirements.txt.
Existing unused packages in the venv do not need to be removed.

## Real Gemini run

If `.env` does not exist, copy `.env.example` to `.env`, then set `GEMINI_API_KEY`.
Keep an existing `.env`; do not overwrite it. It is ignored by Git.

```sh
./venv/bin/python run_real_agent.py
# Or run without a terminal prompt:
./venv/bin/python run_real_agent.py --query 'What is 40 + 2?'
```

Configuration is read when `AgentConfig()` is instantiated. It loads the root `.env`
without overriding existing environment variables. `AgentConfig(env_file=None)`
skips dotenv loading for isolated tests. Missing credentials produce setup instructions
and exit code 1; mock tests and the demo require no key. The CLI prints status, answer,
iterations, and JSON trace entries. An unfinished run also exits with code 1.

| Setting | Default / meaning |
| --- | --- |
| `LLM_PROVIDER` | `gemini`; only implemented real provider |
| `GEMINI_API_KEY` | Required only for real Gemini requests |
| `OPENAI_API_KEY` | Read for future integration; no OpenAI adapter yet |
| `GEMINI_MODEL` | `gemini-3.5-flash` |
| `MAX_AGENT_ITERATIONS` | Positive integer, default `10` |
| `LLM_TIMEOUT` | Finite positive seconds per call, default `30` |

The previous unverified `gemini-3.8-flash` default was replaced with
[`gemini-3.5-flash`, documented as stable](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash).
Model availability still depends on the API account. The SDK accepts model IDs as
strings; the server determines access. A custom `GEMINI_MODEL` is respected.
Transport errors omit raw SDK request details so credentials cannot enter traces.

## Integration boundaries

- `BaseLLMClient.generate(messages, tools)` is async and returns text. Production
  adapters own their transport and resource cleanup, not the agent loop.
- The structural `backend.tools.registry.ToolRegistry` contract exposes synchronous
  `get_tool_schemas() -> list[dict]` and async
  `execute(tool_name, arguments) -> ToolResult`. Parameters use JSON Schema.
  Tool failures should return `ToolResult(success=False, tool=..., error=...)`.
  Raised execution exceptions also become observations. Schema discovery is expected
  to succeed and return JSON-serializable definitions.
- Replace `MockToolRegistry` in the entry points with the teammate's implementation;
  the engine needs no tool-specific changes. The mock calculator supports bounded
  basic arithmetic without `eval`. Mock search is synthetic and does not browse.
- State/history and trace are local to each run. The existing `trace_manager` argument
  remains reserved and unused until the state/tracing teammate defines its interface.
  Trace step numbers start at 1 per run. FastAPI/frontend can consume
  `AgentResult.model_dump(mode="json")` without importing CLI code.
- Cancellation propagates; ordinary failures are bounded by the iteration budget.
  Persistent errors currently end with `max_iterations`; backoff, durable state,
  and specialized recovery policies remain future integration work.
