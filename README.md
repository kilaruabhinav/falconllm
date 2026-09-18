# FalconLLM Tool System

This is Teammate 1's modular tool layer. It performs tool discovery and safe
execution only; it does not contain an agent loop or keyword-based routing.

## Public integration API

The agent core should import `ToolRegistry` and register tool instances:

```python
from backend.tools import CalculatorTool, FileReaderTool, SearchTool, ToolRegistry
from backend.tools.providers import TavilySearchProvider

registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register(FileReaderTool(root_directory="data"))
registry.register(SearchTool(TavilySearchProvider()))

tool_definitions = registry.get_tool_descriptions()
observation = registry.execute("calculator", {"expression": "25 * 4"})
```

`get_tool_descriptions()` supplies each tool's `name`, `description`, and
`input_schema` for an LLM function-calling API. `execute()` always returns a
dictionary shaped like:

```json
{"success": true, "tool": "calculator", "result": 100, "error": null}
```

Unknown names, invalid arguments, provider errors, and unexpected tool failures
are returned as structured error observations, rather than crashing the engine.

## Tools

- `CalculatorTool`: bounded AST-based arithmetic; it never uses `eval`.
- `SearchTool`: depends on the `SearchProvider` interface. It defaults to a
  failure observation when no provider is configured and never makes up results.
  `TavilySearchProvider` uses `TAVILY_API_KEY` from the process environment.
- `FileReaderTool`: reads UTF-8 `.txt`, `.md`, `.csv`, `.json`, and `.log` files
  only below its configured root directory. It blocks absolute paths, traversal,
  empty files, unsupported types, and oversized files.

## Setup and test

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Copy `.env.example` to `.env` and provide a real Tavily key only for live web
search. `.env` is ignored by Git and must never be committed.
