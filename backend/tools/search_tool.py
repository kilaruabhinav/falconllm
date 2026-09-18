"""Provider-independent web search tool."""

from __future__ import annotations

from typing import Any, Mapping

from .base_tool import BaseTool, ToolResult
from .providers import SearchProvider, UnavailableSearchProvider


class SearchTool(BaseTool):
    name = "search"
    description = "Search the web using the configured search provider."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Topic or question to search for."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self.provider = provider or UnavailableSearchProvider()

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = arguments.get("query")
        max_results = arguments.get("max_results", 5)
        if not isinstance(query, str) or not query.strip():
            return ToolResult(False, self.name, error="Search query must be a non-empty string.")
        if len(query) > 500:
            return ToolResult(False, self.name, error="Search query is too long.")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 10:
            return ToolResult(False, self.name, error="max_results must be an integer from 1 to 10.")
        try:
            results = self.provider.search(query.strip(), max_results)
            return ToolResult(True, self.name, result=results)
        except RuntimeError as exc:
            return ToolResult(False, self.name, error=str(exc))
        except Exception:
            return ToolResult(False, self.name, error="Search failed unexpectedly.")
