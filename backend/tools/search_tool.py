"""Provider-independent web search tool."""

from __future__ import annotations

from typing import Any, Mapping

from .base_tool import BaseTool, ToolResult
from .providers import SearchProvider, UnavailableSearchProvider


class SearchTool(BaseTool):
    blocking = True
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

    def __init__(
        self,
        provider: SearchProvider | None = None,
        default_max_results: int = 3,
        snippet_chars: int = 700,
    ) -> None:
        self.provider = provider or UnavailableSearchProvider()
        self.default_max_results = default_max_results
        self.snippet_chars = snippet_chars

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = arguments.get("query")
        max_results = arguments.get("max_results", self.default_max_results)
        if not isinstance(query, str) or not query.strip():
            return ToolResult(success=False, tool=self.name, error="Search query must be a non-empty string.")
        if len(query) > 500:
            return ToolResult(success=False, tool=self.name, error="Search query is too long.")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 10:
            return ToolResult(success=False, tool=self.name, error="max_results must be an integer from 1 to 10.")
        try:
            results = self.provider.search(query.strip(), max_results)
            compressed = []
            for result in results[:max_results]:
                if not isinstance(result, Mapping):
                    continue
                snippet = result.get("snippet", result.get("content", ""))
                compressed.append(
                    {
                        "title": str(result.get("title", "")),
                        "url": str(result.get("url", "")),
                        "snippet": str(snippet)[: self.snippet_chars],
                    }
                )
            return ToolResult(success=True, tool=self.name, result=compressed)
        except RuntimeError as exc:
            return ToolResult(success=False, tool=self.name, error=str(exc))
        except Exception:
            return ToolResult(success=False, tool=self.name, error="Search failed unexpectedly.")
