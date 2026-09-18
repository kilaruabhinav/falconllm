"""Provider adapters for :class:`SearchTool`."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class SearchProvider(ABC):
    """Boundary that makes the search tool independent of a search vendor."""

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Return result objects normalized by the selected adapter."""


class UnavailableSearchProvider(SearchProvider):
    """Development default; intentionally never fabricates search results."""

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        raise RuntimeError("No search provider is configured.")


class TavilySearchProvider(SearchProvider):
    """Adapter for Tavily's REST search endpoint."""

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 10) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")
        try:
            import requests

            response = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "max_results": max_results},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError("Search provider request failed.") from exc

        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError("Search provider returned an invalid response.")
        return results
