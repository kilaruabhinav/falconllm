"""Shared contracts for tools exposed to the custom agent engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from backend.agent.schemas import ToolResult

class BaseTool(ABC):
    """Contract implemented by every tool registered with :class:`ToolRegistry`."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Execute this tool and return a structured observation."""

    def metadata(self) -> dict[str, Any]:
        """Return the canonical definition consumed by the planner/LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


__all__ = ["BaseTool", "ToolResult"]
