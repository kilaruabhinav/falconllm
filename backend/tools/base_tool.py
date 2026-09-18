"""Shared contracts for tools exposed to the custom agent engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ToolResult:
    """Serializable observation returned by every tool invocation."""

    success: bool
    tool: str
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseTool(ABC):
    """Contract implemented by every tool registered with :class:`ToolRegistry`."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Execute this tool and return a structured observation."""

    def metadata(self) -> dict[str, Any]:
        """Return the metadata an LLM function-calling client needs for discovery."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
