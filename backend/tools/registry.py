"""Structural integration contract for mock and future production registries."""
from typing import Any, Protocol

from backend.agent.schemas import ToolResult


class ToolRegistry(Protocol):
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool definitions with JSON Schema parameter objects."""
        ...

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Return success/failure as a ToolResult; the engine also catches exceptions."""
        ...
