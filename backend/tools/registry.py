"""Tool registration, discovery, and safe asynchronous dispatch."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from backend.agent.schemas import ToolResult

from .base_tool import BaseTool


class ToolRegistry:
    """Stores tools; a separate agent engine decides which one to execute."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("Only BaseTool instances can be registered.")
        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if tool.name in self._tools:
            raise ValueError(f"A tool named '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def list_tools(self) -> list[str]:
        return list(self._tools)

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.metadata() for tool in self._tools.values()]

    def get_tool_descriptions(self) -> list[dict[str, Any]]:
        """Compatibility alias for the tools branch's original API."""
        return self.get_tool_schemas()

    async def execute(self, name: str, arguments: Mapping[str, Any] | None) -> ToolResult:
        tool = self.get_tool(name)
        if tool is None:
            return ToolResult(success=False, tool=str(name), error=f"Unknown tool: {name}")
        if not isinstance(arguments, Mapping):
            return ToolResult(success=False, tool=tool.name, error="Tool arguments must be an object.")

        try:
            if tool.blocking:
                outcome = await asyncio.to_thread(tool.execute, arguments)
            else:
                outcome = tool.execute(arguments)
            if not isinstance(outcome, ToolResult) or outcome.tool != tool.name:
                return ToolResult(success=False, tool=tool.name, error="Tool returned an invalid response.")
            return outcome
        except Exception:
            return ToolResult(success=False, tool=tool.name, error="Tool execution failed unexpectedly.")
