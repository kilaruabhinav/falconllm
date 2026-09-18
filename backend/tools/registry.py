"""Tool registration, discovery, and safe dispatch with no routing policy."""

from __future__ import annotations

from typing import Any, Mapping

from .base_tool import BaseTool, ToolResult


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

    def get_tool_descriptions(self) -> list[dict[str, Any]]:
        return [tool.metadata() for tool in self._tools.values()]

    def execute(self, name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        tool = self.get_tool(name)
        if tool is None:
            return ToolResult(False, str(name), error=f"Unknown tool: {name}").to_dict()
        if not isinstance(arguments, Mapping):
            return ToolResult(False, tool.name, error="Tool arguments must be an object.").to_dict()

        try:
            outcome = tool.execute(arguments)
            if not isinstance(outcome, ToolResult) or outcome.tool != tool.name:
                return ToolResult(False, tool.name, error="Tool returned an invalid response.").to_dict()
            return outcome.to_dict()
        except Exception:
            return ToolResult(False, tool.name, error="Tool execution failed unexpectedly.").to_dict()
