from typing import Any, Dict, List

from backend.agent.schemas import ToolResult


class MockToolRegistry:

    def __init__(self):

        self.tools = {
            "calculator": {
                "name": "calculator",
                "description": (
                    "Performs basic mathematical calculations."
                ),
                "parameters": {
                    "expression": "string"
                },
            },
            "search": {
                "name": "search",
                "description": (
                    "Searches for information."
                ),
                "parameters": {
                    "query": "string"
                },
            },
        }

    def get_tool_schemas(
        self,
    ) -> List[Dict[str, Any]]:

        return list(
            self.tools.values()
        )

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:

        if tool_name not in self.tools:

            return ToolResult(
                success=False,
                tool=tool_name,
                error=(
                    f"Unknown tool: "
                    f"{tool_name}"
                ),
            )

        if tool_name == "calculator":

            expression = arguments.get(
                "expression"
            )

            if expression == "40 + 2":

                return ToolResult(
                    success=True,
                    tool=tool_name,
                    result=42,
                )

            return ToolResult(
                success=False,
                tool=tool_name,
                error=(
                    "Mock calculator only supports "
                    "'40 + 2' for this test."
                ),
            )

        if tool_name == "search":

            query = arguments.get(
                "query",
                ""
            )

            return ToolResult(
                success=True,
                tool=tool_name,
                result=(
                    f"Mock search result for: "
                    f"{query}"
                ),
            )

        return ToolResult(
            success=False,
            tool=tool_name,
            error="Unhandled mock tool.",
        )
