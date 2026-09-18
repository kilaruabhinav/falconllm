"""Temporary test tools. Search returns synthetic data, never live results."""
import ast
import math
import operator
from copy import deepcopy
from typing import Any

from backend.agent.schemas import ToolResult


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _calculate(expression: str) -> int | float:
    """Bounded arithmetic only: no eval, names, calls, attributes or powers."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        raise ValueError("expression must be a non-empty string of at most 200 characters.")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("Expression is too complex.")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            value = _BINARY[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            value = _UNARY[type(node.op)](visit(node.operand))
        else:
            raise ValueError("Only numbers, parentheses and +, -, *, /, //, % are supported.")
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Arithmetic result is out of range.")
        return value

    return visit(tree.body)


class MockToolRegistry:
    def __init__(self):
        self.tools = {
            "calculator": {
                "name": "calculator",
                "description": "Mock arithmetic calculator: numbers, parentheses, +, -, *, /, //, %.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
            "search": {
                "name": "search",
                "description": "Mock search returning synthetic test data; does not search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return deepcopy(list(self.tools.values()))

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(success=False, tool=tool_name, error=f"Unknown tool: {tool_name}")
        try:
            if tool_name == "calculator":
                result = _calculate(arguments.get("expression"))
            elif tool_name == "search":
                query = arguments.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query must be a non-empty string.")
                result = f"Mock search result for: {query}"
            else:
                raise ValueError("Unhandled mock tool.")
            return ToolResult(success=True, tool=tool_name, result=result)
        except (ValueError, SyntaxError, ArithmeticError) as exc:
            return ToolResult(success=False, tool=tool_name, error=str(exc))
