"""A bounded arithmetic evaluator that never executes arbitrary Python code."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Mapping

from .base_tool import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Safely evaluates a basic arithmetic expression."
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, for example (25 * 4) / 5.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }
    _MAX_EXPRESSION_LENGTH = 500
    _MAX_AST_NODES = 100
    _MAX_EXPONENT = 1_000
    _OPERATIONS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        expression = arguments.get("expression")
        if not isinstance(expression, str):
            return ToolResult(False, self.name, error="Expression must be a string.")
        expression = expression.strip()
        if not expression:
            return ToolResult(False, self.name, error="Expression cannot be empty.")
        if len(expression) > self._MAX_EXPRESSION_LENGTH:
            return ToolResult(False, self.name, error="Expression is too long.")

        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > self._MAX_AST_NODES:
                raise ValueError("Expression is too complex.")
            result = self._calculate(tree.body)
            if not math.isfinite(result):
                raise ValueError("Result is not finite.")
            return ToolResult(True, self.name, result=result)
        except ZeroDivisionError:
            return ToolResult(False, self.name, error="Cannot divide by zero.")
        except (SyntaxError, TypeError, ValueError, OverflowError):
            return ToolResult(False, self.name, error="Invalid mathematical expression.")

    def _calculate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed.")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._calculate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            operation = self._OPERATIONS.get(type(node.op))
            if operation is None:
                raise ValueError("Operator is not allowed.")
            left, right = self._calculate(node.left), self._calculate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > self._MAX_EXPONENT:
                raise ValueError("Exponent is too large.")
            return operation(left, right)
        raise ValueError("Expression contains unsupported syntax.")
