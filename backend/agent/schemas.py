from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolResult(BaseModel):
    success: bool
    tool: str
    result: Optional[Any] = None
    error: Optional[str] = None


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["tool", "final"]

    plan: str = Field(
        description="Short operational summary of the next action"
    )

    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None

    @model_validator(mode="after")
    def validate_action(self):

        if not self.plan.strip():
            raise ValueError("Action requires a non-empty operational summary.")

        if self.type == "tool":

            if not self.tool or not self.tool.strip():
                raise ValueError(
                    "Tool action requires 'tool'."
                )

            if self.arguments is None:
                raise ValueError("Tool action requires an arguments object (possibly empty).")
            if self.answer is not None:
                raise ValueError("Tool action must not include an answer.")

        elif self.type == "final":

            if self.answer is None or not self.answer.strip():
                raise ValueError(
                    "Final action requires 'answer'."
                )

            if self.tool is not None or self.arguments is not None:
                raise ValueError("Final action must not include tool or arguments.")

        return self


class TraceStep(BaseModel):
    step: int
    type: str
    content: Optional[str] = None
    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class AgentResult(BaseModel):
    status: Literal[
        "completed",
        "failed",
        "max_iterations"
    ]

    answer: Optional[str] = None
    iterations: int
    trace: list[TraceStep]
