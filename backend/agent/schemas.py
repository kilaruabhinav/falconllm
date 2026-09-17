from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, model_validator


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
    type: Literal["tool", "final"]

    plan: str = Field(
        description="Short operational summary of the next action"
    )

    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None

    @model_validator(mode="after")
    def validate_action(self):

        if self.type == "tool":

            if not self.tool:
                raise ValueError(
                    "Tool action requires 'tool'."
                )

            if self.arguments is None:
                self.arguments = {}

        elif self.type == "final":

            if self.answer is None:
                raise ValueError(
                    "Final action requires 'answer'."
                )

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
