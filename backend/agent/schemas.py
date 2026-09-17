from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


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


class TraceStep(BaseModel):
    step: int
    type: str
    content: Optional[str] = None
    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None
