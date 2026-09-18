"""Structured agent state management for the AI Agent Framework.

Provides thread-isolated, validated, and serializable agent execution state
with automated credential redaction, payload truncation, and timezone-aware UTC timestamps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from enum import Enum
import json
import re
from typing import Any
import uuid

# Pattern for sensitive credentials, API keys, passwords, and tokens
SENSITIVE_KEY_RE = re.compile(
    r"(^|_)(api[_-]?key|secret|password|bearer|credential|private[_-]?key)($|_)|(^|_)(access[_-]?token|refresh[_-]?token|auth[_-]?token|secret[_-]?token|jwt|session[_-]?token)($|_)|^token$|^authorization$",
    re.IGNORECASE,
)



def current_utc_time() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(dt: datetime | str | None) -> datetime:
    """Ensure datetime object is timezone-aware in UTC."""
    if dt is None:
        return current_utc_time()
    if isinstance(dt, str):
        parsed = datetime.fromisoformat(dt)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def redact_credentials(val: Any) -> Any:
    """Recursively scrub sensitive credential keys and tokens."""
    if isinstance(val, dict):
        scrubbed: dict[str, Any] = {}
        for k, v in val.items():
            if SENSITIVE_KEY_RE.search(str(k)):
                scrubbed[str(k)] = "[REDACTED]"
            else:
                scrubbed[str(k)] = redact_credentials(v)
        return scrubbed
    if isinstance(val, list):
        return [redact_credentials(item) for item in val]
    if isinstance(val, tuple):
        return tuple(redact_credentials(item) for item in val)
    return val


def truncate_payload(val: Any, max_chars: int = 50_000) -> Any:
    """Truncate payloads exceeding max_chars and append explicit marker."""
    if isinstance(val, str):
        if len(val) > max_chars:
            return {
                "truncated": True,
                "original_length": len(val),
                "preview": val[:max_chars] + "... [TRUNCATED]",
            }
        return val
    if isinstance(val, dict):
        truncated_dict: dict[str, Any] = {}
        for k, v in val.items():
            truncated_dict[k] = truncate_payload(v, max_chars)
        return truncated_dict
    if isinstance(val, list):
        return [truncate_payload(item, max_chars) for item in val]
    return val


def safe_json_default(obj: Any) -> Any:
    """Handle serialization for non-JSON native objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


class AgentStatus(str, Enum):
    """Execution status lifecycle for an agent run."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MAX_ITERATIONS_EXCEEDED = "MAX_ITERATIONS_EXCEEDED"
    CANCELLED = "CANCELLED"


@dataclass
class Action:
    """Represents a planned or executed action."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    action_type: str = "TOOL_CALL"
    timestamp: datetime = field(default_factory=current_utc_time)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)
        self.arguments = redact_credentials(self.arguments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "action_type": self.action_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToolCall:
    """Represents an invoked tool and its lifecycle result."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    result: Any = None
    error: str | None = None
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=current_utc_time)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)
        self.arguments = redact_credentials(self.arguments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status,
            "result": redact_credentials(self.result),
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Observation:
    """Represents the observation returned from an action/tool."""

    success: bool
    result: Any = None
    error_type: str | None = None
    message: str | None = None
    retryable: bool = False
    timestamp: datetime = field(default_factory=current_utc_time)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)
        self.result = redact_credentials(self.result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "error_type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ErrorRecord:
    """Represents a recorded agent failure event."""

    error_type: str
    message: str
    retryable: bool = False
    error_code: str | None = None
    timestamp: datetime = field(default_factory=current_utc_time)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TokenUsage:
    """Token consumption and cost tracking metadata."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
        }


@dataclass
class AgentState:
    """Structured, serializable state for an agent execution run."""

    user_query: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict[str, Any]] = field(default_factory=list)
    current_iteration: int = 0
    maximum_iterations: int = 10
    actions_taken: list[Action] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    final_answer: str | None = None
    status: AgentStatus | str = AgentStatus.CREATED
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=current_utc_time)
    updated_at: datetime = field(default_factory=current_utc_time)

    def __post_init__(self) -> None:
        # Validate critical numeric fields
        if self.maximum_iterations <= 0:
            raise ValueError(f"maximum_iterations must be positive, got {self.maximum_iterations}")
        if self.current_iteration < 0:
            raise ValueError(f"current_iteration cannot be negative, got {self.current_iteration}")

        # Ensure status is recognized
        if isinstance(self.status, str):
            try:
                self.status = AgentStatus(self.status)
            except ValueError:
                pass  # Allow custom string statuses if extended

        # Ensure timezone-aware UTC timestamps
        self.created_at = ensure_utc(self.created_at)
        self.updated_at = ensure_utc(self.updated_at)

        # Redact any credentials accidentally passed in metadata
        self.metadata = redact_credentials(self.metadata)

    # Backward compatibility property for existing engine.py
    @property
    def actions(self) -> list[Action]:
        return self.actions_taken

    @actions.setter
    def actions(self, val: list[Action]) -> None:
        self.actions_taken = val

    def add_message(self, role: str, content: str) -> None:
        """Add a conversation message to state."""
        sanitized_content = redact_credentials(content)
        self.messages.append({
            "role": role,
            "content": sanitized_content,
            "timestamp": current_utc_time().isoformat(),
        })
        self.updated_at = current_utc_time()

    def add_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        action_type: str = "TOOL_CALL",
    ) -> Action:
        """Record an action in state."""
        action = Action(
            tool_name=tool_name,
            arguments=arguments,
            action_type=action_type,
            timestamp=current_utc_time(),
        )
        self.actions_taken.append(action)
        self.updated_at = current_utc_time()
        return action

    def add_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> ToolCall:
        """Record a tool call in state."""
        call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            status="PENDING",
            timestamp=current_utc_time(),
        )
        self.tool_calls.append(call)
        self.updated_at = current_utc_time()
        return call

    def add_observation(
        self,
        success: bool,
        result: Any = None,
        error_type: str | None = None,
        message: str | None = None,
        retryable: bool = False,
    ) -> Observation:
        """Record an observation in state."""
        obs = Observation(
            success=success,
            result=result,
            error_type=error_type,
            message=message,
            retryable=retryable,
            timestamp=current_utc_time(),
        )
        self.observations.append(obs)
        self.updated_at = current_utc_time()
        return obs

    def add_error(
        self,
        error_type: str,
        message: str,
        retryable: bool = False,
        error_code: str | None = None,
    ) -> ErrorRecord:
        """Record an error in state."""
        err = ErrorRecord(
            error_type=error_type,
            message=message,
            retryable=retryable,
            error_code=error_code,
            timestamp=current_utc_time(),
        )
        self.errors.append(err)
        self.updated_at = current_utc_time()
        return err

    def update_token_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: float = 0.0,
    ) -> None:
        """Update token counts and cost estimates."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        self.estimated_cost += estimated_cost
        self.updated_at = current_utc_time()

    def set_status(self, status: AgentStatus | str) -> None:
        """Update status and touch updated_at."""
        if isinstance(status, str):
            try:
                self.status = AgentStatus(status)
            except ValueError:
                self.status = status
        else:
            self.status = status
        self.updated_at = current_utc_time()

    def to_dict(self, max_payload_chars: int = 50_000) -> dict[str, Any]:
        """Serialize state to a safe dictionary."""
        status_val = self.status.value if isinstance(self.status, Enum) else str(self.status)

        raw = {
            "run_id": self.run_id,
            "user_query": self.user_query,
            "status": status_val,
            "current_iteration": self.current_iteration,
            "maximum_iterations": self.maximum_iterations,
            "messages": self.messages,
            "actions": [a.to_dict() for a in self.actions_taken],
            "actions_taken": [a.to_dict() for a in self.actions_taken],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "observations": [o.to_dict() for o in self.observations],
            "errors": [e.to_dict() for e in self.errors],
            "final_answer": self.final_answer,
            "token_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "estimated_cost": self.estimated_cost,
            },
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        # Scrub credentials and truncate large payloads
        scrubbed = redact_credentials(raw)
        return truncate_payload(scrubbed, max_chars=max_payload_chars)

    def to_json(self, indent: int | None = None, max_payload_chars: int = 50_000) -> str:
        """Serialize state to JSON string with safe defaults."""
        return json.dumps(
            self.to_dict(max_payload_chars=max_payload_chars),
            default=safe_json_default,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Construct an AgentState instance from a dictionary."""
        actions_raw = data.get("actions_taken") or data.get("actions", [])
        actions = [
            Action(
                tool_name=a["tool_name"],
                arguments=a.get("arguments", {}),
                action_type=a.get("action_type", "TOOL_CALL"),
                timestamp=ensure_utc(a.get("timestamp")),
            )
            for a in actions_raw
        ]

        tool_calls = [
            ToolCall(
                call_id=tc.get("call_id", str(uuid.uuid4())),
                tool_name=tc["tool_name"],
                arguments=tc.get("arguments", {}),
                status=tc.get("status", "PENDING"),
                result=tc.get("result"),
                error=tc.get("error"),
                timestamp=ensure_utc(tc.get("timestamp")),
            )
            for tc in data.get("tool_calls", [])
        ]

        observations = [
            Observation(
                success=o["success"],
                result=o.get("result"),
                error_type=o.get("error_type"),
                message=o.get("message"),
                retryable=o.get("retryable", False),
                timestamp=ensure_utc(o.get("timestamp")),
            )
            for o in data.get("observations", [])
        ]

        errors = [
            ErrorRecord(
                error_type=e["error_type"],
                message=e["message"],
                retryable=e.get("retryable", False),
                error_code=e.get("error_code"),
                timestamp=ensure_utc(e.get("timestamp")),
            )
            for e in data.get("errors", [])
        ]

        token_usage = data.get("token_usage")
        if not isinstance(token_usage, dict):
            token_usage = {}
        input_tokens = token_usage.get("input_tokens", data.get("input_tokens", 0))
        output_tokens = token_usage.get("output_tokens", data.get("output_tokens", 0))
        total_tokens = token_usage.get("total_tokens", data.get("total_tokens", 0))
        estimated_cost = token_usage.get("estimated_cost", data.get("estimated_cost", 0.0))


        return cls(
            user_query=data.get("user_query", ""),
            run_id=data.get("run_id", str(uuid.uuid4())),
            messages=data.get("messages", []),
            current_iteration=data.get("current_iteration", 0),
            maximum_iterations=data.get("maximum_iterations", 10),
            actions_taken=actions,
            tool_calls=tool_calls,
            observations=observations,
            errors=errors,
            final_answer=data.get("final_answer"),
            status=data.get("status", AgentStatus.CREATED),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            metadata=data.get("metadata", {}),
            created_at=ensure_utc(data.get("created_at")),
            updated_at=ensure_utc(data.get("updated_at")),
        )

    @classmethod
    def from_json(cls, json_str: str) -> AgentState:
        """Construct an AgentState instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class StateManager:
    """High-level state manager for AgentEngine lifecycle operations."""

    @staticmethod
    def create(
        user_query: str,
        maximum_iterations: int = 10,
        run_id: str | None = None,
        **metadata: Any,
    ) -> AgentState:
        """Create and initialize a new AgentState."""
        return AgentState(
            user_query=user_query,
            maximum_iterations=maximum_iterations,
            run_id=run_id or str(uuid.uuid4()),
            metadata=metadata,
        )

    @staticmethod
    def update_iteration(state: AgentState, increment: int = 1) -> int:
        """Increment current iteration and touch updated_at."""
        state.current_iteration += increment
        state.updated_at = current_utc_time()
        return state.current_iteration

    @staticmethod
    def update_observation(
        state: AgentState,
        observation: dict[str, Any] | Observation,
    ) -> Observation:
        """Record an observation in state."""
        if isinstance(observation, Observation):
            state.observations.append(observation)
            state.updated_at = current_utc_time()
            return observation

        return state.add_observation(
            success=observation.get("success", False),
            result=observation.get("result"),
            error_type=observation.get("error_type"),
            message=observation.get("message"),
            retryable=observation.get("retryable", False),
        )

    @staticmethod
    def set_final_answer(state: AgentState, final_answer: str) -> None:
        """Set the final agent answer and mark state as COMPLETED."""
        state.final_answer = final_answer
        state.status = AgentStatus.COMPLETED
        state.updated_at = current_utc_time()

    @staticmethod
    def mark_failed(state: AgentState, error_message: str) -> None:
        """Mark state as FAILED."""
        state.status = AgentStatus.FAILED
        state.updated_at = current_utc_time()