"""Failure taxonomy, error models, and exception hierarchy for the AI Agent Framework.

Provides structured, categorized error representations independent of concrete tools
or LLM providers, ensuring failures are never silently swallowed.
"""

from __future__ import annotations

from datetime import datetime, UTC
from enum import Enum
import re
from typing import Any


def current_utc_time() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(UTC)


class ErrorType(str, Enum):
    """Categorized failure types across agent lifecycle."""

    # Tool execution failures
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    TOOL_EXCEPTION = "TOOL_EXCEPTION"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    INVALID_TOOL_ARGUMENTS = "INVALID_TOOL_ARGUMENTS"
    EMPTY_TOOL_RESPONSE = "EMPTY_TOOL_RESPONSE"

    # LLM interaction failures
    MALFORMED_LLM_OUTPUT = "MALFORMED_LLM_OUTPUT"
    LLM_API_TIMEOUT = "LLM_API_TIMEOUT"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"

    # Agent loop / governance failures
    REPEATED_ACTION = "REPEATED_ACTION"
    AGENT_LOOP_DETECTED = "AGENT_LOOP_DETECTED"
    MAX_ITERATIONS_EXCEEDED = "MAX_ITERATIONS_EXCEEDED"

    # System & infrastructure failures
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"
    SERIALIZATION_ERROR = "SERIALIZATION_ERROR"
    CANCELLED = "CANCELLED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

    # Backward compatibility aliases for existing codebase
    INVALID_ARGUMENTS = "INVALID_TOOL_ARGUMENTS"
    EMPTY_RESPONSE = "EMPTY_TOOL_RESPONSE"
    MALFORMED_LLM_JSON = "MALFORMED_LLM_OUTPUT"
    LLM_TIMEOUT = "LLM_API_TIMEOUT"
    RATE_LIMIT = "LLM_RATE_LIMIT"
    STUCK_LOOP = "AGENT_LOOP_DETECTED"
    MAX_ITERATIONS = "MAX_ITERATIONS_EXCEEDED"


DEFAULT_RECOVERY_ACTIONS: dict[ErrorType, str] = {
    ErrorType.UNKNOWN_TOOL: "choose_different_tool",
    ErrorType.TOOL_EXCEPTION: "retry_or_replan",
    ErrorType.TOOL_TIMEOUT: "retry_with_backoff",
    ErrorType.INVALID_TOOL_ARGUMENTS: "replan_arguments",
    ErrorType.EMPTY_TOOL_RESPONSE: "retry_or_replan",
    ErrorType.MALFORMED_LLM_OUTPUT: "retry_generation",
    ErrorType.LLM_API_TIMEOUT: "retry_with_backoff",
    ErrorType.LLM_RATE_LIMIT: "wait_and_retry",
    ErrorType.LLM_PROVIDER_ERROR: "retry_with_backoff",
    ErrorType.REPEATED_ACTION: "stop_or_replan",
    ErrorType.AGENT_LOOP_DETECTED: "break_loop_replan",
    ErrorType.MAX_ITERATIONS_EXCEEDED: "terminate_run",
    ErrorType.PERSISTENCE_ERROR: "retry_storage",
    ErrorType.SERIALIZATION_ERROR: "sanitize_payload",
    ErrorType.CANCELLED: "abort_immediately",
    ErrorType.UNKNOWN_ERROR: "terminate_run",
}

DEFAULT_RETRYABLE: dict[ErrorType, bool] = {
    ErrorType.UNKNOWN_TOOL: False,
    ErrorType.TOOL_EXCEPTION: False,
    ErrorType.TOOL_TIMEOUT: True,
    ErrorType.INVALID_TOOL_ARGUMENTS: False,
    ErrorType.EMPTY_TOOL_RESPONSE: False,
    ErrorType.MALFORMED_LLM_OUTPUT: True,
    ErrorType.LLM_API_TIMEOUT: True,
    ErrorType.LLM_RATE_LIMIT: True,
    ErrorType.LLM_PROVIDER_ERROR: True,
    ErrorType.REPEATED_ACTION: False,
    ErrorType.AGENT_LOOP_DETECTED: False,
    ErrorType.MAX_ITERATIONS_EXCEEDED: False,
    ErrorType.PERSISTENCE_ERROR: True,
    ErrorType.SERIALIZATION_ERROR: False,
    ErrorType.CANCELLED: False,
    ErrorType.UNKNOWN_ERROR: False,
}

SENSITIVE_PATTERN = re.compile(
    r"(^|_)(api[_-]?key|secret|password|bearer|credential|private[_-]?key)($|_)|(^|_)(access[_-]?token|refresh[_-]?token|auth[_-]?token|secret[_-]?token|jwt|session[_-]?token)($|_)|^token$|^authorization$",
    re.IGNORECASE,
)



def redact_sensitive_data(val: Any) -> Any:
    """Recursively scrub sensitive keys and credential patterns."""
    if isinstance(val, dict):
        cleaned: dict[str, Any] = {}
        for k, v in val.items():
            if SENSITIVE_PATTERN.search(str(k)):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = redact_sensitive_data(v)
        return cleaned
    if isinstance(val, list):
        return [redact_sensitive_data(item) for item in val]
    if isinstance(val, tuple):
        return tuple(redact_sensitive_data(item) for item in val)
    return val


class AgentError(Exception):
    """Base exception for all agent framework failures."""

    def __init__(
        self,
        error_type: ErrorType | str,
        message: str,
        retryable: bool | None = None,
        error_code: str | None = None,
        recommended_action: str | None = None,
        original_exception: Exception | str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        retry_after_ms: float | None = None,
    ) -> None:
        if isinstance(error_type, str):
            try:
                self.error_type = ErrorType(error_type)
            except ValueError:
                # Handle direct name matching if value differs
                self.error_type = ErrorType[error_type] if error_type in ErrorType.__members__ else ErrorType.UNKNOWN_ERROR
        else:
            self.error_type = error_type

        self.message = message
        self.retryable = (
            retryable if retryable is not None
            else DEFAULT_RETRYABLE.get(self.error_type, False)
        )
        self.error_code = error_code or f"ERR_{self.error_type.name}"
        self.recommended_action = (
            recommended_action or DEFAULT_RECOVERY_ACTIONS.get(self.error_type, "stop")
        )
        self.original_exception = (
            type(original_exception).__name__
            if isinstance(original_exception, Exception)
            else (str(original_exception) if original_exception else None)
        )
        self.timestamp = timestamp or current_utc_time()
        self.metadata = redact_sensitive_data(metadata or {})
        self.retry_after_ms = retry_after_ms

        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert failure details into a safe dictionary without secrets."""
        data: dict[str, Any] = {
            "error_code": self.error_code,
            "error_type": self.error_type.value,
            "message": self.message,
            "retryable": self.retryable,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        if self.original_exception:
            data["original_exception"] = self.original_exception
        if self.retry_after_ms is not None:
            data["retry_after_ms"] = self.retry_after_ms
        return data

    def to_observation(self) -> dict[str, Any]:
        """Convert failure into an observation dict for agent reasoning."""
        obs: dict[str, Any] = {
            "success": False,
            "error_type": self.error_type.value,
            "message": self.message,
            "retryable": self.retryable,
            "recommended_action": self.recommended_action,
        }
        if self.retry_after_ms is not None:
            obs["retry_after_ms"] = self.retry_after_ms
        return obs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(code={self.error_code!r}, "
            f"type={self.error_type.value!r}, message={self.message!r}, "
            f"retryable={self.retryable})"
        )


# Specific typed subclasses
class ToolTimeoutError(AgentError):
    def __init__(self, message: str = "Tool timed out", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(ErrorType.TOOL_TIMEOUT, message, **kwargs)


class UnknownToolError(AgentError):
    def __init__(self, message: str = "Unknown tool requested", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorType.UNKNOWN_TOOL, message, **kwargs)


class ToolExecutionError(AgentError):
    def __init__(self, message: str = "Tool execution failed", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorType.TOOL_EXCEPTION, message, **kwargs)


class InvalidToolArgumentsError(AgentError):
    def __init__(self, message: str = "Invalid tool arguments", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorType.INVALID_TOOL_ARGUMENTS, message, **kwargs)


class EmptyToolResponseError(AgentError):
    def __init__(self, message: str = "Empty response returned by tool", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorType.EMPTY_TOOL_RESPONSE, message, **kwargs)


class MalformedLLMOutputError(AgentError):
    def __init__(self, message: str = "Malformed LLM output", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(ErrorType.MALFORMED_LLM_OUTPUT, message, **kwargs)


class LLMTimeoutError(AgentError):
    def __init__(self, message: str = "LLM call timed out", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(ErrorType.LLM_API_TIMEOUT, message, **kwargs)


class LLMRateLimitError(AgentError):
    def __init__(self, message: str = "LLM rate limit reached", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(ErrorType.LLM_RATE_LIMIT, message, **kwargs)


class LLMProviderError(AgentError):
    def __init__(self, message: str = "LLM provider internal error", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(ErrorType.LLM_PROVIDER_ERROR, message, **kwargs)


class RepeatedActionError(AgentError):
    def __init__(self, message: str = "Repeated identical action detected", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorType.REPEATED_ACTION, message, **kwargs)


class AgentLoopDetectedError(AgentError):
    def __init__(self, message: str = "Agent loop pattern detected", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorType.AGENT_LOOP_DETECTED, message, **kwargs)


class MaxIterationsExceededError(AgentError):
    def __init__(self, message: str = "Maximum iterations exceeded", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorType.MAX_ITERATIONS_EXCEEDED, message, **kwargs)


class PersistenceError(AgentError):
    def __init__(self, message: str = "Database or persistence error", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(ErrorType.PERSISTENCE_ERROR, message, **kwargs)


class SerializationError(AgentError):
    def __init__(self, message: str = "Payload serialization error", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorType.SERIALIZATION_ERROR, message, **kwargs)


class AgentCancelledError(AgentError):
    def __init__(self, message: str = "Agent run was cancelled", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorType.CANCELLED, message, **kwargs)