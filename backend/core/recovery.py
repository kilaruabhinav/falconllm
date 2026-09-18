"""Failure recovery policies, exponential backoff, and decision engine.

Provides provider-independent and tool-independent recovery logic for the agent framework,
safeguarding against infinite retry loops and non-idempotent side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import random
import time
from typing import Any, Callable

from errors import AgentError, ErrorType

logger = logging.getLogger("ai_agent.recovery")

# Tools considered safe to retry without harmful side effects
DEFAULT_IDEMPOTENT_TOOLS: set[str] = {
    "search",
    "read",
    "read_file",
    "get_info",
    "lookup",
    "check_status",
    "fetch_url",
    "calculate",
}


@dataclass
class RecoveryDecision:
    """Structured recovery outcome and recommendation."""

    action: str  # RETRY, REPLAN, SWITCH_TOOL, STOP
    reason: str
    retry: bool = False
    recoverable: bool = False
    retry_allowed: bool = False
    retry_delay: float = 0.0
    max_retries: int = 2
    recommended_action: str = "stop"
    observation: dict[str, Any] = field(default_factory=dict)
    terminate_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "retry": self.retry,
            "recoverable": self.recoverable,
            "retry_allowed": self.retry_allowed,
            "retry_delay": self.retry_delay,
            "max_retries": self.max_retries,
            "recommended_action": self.recommended_action,
            "observation": self.observation,
            "terminate_run": self.terminate_run,
        }


class RecoveryManager:
    """Evaluates failures and determines bounded, backoff-aware recovery actions."""

    def __init__(
        self,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        enable_jitter: bool = True,
        idempotent_tools: set[str] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.enable_jitter = enable_jitter
        self.idempotent_tools = set(idempotent_tools or DEFAULT_IDEMPOTENT_TOOLS)
        self.sleep_fn = sleep_fn or time.sleep
        self.retry_counts: dict[str, int] = {}

    def _calculate_backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Calculate exponential backoff duration with optional random jitter."""
        if retry_after is not None and retry_after > 0:
            return min(self.max_delay, retry_after)

        backoff = self.base_delay * (2 ** attempt)
        if self.enable_jitter:
            jitter = random.uniform(0.0, min(1.0, self.base_delay))
            backoff += jitter

        return min(self.max_delay, round(backoff, 3))

    def is_tool_idempotent(self, tool_name: str | None) -> bool:
        """Check whether the tool is known to be idempotent and safe to retry."""
        if not tool_name:
            return False
        return tool_name.strip().lower() in self.idempotent_tools

    def recover(
        self,
        error: AgentError | Exception,
        tool_name: str | None = None,
        is_idempotent: bool | None = None,
    ) -> RecoveryDecision:
        """Evaluate an error and construct a structured RecoveryDecision."""
        if not isinstance(error, AgentError):
            agent_err = AgentError(
                error_type=ErrorType.UNKNOWN_ERROR,
                message=str(error),
                retryable=False,
                original_exception=error,
            )
        else:
            agent_err = error

        error_type = agent_err.error_type
        error_key = f"{error_type.value}:{tool_name or 'global'}"
        current_retries = self.retry_counts.get(error_key, 0)

        # Build base observation for the agent
        base_observation = agent_err.to_observation()

        # Check tool idempotency: don't retry non-idempotent tools unless explicitly configured
        idempotent = is_idempotent if is_idempotent is not None else self.is_tool_idempotent(tool_name)

        # 1. Non-retryable governance stops
        if error_type in (ErrorType.MAX_ITERATIONS_EXCEEDED, ErrorType.CANCELLED):
            return RecoveryDecision(
                action="STOP",
                reason=agent_err.message,
                retry=False,
                recoverable=False,
                retry_allowed=False,
                retry_delay=0.0,
                max_retries=self.max_retries,
                recommended_action="terminate_run",
                observation=base_observation,
                terminate_run=True,
            )

        # 2. Unknown tool or invalid arguments -> REPLAN / SWITCH_TOOL
        if error_type == ErrorType.UNKNOWN_TOOL:
            base_observation["recommended_action"] = "switch_tool"
            return RecoveryDecision(
                action="STOP",  # backward compatibility reason: STOP execution of this tool call
                reason="The requested tool does not exist. Agent should select an available tool.",
                retry=False,
                recoverable=True,
                retry_allowed=False,
                retry_delay=0.0,
                max_retries=self.max_retries,
                recommended_action="choose_different_tool",
                observation=base_observation,
                terminate_run=False,
            )

        if error_type in (ErrorType.INVALID_TOOL_ARGUMENTS, ErrorType.EMPTY_TOOL_RESPONSE):
            base_observation["recommended_action"] = "replan"
            return RecoveryDecision(
                action="REPLAN",
                reason=agent_err.message,
                retry=False,
                recoverable=True,
                retry_allowed=False,
                retry_delay=0.0,
                max_retries=self.max_retries,
                recommended_action="replan_arguments",
                observation=base_observation,
                terminate_run=False,
            )

        if error_type in (ErrorType.REPEATED_ACTION, ErrorType.AGENT_LOOP_DETECTED):
            base_observation["recommended_action"] = "break_loop"
            return RecoveryDecision(
                action="REPLAN",
                reason=agent_err.message,
                retry=False,
                recoverable=True,
                retry_allowed=False,
                retry_delay=0.0,
                max_retries=self.max_retries,
                recommended_action="break_loop_replan",
                observation=base_observation,
                terminate_run=False,
            )

        # 3. Retryable errors (e.g. timeouts, rate limits, provider errors)
        if agent_err.retryable and current_retries < self.max_retries:
            # Check non-idempotent tool protection
            if tool_name and not idempotent and error_type not in (
                ErrorType.LLM_API_TIMEOUT,
                ErrorType.LLM_RATE_LIMIT,
                ErrorType.LLM_PROVIDER_ERROR,
            ):
                logger.warning(
                    "Skipping retry for non-idempotent tool '%s' on %s",
                    tool_name,
                    error_type.value,
                )
                base_observation["message"] = (
                    f"{agent_err.message} (Automatic retry suppressed: "
                    f"tool '{tool_name}' is not marked idempotent)"
                )
                return RecoveryDecision(
                    action="REPLAN",
                    reason="Tool is non-idempotent; automatic retry disallowed.",
                    retry=False,
                    recoverable=True,
                    retry_allowed=False,
                    retry_delay=0.0,
                    max_retries=self.max_retries,
                    recommended_action="replan_or_manual_verify",
                    observation=base_observation,
                    terminate_run=False,
                )

            # Compute backoff
            retry_delay = self._calculate_backoff(
                current_retries,
                retry_after=agent_err.retry_after_ms / 1000.0 if agent_err.retry_after_ms else None,
            )
            self.retry_counts[error_key] = current_retries + 1
            base_observation["retry_delay"] = retry_delay
            base_observation["attempt"] = current_retries + 1

            return RecoveryDecision(
                action="RETRY",
                reason=f"Retrying after {error_type.value} (attempt {current_retries + 1}/{self.max_retries})",
                retry=True,
                recoverable=True,
                retry_allowed=True,
                retry_delay=retry_delay,
                max_retries=self.max_retries,
                recommended_action="retry_with_backoff",
                observation=base_observation,
                terminate_run=False,
            )

        # 4. Exceeded max retries or non-retryable failure
        return RecoveryDecision(
            action="STOP",
            reason=f"Exceeded max retries or non-retryable error: {agent_err.message}",
            retry=False,
            recoverable=False,
            retry_allowed=False,
            retry_delay=0.0,
            max_retries=self.max_retries,
            recommended_action="stop",
            observation=base_observation,
            terminate_run=True,
        )

    def reset(self, error_key: str | None = None) -> None:
        """Reset retry counters."""
        if error_key:
            self.retry_counts.pop(error_key, None)
        else:
            self.retry_counts.clear()

    def handle_tool_result(
        self,
        run_id: str,
        state: Any,
        tool_name: str,
        result: Any,
    ) -> RecoveryDecision:
        """Process tool outcome, translating failures into structured recovery decisions."""
        if isinstance(result, Exception):
            return self.recover(result, tool_name=tool_name)

        if isinstance(result, dict) and result.get("success") is False:
            err = AgentError(
                error_type=result.get("error_type", ErrorType.TOOL_EXCEPTION),
                message=result.get("message", f"Tool '{tool_name}' failed"),
                retryable=result.get("retryable"),
            )
            return self.recover(err, tool_name=tool_name)


        # Successful tool outcome
        return RecoveryDecision(
            action="CONTINUE",
            reason="Tool executed successfully",
            retry=False,
            recoverable=True,
            retry_allowed=False,
            recommended_action="continue_plan",
            observation={"success": True, "result": result},
            terminate_run=False,
        )