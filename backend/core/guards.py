"""Guards, loop detection, and payload protection for the AI Agent Framework.

Provides structured pre-action validation:
1. Maximum iteration guard
2. Consecutive repeated-action guard with canonical argument ordering
3. Multi-period loop and cycle detection (A->B->A->B, A->B->C->A->B->C)
4. Payload size limits and explicit truncation markers
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .errors import (
    AgentError,
    ErrorType,
    MaxIterationsExceededError,
    RepeatedActionError,
    AgentLoopDetectedError,
)


def canonicalize_arguments(args: Any) -> str:
    """Deterministically serialize arguments so dictionary key order does not affect equality."""
    if isinstance(args, dict):
        # Recursively sort keys and canonicalize values
        sorted_dict = {
            str(k): json.loads(canonicalize_arguments(v))
            if isinstance(v, (dict, list))
            else v
            for k, v in sorted(args.items(), key=lambda item: str(item[0]))
        }
        return json.dumps(sorted_dict, sort_keys=True, separators=(",", ":"))
    if isinstance(args, (list, tuple)):
        canonical_list = [
            json.loads(canonicalize_arguments(item))
            if isinstance(item, (dict, list))
            else item
            for item in args
        ]
        return json.dumps(canonical_list, sort_keys=True, separators=(",", ":"))
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


@dataclass
class GuardResult:
    """Structured result of a guard check."""

    passed: bool
    violation_type: str | None = None
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violation_type": self.violation_type,
            "message": self.message,
            "details": self.details,
        }


class IterationGuard:
    """Guards against exceeding the configured maximum execution iterations."""

    def __init__(
        self,
        maximum_iterations: int = 10,
        raise_on_violation: bool = True,
    ) -> None:
        self.maximum_iterations = maximum_iterations
        self.raise_on_violation = raise_on_violation

    def evaluate(self, current_iteration: int) -> GuardResult:
        """Evaluate iteration count and return a structured GuardResult."""
        if current_iteration >= self.maximum_iterations:
            msg = (
                f"Maximum iterations exceeded: current {current_iteration} "
                f">= maximum {self.maximum_iterations}"
            )
            return GuardResult(
                passed=False,
                violation_type=ErrorType.MAX_ITERATIONS_EXCEEDED.value,
                message=msg,
                details={
                    "current_iteration": current_iteration,
                    "maximum_iterations": self.maximum_iterations,
                },
            )
        return GuardResult(passed=True)

    def check_iteration(self, current_iteration: int) -> bool:
        """Backward-compatible check method that raises if violation occurs."""
        result = self.evaluate(current_iteration)
        if not result.passed:
            if self.raise_on_violation:
                raise MaxIterationsExceededError(
                    message=result.message or "Maximum iterations exceeded",
                    metadata=result.details,
                )
            return False
        return True


class RepeatedActionGuard:
    """Detects identical consecutive actions with canonical argument normalization."""

    def __init__(
        self,
        max_repeated_actions: int = 3,
        raise_on_violation: bool = True,
    ) -> None:
        self.max_repeated_actions = max_repeated_actions
        self.raise_on_violation = raise_on_violation
        self.action_history: list[str] = []

    def evaluate_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        action_type: str = "TOOL_CALL",
    ) -> GuardResult:
        """Evaluate whether the given action repeats consecutively beyond threshold."""
        canonical_args = canonicalize_arguments(arguments)
        normalized_action_key = f"{action_type.upper()}:{tool_name.strip().lower()}:{canonical_args}"

        self.action_history.append(normalized_action_key)

        if len(self.action_history) < self.max_repeated_actions:
            return GuardResult(passed=True)

        recent = self.action_history[-self.max_repeated_actions:]
        if all(action == normalized_action_key for action in recent):
            msg = (
                f"Same action repeated {self.max_repeated_actions} consecutive times: "
                f"tool '{tool_name}' with arguments {arguments}"
            )
            return GuardResult(
                passed=False,
                violation_type=ErrorType.REPEATED_ACTION.value,
                message=msg,
                details={
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "consecutive_count": self.max_repeated_actions,
                },
            )

        return GuardResult(passed=True)

    def check_action(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Check action and raise if violation occurs (backward compatible)."""
        result = self.evaluate_action(tool_name, arguments)
        if not result.passed:
            if self.raise_on_violation:
                raise RepeatedActionError(
                    message=result.message or "Repeated action detected",
                    metadata=result.details,
                )
            return False
        return True


class LoopGuard:
    """Advanced loop detection supporting consecutive repeats and multi-step cycles (A-B-A-B)."""

    def __init__(
        self,
        max_repeated_actions: int = 3,
        history_window: int = 12,
        cycle_threshold: int = 2,
        raise_on_violation: bool = True,
    ) -> None:
        self.max_repeated_actions = max_repeated_actions
        self.history_window = history_window
        self.cycle_threshold = cycle_threshold
        self.raise_on_violation = raise_on_violation
        self.action_history: list[str] = []
        self.structured_history: list[dict[str, Any]] = []

    def evaluate_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        action_type: str = "TOOL_CALL",
    ) -> GuardResult:
        """Evaluate action for consecutive repetition or repeating cycle patterns."""
        canonical_args = canonicalize_arguments(arguments)
        normalized_key = f"{action_type.upper()}:{tool_name.strip().lower()}:{canonical_args}"

        self.action_history.append(normalized_key)
        self.structured_history.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "key": normalized_key,
        })

        # Keep history bounded by window
        if len(self.action_history) > self.history_window * 2:
            self.action_history = self.action_history[-self.history_window * 2:]
            self.structured_history = self.structured_history[-self.history_window * 2:]

        # 1. Check consecutive identical actions
        if len(self.action_history) >= self.max_repeated_actions:
            recent_consecutive = self.action_history[-self.max_repeated_actions:]
            if all(k == normalized_key for k in recent_consecutive):
                msg = f"Same action repeated {self.max_repeated_actions} times: {tool_name}"
                return GuardResult(
                    passed=False,
                    violation_type=ErrorType.REPEATED_ACTION.value,
                    message=msg,
                    details={
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "repeats": self.max_repeated_actions,
                    },
                )

        # 2. Check multi-step cycle patterns (period 2 up to period 4)
        history_slice = self.action_history[-self.history_window:]
        n = len(history_slice)
        for period in (2, 3, 4):
            required_len = period * self.cycle_threshold
            if n >= required_len:
                pattern = history_slice[-period:]
                # Check if the last (period * cycle_threshold) elements match repeated pattern
                is_cycle = True
                for i in range(1, self.cycle_threshold):
                    offset = -period * (i + 1)
                    sub = history_slice[offset : offset + period]
                    if sub != pattern:
                        is_cycle = False
                        break
                if is_cycle:
                    msg = (
                        f"Agent loop cycle of period {period} detected "
                        f"repeating {self.cycle_threshold} times"
                    )
                    return GuardResult(
                        passed=False,
                        violation_type=ErrorType.AGENT_LOOP_DETECTED.value,
                        message=msg,
                        details={
                            "period": period,
                            "cycle_threshold": self.cycle_threshold,
                            "pattern": [
                                self.structured_history[-period + i]["tool_name"]
                                for i in range(period)
                            ],
                        },
                    )

        return GuardResult(passed=True)

    def check_action(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Check action and raise error on violation (backward compatible)."""
        result = self.evaluate_action(tool_name, arguments)
        if not result.passed:
            if self.raise_on_violation:
                if result.violation_type == ErrorType.AGENT_LOOP_DETECTED.value:
                    raise AgentLoopDetectedError(
                        message=result.message or "Agent loop detected",
                        metadata=result.details,
                    )
                raise RepeatedActionError(
                    message=result.message or "Repeated action detected",
                    metadata=result.details,
                )
            return False
        return True


class PayloadGuard:
    """Guards payload sizes, preventing unbounded memory growth or API denial."""

    def __init__(
        self,
        max_observation_chars: int = 50_000,
        max_error_chars: int = 10_000,
        max_argument_chars: int = 20_000,
        max_trace_steps: int = 1_000,
    ) -> None:
        self.max_observation_chars = max_observation_chars
        self.max_error_chars = max_error_chars
        self.max_argument_chars = max_argument_chars
        self.max_trace_steps = max_trace_steps

    def check_trace_steps(self, total_steps: int) -> GuardResult:
        if total_steps >= self.max_trace_steps:
            return GuardResult(
                passed=False,
                violation_type="MAX_TRACE_STEPS_EXCEEDED",
                message=f"Trace step limit reached: {total_steps} >= {self.max_trace_steps}",
                details={"total_steps": total_steps, "max_trace_steps": self.max_trace_steps},
            )
        return GuardResult(passed=True)

    def sanitize_observation(self, obs: Any) -> Any:
        return self._truncate_if_needed(obs, self.max_observation_chars)

    def sanitize_arguments(self, args: Any) -> Any:
        return self._truncate_if_needed(args, self.max_argument_chars)

    def sanitize_error(self, err: Any) -> Any:
        return self._truncate_if_needed(err, self.max_error_chars)

    def _truncate_if_needed(self, val: Any, limit: int) -> Any:
        if isinstance(val, str) and len(val) > limit:
            return {
                "truncated": True,
                "original_length": len(val),
                "preview": val[:limit] + "... [TRUNCATED]",
            }
        if isinstance(val, dict):
            return {k: self._truncate_if_needed(v, limit) for k, v in val.items()}
        if isinstance(val, list):
            return [self._truncate_if_needed(item, limit) for item in val]
        return val


class GuardManager:
    """Unified guard coordinator for the AgentEngine."""

    def __init__(
        self,
        maximum_iterations: int = 10,
        max_repeated_actions: int = 3,
        history_window: int = 12,
        cycle_threshold: int = 2,
        max_observation_chars: int = 50_000,
        max_trace_steps: int = 1_000,
        raise_on_violation: bool = False,
    ) -> None:
        self.iteration_guard = IterationGuard(
            maximum_iterations=maximum_iterations,
            raise_on_violation=raise_on_violation,
        )
        self.loop_guard = LoopGuard(
            max_repeated_actions=max_repeated_actions,
            history_window=history_window,
            cycle_threshold=cycle_threshold,
            raise_on_violation=raise_on_violation,
        )
        self.payload_guard = PayloadGuard(
            max_observation_chars=max_observation_chars,
            max_trace_steps=max_trace_steps,
        )

    def check_iteration(self, current_iteration: int) -> GuardResult:
        """Check whether current iteration is within allowed bounds."""
        return self.iteration_guard.evaluate(current_iteration)

    def check_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        action_type: str = "TOOL_CALL",
    ) -> GuardResult:
        """Check whether action violates repetition or loop constraints."""
        return self.loop_guard.evaluate_action(tool_name, arguments, action_type)

    def check_trace_capacity(self, total_steps: int) -> GuardResult:
        """Check whether trace step capacity is within limits."""
        return self.payload_guard.check_trace_steps(total_steps)

    def sanitize_observation(self, observation: Any) -> Any:
        """Sanitize and truncate observation payload if necessary."""
        return self.payload_guard.sanitize_observation(observation)
