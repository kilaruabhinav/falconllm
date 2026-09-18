"""Execution trace recording and step management for the AI Agent Framework.

Provides thread-safe, monotonically ordered, and auditable trace logging
with automatic credential redaction, sequence protection, and frontend-friendly serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
import json
import logging
import re
import threading
import time
from typing import Any
import uuid

logger = logging.getLogger("ai_agent.trace")

# Pattern for sensitive credentials, API keys, passwords, and tokens
SENSITIVE_KEY_RE = re.compile(
    r"(^|_)(api[_-]?key|secret|password|bearer|credential|private[_-]?key)($|_)|(^|_)(access[_-]?token|refresh[_-]?token|auth[_-]?token|secret[_-]?token|jwt|session[_-]?token)($|_)|^token$|^authorization$",
    re.IGNORECASE,
)



def current_utc_time() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(dt: datetime | str | None) -> datetime:
    """Ensure datetime is timezone-aware UTC."""
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


def redact_sensitive(val: Any) -> Any:
    """Recursively scrub sensitive keys and token values."""
    if isinstance(val, dict):
        cleaned: dict[str, Any] = {}
        for k, v in val.items():
            if SENSITIVE_KEY_RE.search(str(k)):
                cleaned[str(k)] = "[REDACTED]"
            else:
                cleaned[str(k)] = redact_sensitive(v)
        return cleaned
    if isinstance(val, list):
        return [redact_sensitive(item) for item in val]
    if isinstance(val, tuple):
        return tuple(redact_sensitive(item) for item in val)
    return val


def truncate_value(val: Any, max_chars: int = 50_000) -> Any:
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
        return {k: truncate_value(v, max_chars) for k, v in val.items()}
    if isinstance(val, list):
        return [truncate_value(item, max_chars) for item in val]
    return val


def safe_json_serialize(obj: Any) -> Any:
    """Fallback handler for JSON serialization."""
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


class TraceStepType(str, Enum):
    """Canonical trace step categories."""

    RUN_STARTED = "RUN_STARTED"
    PLAN = "PLAN"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_ERROR = "TOOL_ERROR"
    PARSE_ERROR = "PARSE_ERROR"
    LLM_ERROR = "LLM_ERROR"
    LLM_PROVIDER_SELECTED = "LLM_PROVIDER_SELECTED"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_FALLBACK = "LLM_FALLBACK"
    FINAL = "FINAL"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"
    RECOVERY = "RECOVERY"
    ERROR = "ERROR"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"
    RUN_CANCELLED = "RUN_CANCELLED"

    # Backward compatibility mappings
    START = "RUN_STARTED"
    ACT = "ACTION"
    OBSERVE = "OBSERVATION"


def normalize_step_type(step_type: TraceStepType | str) -> str:
    """Normalize step type string or Enum to canonical format."""
    if isinstance(step_type, TraceStepType):
        return step_type.value
    st_upper = str(step_type).upper()
    if st_upper in TraceStepType.__members__:
        return TraceStepType[st_upper].value
    # Direct mapping checks
    if st_upper == "START":
        return TraceStepType.RUN_STARTED.value
    if st_upper == "ACT":
        return TraceStepType.ACTION.value
    if st_upper == "OBSERVE":
        return TraceStepType.OBSERVATION.value
    if st_upper == "FINAL":
        return TraceStepType.RUN_COMPLETED.value
    return st_upper


@dataclass
class TraceStep:
    """Represents an atomic, immutable step in the agent execution trace."""

    run_id: str
    sequence: int
    step_type: str
    status: str
    created_at: datetime
    content: str
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    observation: Any = None
    error: Any = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.created_at = ensure_utc(self.created_at)
        self.step_type = normalize_step_type(self.step_type)
        self.arguments = redact_sensitive(self.arguments) if self.arguments else None
        self.observation = redact_sensitive(self.observation) if self.observation else None
        self.metadata = redact_sensitive(self.metadata) if self.metadata else {}

    # Backward compatibility properties for existing engine.py & database.py
    @property
    def step_number(self) -> int:
        return self.sequence

    @property
    def step(self) -> int:
        return self.sequence

    @property
    def type(self) -> str:
        return self.step_type.lower()

    @property
    def tool(self) -> str | None:
        return self.tool_name

    @property
    def result(self) -> Any:
        return self.observation

    @property
    def description(self) -> str:
        return self.content

    @property
    def timestamp(self) -> datetime:
        return self.created_at

    @property
    def data(self) -> dict[str, Any]:
        """Aggregate data dictionary for legacy access."""
        res: dict[str, Any] = dict(self.metadata)
        if self.tool_name:
            res["tool"] = self.tool_name
        if self.arguments:
            res["arguments"] = self.arguments
        if self.observation is not None:
            if isinstance(self.observation, dict):
                res.update(self.observation)
            else:
                res["result"] = self.observation
        if self.error:
            if isinstance(self.error, dict):
                res["error"] = self.error
            else:
                res["error"] = str(self.error)
        return res

    def to_dict(self, max_chars: int = 50_000) -> dict[str, Any]:
        """Convert trace step into a sanitized dictionary."""
        payload: dict[str, Any] = {
            "step_id": self.step_id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "step_number": self.sequence,  # legacy support
            "step_type": self.step_type,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "timestamp": self.created_at.isoformat(),  # legacy support
            "content": self.content,
            "description": self.content,  # legacy support
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "observation": self.observation,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "data": self.data,  # legacy support
        }
        return truncate_value(redact_sensitive(payload), max_chars=max_chars)


class ExecutionTrace:
    """Thread-safe, monotonically sequenced execution trace for an agent run."""

    def __init__(
        self,
        run_id: str | None = None,
        started_at: datetime | None = None,
        status: str = "RUNNING",
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.started_at = ensure_utc(started_at)
        self.completed_at: datetime | None = None
        self.status = status
        self.error: str | None = None
        self.final_output: str | None = None
        self.steps: list[TraceStep] = []
        self._lock = threading.Lock()
        self._sequences: set[int] = set()

    def add_step(
        self,
        step_type: str | TraceStepType,
        description: str | None = None,
        content: str | None = None,
        status: str = "SUCCESS",
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        observation: Any = None,
        error: Any = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        sequence: int | None = None,
    ) -> TraceStep:
        """Atomically append a step with duplicate sequence protection."""
        text_content = content or description or ""
        meta = dict(metadata or {})
        if data:
            meta.update(data)
            if "tool" in data and not tool_name:
                tool_name = data["tool"]
            if "arguments" in data and not arguments:
                arguments = data["arguments"]
            if "result" in data and observation is None:
                observation = data["result"]

        with self._lock:
            if sequence is None:
                seq = len(self.steps) + 1
            else:
                seq = sequence

            if seq in self._sequences:
                raise ValueError(f"Duplicate sequence number {seq} in trace for run {self.run_id}")

            step = TraceStep(
                run_id=self.run_id,
                sequence=seq,
                step_type=normalize_step_type(step_type),
                status=status,
                created_at=current_utc_time(),
                content=text_content,
                tool_name=tool_name,
                arguments=arguments,
                observation=observation,
                error=error,
                duration_ms=duration_ms,
                metadata=meta,
            )
            self.steps.append(step)
            self._sequences.add(seq)
            return step

    def add_plan(
        self,
        description: str = "Agent created a plan",
        data: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> TraceStep:
        return self.add_step(
            step_type=TraceStepType.PLAN,
            content=content or description,
            data=data,
            status="SUCCESS",
        )

    def add_action(
        self,
        description: str = "Agent selected tool",
        data: dict[str, Any] | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> TraceStep:
        return self.add_step(
            step_type=TraceStepType.ACTION,
            content=content or description,
            tool_name=tool_name,
            arguments=arguments,
            data=data,
            status="STARTED",
        )

    def add_observation(
        self,
        description: str = "Agent received tool result",
        data: dict[str, Any] | None = None,
        observation: Any = None,
        content: str | None = None,
    ) -> TraceStep:
        return self.add_step(
            step_type=TraceStepType.OBSERVATION,
            content=content or description,
            observation=observation,
            data=data,
            status="SUCCESS",
        )

    def add_error(
        self,
        description: str = "Agent encountered an error",
        data: dict[str, Any] | None = None,
        error: Any = None,
        content: str | None = None,
    ) -> TraceStep:
        return self.add_step(
            step_type=TraceStepType.ERROR,
            content=content or description,
            error=error or (data.get("message") if data else None),
            data=data,
            status="FAILED",
        )

    def add_recovery(
        self,
        description: str = "Recovery decision created",
        data: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> TraceStep:
        return self.add_step(
            step_type=TraceStepType.RECOVERY,
            content=content or description,
            data=data,
            status="SUCCESS",
        )

    def complete(
        self,
        final_output: str,
        *,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceStep:
        """Mark execution as completed with final output."""
        with self._lock:
            self.status = "COMPLETED"
            self.final_output = final_output
            self.completed_at = current_utc_time()

        return self.add_step(
            step_type=TraceStepType.RUN_COMPLETED,
            content=f"Agent completed execution with final answer",
            data={"final_answer": final_output, **(metadata or {})},
            duration_ms=duration_ms,
            status="SUCCESS",
        )

    def fail(self, error_message: str) -> TraceStep:
        """Mark execution as failed with error message."""
        with self._lock:
            self.status = "FAILED"
            self.error = error_message
            self.completed_at = current_utc_time()

        return self.add_step(
            step_type=TraceStepType.RUN_FAILED,
            content=f"Agent execution failed: {error_message}",
            error=error_message,
            status="FAILED",
        )

    def cancel(self, reason: str = "Cancelled by user") -> TraceStep:
        """Mark execution as cancelled."""
        with self._lock:
            self.status = "CANCELLED"
            self.completed_at = current_utc_time()

        return self.add_step(
            step_type=TraceStepType.RUN_CANCELLED,
            content=f"Agent execution cancelled: {reason}",
            status="CANCELLED",
        )

    def to_dict(self, max_payload_chars: int = 50_000) -> dict[str, Any]:
        """Serialize trace to frontend-friendly dictionary."""
        with self._lock:
            steps_dict = [s.to_dict(max_chars=max_payload_chars) for s in self.steps]
            data: dict[str, Any] = {
                "run_id": self.run_id,
                "status": self.status,
                "started_at": self.started_at.isoformat(),
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "error": self.error,
                "final_output": self.final_output,
                "total_steps": len(self.steps),
                "steps": steps_dict,
            }
            return truncate_value(redact_sensitive(data), max_chars=max_payload_chars)

    def to_json(self, indent: int | None = None, max_payload_chars: int = 50_000) -> str:
        """Serialize trace to JSON string."""
        return json.dumps(
            self.to_dict(max_payload_chars=max_payload_chars),
            default=safe_json_serialize,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionTrace:
        """Reconstruct an ExecutionTrace instance from a dictionary."""
        trace = cls(
            run_id=data["run_id"],
            started_at=ensure_utc(data.get("started_at")),
            status=data.get("status", "RUNNING"),
        )
        trace.error = data.get("error")
        trace.final_output = data.get("final_output")
        if data.get("completed_at"):
            trace.completed_at = ensure_utc(data["completed_at"])

        for s in data.get("steps", []):
            step = TraceStep(
                step_id=s.get("step_id", str(uuid.uuid4())),
                run_id=data["run_id"],
                sequence=s.get("sequence", s.get("step_number", 1)),
                step_type=s["step_type"],
                status=s.get("status", "SUCCESS"),
                created_at=ensure_utc(s.get("created_at", s.get("timestamp"))),
                content=s.get("content", s.get("description", "")),
                tool_name=s.get("tool_name"),
                arguments=s.get("arguments"),
                observation=s.get("observation"),
                error=s.get("error"),
                duration_ms=s.get("duration_ms"),
                metadata=s.get("metadata", {}),
            )
            trace.steps.append(step)
            trace._sequences.add(step.sequence)

        return trace


class TraceManager:
    """High-level trace manager coordinating execution tracing and optional persistence."""

    def __init__(self, store: Any = None, publisher: Any = None) -> None:
        self.store = store
        self.publisher = publisher
        self.active_traces: dict[str, ExecutionTrace] = {}
        self._lock = threading.Lock()
        self._persistence_durations_ms: dict[str, list[float]] = {}

    def start_run(
        self,
        state_or_query: Any,
        run_id: str | None = None,
    ) -> ExecutionTrace:
        """Initialize and register a new execution trace for a run."""
        if hasattr(state_or_query, "run_id"):
            rid = state_or_query.run_id
            user_query = getattr(state_or_query, "user_query", "")
        else:
            rid = run_id or str(uuid.uuid4())
            user_query = str(state_or_query)

        trace = ExecutionTrace(run_id=rid)
        trace.add_step(
            step_type=TraceStepType.RUN_STARTED,
            content="Agent execution started",
            data={"user_query": user_query},
            status="SUCCESS",
        )

        with self._lock:
            self.active_traces[rid] = trace

        self._publish(trace.steps[0])

        # Persist if store attached
        if self.store is not None:
            started = time.perf_counter()
            try:
                from .protocols import RunRecord
                rec = RunRecord(
                    run_id=rid,
                    user_query=user_query,
                    status="RUNNING",
                    created_at=trace.started_at.isoformat(),
                    updated_at=trace.started_at.isoformat(),
                )
                self.store.create_run(rec)
                self.store.append_step(trace.steps[0])
            except Exception as e:
                logger.warning("TraceStore persist error in start_run: %s", e)
            finally:
                self._track_persistence(rid, started)

        return trace

    def record_plan(
        self,
        run_id: str,
        content: str,
        data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> TraceStep:
        """Record a planning step."""
        trace = self._get_or_create(run_id)
        step = trace.add_step(
            step_type=TraceStepType.PLAN,
            content=content,
            data=data,
            duration_ms=duration_ms,
            status="SUCCESS",
        )
        self._publish(step)
        self._persist_step(step)
        return step

    def record_step(
        self,
        run_id: str,
        step_type: str | TraceStepType,
        *,
        content: str = "",
        status: str = "SUCCESS",
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        observation: Any = None,
        error: Any = None,
        metadata: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> TraceStep:
        """Record and persist an arbitrary canonical lifecycle event."""
        trace = self._get_or_create(run_id)
        step = trace.add_step(
            step_type=step_type,
            content=content,
            status=status,
            tool_name=tool_name,
            arguments=arguments,
            observation=observation,
            error=error,
            metadata=metadata,
            duration_ms=duration_ms,
        )
        self._publish(step)
        self._persist_step(step)
        return step

    def record_action(
        self,
        run_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        content: str | None = None,
    ) -> TraceStep:
        """Record an action step."""
        trace = self._get_or_create(run_id)
        step = trace.add_action(
            tool_name=tool_name,
            arguments=arguments,
            content=content or f"Agent selected tool '{tool_name}'",
        )
        self._publish(step)
        self._persist_step(step)
        return step

    def record_observation(
        self,
        run_id: str,
        observation: Any,
        content: str | None = None,
    ) -> TraceStep:
        """Record an observation step."""
        trace = self._get_or_create(run_id)
        step = trace.add_observation(
            observation=observation,
            content=content or "Agent received observation",
        )
        self._publish(step)
        self._persist_step(step)
        return step

    def record_error(
        self,
        run_id: str,
        error: Any,
        content: str | None = None,
    ) -> TraceStep:
        """Record an error step."""
        trace = self._get_or_create(run_id)
        err_msg = error.message if hasattr(error, "message") else str(error)
        err_data = error.to_dict() if hasattr(error, "to_dict") else {"error": err_msg}
        step = trace.add_error(
            error=err_msg,
            data=err_data,
            content=content or f"Agent error: {err_msg}",
        )
        self._publish(step)
        self._persist_step(step)
        return step

    def record_recovery(
        self,
        run_id: str,
        recovery_decision: Any,
        content: str | None = None,
    ) -> TraceStep:
        """Record a recovery decision step."""
        trace = self._get_or_create(run_id)
        data = recovery_decision.to_dict() if hasattr(recovery_decision, "to_dict") else {"decision": str(recovery_decision)}
        step = trace.add_recovery(
            data=data,
            content=content or "Recovery decision generated",
        )
        self._publish(step)
        self._persist_step(step)
        return step

    def complete_run(
        self, run_id: str, final_output: str, duration_ms: float | None = None
    ) -> TraceStep:
        """Complete an execution run."""
        trace = self._get_or_create(run_id)
        timings = self._persistence_durations_ms.get(run_id, [])
        persistence = {
            "database_writes": len(timings),
            "database_total_ms": round(sum(timings), 3),
            "database_average_ms": round(sum(timings) / len(timings), 3) if timings else 0,
        }
        step = trace.complete(
            final_output=final_output,
            duration_ms=duration_ms,
            metadata={"timing": persistence},
        )
        self._publish(step)
        self._persist_step(step)

        if self.store is not None:
            try:
                from .protocols import RunUpdate
                self.store.update_run(
                    run_id=run_id,
                    update=RunUpdate(
                        status="COMPLETED",
                        updated_at=trace.completed_at.isoformat() if trace.completed_at else None,
                        completed_at=trace.completed_at.isoformat() if trace.completed_at else None,
                        final_answer=final_output,
                    ),
                )
            except Exception as e:
                logger.warning("TraceStore complete_run error: %s", e)

        return step

    def fail_run(self, run_id: str, error_message: str) -> TraceStep:
        """Fail an execution run."""
        trace = self._get_or_create(run_id)
        step = trace.fail(error_message=error_message)
        self._publish(step)
        self._persist_step(step)

        if self.store is not None:
            try:
                from .protocols import RunUpdate
                self.store.update_run(
                    run_id=run_id,
                    update=RunUpdate(
                        status="FAILED",
                        updated_at=trace.completed_at.isoformat() if trace.completed_at else None,
                        completed_at=trace.completed_at.isoformat() if trace.completed_at else None,
                        error_json=error_message,
                    ),
                )
            except Exception as e:
                logger.warning("TraceStore fail_run error: %s", e)

        return step

    def cancel_run(self, run_id: str, reason: str = "Cancelled by user") -> TraceStep:
        """Cancel an execution run and publish/persist its terminal event."""
        trace = self._get_or_create(run_id)
        step = trace.cancel(reason)
        self._publish(step)
        self._persist_step(step)
        return step

    def get_trace(self, run_id: str) -> ExecutionTrace | None:
        """Get the active or persisted execution trace."""
        with self._lock:
            if run_id in self.active_traces:
                return self.active_traces[run_id]

        if self.store is not None:
            run_rec = self.store.get_run(run_id)
            if run_rec:
                steps = self.store.get_steps(run_id)
                trace = ExecutionTrace(
                    run_id=run_id,
                    started_at=ensure_utc(run_rec.created_at),
                    status=run_rec.status,
                )
                trace.final_output = run_rec.final_answer
                trace.error = run_rec.error_json
                if run_rec.completed_at:
                    trace.completed_at = ensure_utc(run_rec.completed_at)
                for st in steps:
                    trace.steps.append(st)
                    trace._sequences.add(st.sequence)
                return trace

        return None

    def _get_or_create(self, run_id: str) -> ExecutionTrace:
        with self._lock:
            if run_id not in self.active_traces:
                self.active_traces[run_id] = ExecutionTrace(run_id=run_id)
            return self.active_traces[run_id]

    def _persist_step(self, step: TraceStep) -> None:
        if self.store is not None:
            started = time.perf_counter()
            try:
                self.store.append_step(step)
            except Exception as e:
                logger.warning("Failed to persist step %s: %s", step.step_id, e)
            finally:
                self._track_persistence(step.run_id, started)

    def _publish(self, step: TraceStep) -> None:
        if self.publisher is None:
            return
        try:
            self.publisher(step)
        except Exception as exc:
            logger.warning("Live trace publish failed for %s: %s", step.step_id, exc)

    def _track_persistence(self, run_id: str, started: float) -> None:
        duration = (time.perf_counter() - started) * 1000
        with self._lock:
            self._persistence_durations_ms.setdefault(run_id, []).append(duration)
