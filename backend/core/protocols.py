"""Core protocols and abstract interfaces for the AI Agent Framework.

Defines structural interfaces (PEP 544 Protocols) for persistence, tools, guards,
and time abstractions to decouple the AgentEngine from concrete implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class RunRecord:
    """Represents a persisted agent execution run."""

    run_id: str
    user_query: str
    status: str
    created_at: str
    updated_at: str
    completed_at: str | None = None
    final_answer: str | None = None
    metadata_json: str | None = None
    error_json: str | None = None


@dataclass
class RunUpdate:
    """Represents updates applied to an existing agent run."""

    status: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    final_answer: str | None = None
    metadata_json: str | None = None
    error_json: str | None = None


@runtime_checkable
class TraceStore(Protocol):
    """Abstract interface for execution trace persistence."""

    def create_run(self, run: RunRecord) -> None:
        """Create a new run record in persistence."""
        ...

    def append_step(self, step: Any) -> None:
        """Append a trace step to persistence."""
        ...

    def update_run(self, run_id: str, update: RunUpdate) -> None:
        """Update an existing run record in persistence."""
        ...

    def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a run record by run_id."""
        ...

    def get_steps(self, run_id: str) -> list[Any]:
        """Retrieve all ordered trace steps for a run_id."""
        ...

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        """Retrieve recent runs in reverse chronological order."""
        ...

    def close(self) -> None:
        """Close persistence resources."""
        ...


@runtime_checkable
class ToolProtocol(Protocol):
    """Abstract protocol for tools executable by the agent."""

    name: str
    description: str

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given arguments."""
        ...


@runtime_checkable
class ClockProtocol(Protocol):
    """Abstract clock for deterministic time injection."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC datetime."""
        ...


@runtime_checkable
class SleeperProtocol(Protocol):
    """Abstract sleeper for deterministic sleep injection."""

    def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration."""
        ...
