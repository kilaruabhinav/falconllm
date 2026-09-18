"""Backward-compatible SQLiteStorage adapter for existing codebase.

Delegates to SQLiteTraceStore in backend.core.persistence while preserving
the legacy save_run, save_step, get_run, and get_steps signatures.
"""

from __future__ import annotations

import json
from typing import Any

import sys
from pathlib import Path

# Ensure core package is resolvable regardless of invocation path
CORE_DIR = Path(__file__).resolve().parent.parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from persistence import SQLiteTraceStore
from protocols import RunRecord, RunUpdate
from trace import TraceStep


class SQLiteStorage:
    """Backward-compatible adapter providing legacy methods over SQLiteTraceStore."""

    def __init__(self, database_path: str = "agent.db") -> None:
        self.database_path = database_path
        self.store = SQLiteTraceStore(database_path=database_path)

    def create_tables(self) -> None:
        """Schema creation is handled automatically by SQLiteTraceStore."""
        self.store._init_schema()

    def connect(self):
        """Expose direct connection for legacy code if needed."""
        return self.store._get_connection()

    def save_run(
        self,
        run_id: str,
        user_query: str,
        started_at: str,
        completed_at: str | None,
        status: str,
        error: str | None = None,
        final_output: str | None = None,
    ) -> None:
        """Save or update an agent run (backward compatible)."""
        existing = self.store.get_run(run_id)
        if existing is None:
            self.store.create_run(
                RunRecord(
                    run_id=run_id,
                    user_query=user_query,
                    status=status,
                    created_at=started_at,
                    updated_at=completed_at or started_at,
                    completed_at=completed_at,
                    final_answer=final_output,
                    error_json=error,
                )
            )
        else:
            self.store.update_run(
                run_id=run_id,
                update=RunUpdate(
                    status=status,
                    updated_at=completed_at or started_at,
                    completed_at=completed_at,
                    final_answer=final_output,
                    error_json=error,
                ),
            )

    def save_step(
        self,
        run_id: str,
        step_number: int,
        step_type: str,
        description: str,
        data: dict[str, Any] | None,
        timestamp: str,
    ) -> None:
        """Save a step (backward compatible)."""
        meta = data or {}
        tool_name = meta.get("tool")
        arguments = meta.get("arguments")
        observation = meta.get("result")
        error = meta.get("error")

        self.store.append_step({
            "run_id": run_id,
            "sequence": step_number,
            "step_type": step_type,
            "status": "SUCCESS" if not error else "FAILED",
            "content": description,
            "tool_name": tool_name,
            "arguments": arguments,
            "observation": observation,
            "error": error,
            "timestamp": timestamp,
            "metadata": meta,
        })

    def get_run(self, run_id: str) -> tuple | None:
        """Fetch run returning legacy tuple representation."""
        record = self.store.get_run(run_id)
        if not record:
            return None
        return (
            record.run_id,
            record.user_query,
            record.created_at,
            record.completed_at,
            record.status,
            record.error_json,
            record.final_answer,
        )

    def get_steps(self, run_id: str) -> list[tuple]:
        """Fetch steps returning legacy list of tuples."""
        steps = self.store.get_steps(run_id)
        return [
            (
                s.sequence,
                s.step_type,
                s.content,
                json.dumps(s.metadata),
                s.created_at.isoformat(),
            )
            for s in steps
        ]

    def close(self) -> None:
        self.store.close()