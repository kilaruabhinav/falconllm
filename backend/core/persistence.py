"""SQLite persistence implementation for execution traces and agent runs.

Fulfills the TraceStore protocol using SQLite with:
- WAL journal mode
- Foreign key constraint enforcement
- Busy timeout concurrency protection
- Explicit transactional safety
- Monotonic sequence uniqueness and indexing
"""

from __future__ import annotations

from datetime import datetime, UTC
import json
import logging
import sqlite3
import threading
from typing import Any
import uuid

from .errors import PersistenceError
from .protocols import RunRecord, RunUpdate, TraceStore
from .trace import TraceStep, ensure_utc

logger = logging.getLogger("ai_agent.persistence")


def safe_json_dumps(obj: Any) -> str | None:
    """Serialize object to JSON string, returning None if obj is None."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    return json.dumps(obj)


def safe_json_loads(val: str | None) -> Any:
    """Deserialize JSON string, returning None if val is None or empty."""
    if not val:
        return None
    try:
        return json.loads(val)
    except Exception:
        return val


class SQLiteTraceStore(TraceStore):
    """Concurrency-safe, SQLite-backed trace and run store."""

    def __init__(self, database_path: str = "agent.db") -> None:
        self.database_path = database_path
        self._local = threading.local()
        self._all_connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or initialize a thread-local SQLite connection with optimal pragmas."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(
                self.database_path,
                timeout=5.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row

            # Apply performance and concurrency pragmas
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            if self.database_path != ":memory:":
                try:
                    conn.execute("PRAGMA journal_mode = WAL;")
                    conn.execute("PRAGMA synchronous = NORMAL;")
                except sqlite3.OperationalError:
                    pass  # In-memory or restricted filesystems

            self._local.connection = conn
            with self._lock:
                self._all_connections.append(conn)

        return self._local.connection

    def _init_schema(self) -> None:
        """Initialize database schema matching hackathon requirements."""
        conn = self._get_connection()
        try:
            with conn:
                # 1. Runs table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        user_query TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT,
                        final_answer TEXT,
                        metadata_json TEXT,
                        error_json TEXT
                    );
                """)

                # Auto-migrate legacy columns if an older runs table already existed
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(runs)")
                existing_cols = {row[1] for row in cur.fetchall()}

                required_cols = {
                    "status": "TEXT NOT NULL DEFAULT 'RUNNING'",
                    "created_at": "TEXT NOT NULL DEFAULT ''",
                    "updated_at": "TEXT NOT NULL DEFAULT ''",
                    "completed_at": "TEXT",
                    "final_answer": "TEXT",
                    "metadata_json": "TEXT",
                    "error_json": "TEXT",
                }
                for col_name, col_def in required_cols.items():
                    if col_name not in existing_cols:
                        cur.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_def};")

                # 2. Trace steps table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trace_steps (
                        step_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        step_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        duration_ms REAL,
                        content TEXT,
                        tool_name TEXT,
                        arguments_json TEXT,
                        observation_json TEXT,
                        error_json TEXT,
                        metadata_json TEXT,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                        UNIQUE (run_id, sequence)
                    );
                """)

                # 3. Fast sequence index
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trace_steps_run_sequence
                    ON trace_steps(run_id, sequence);
                """)
        except Exception as e:
            logger.error("Failed to initialize database schema: %s", e)
            raise PersistenceError(f"Schema initialization failed: {e}", original_exception=e)

    def create_run(self, run: RunRecord) -> None:
        """Create a new run record in SQLite."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, user_query, status, created_at, updated_at,
                        completed_at, final_answer, metadata_json, error_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        run.user_query,
                        run.status,
                        run.created_at,
                        run.updated_at,
                        run.completed_at,
                        run.final_answer,
                        safe_json_dumps(run.metadata_json),
                        safe_json_dumps(run.error_json),
                    ),
                )
        except Exception as e:
            raise PersistenceError(f"Failed to create run {run.run_id}: {e}", original_exception=e)

    def append_step(self, step: TraceStep | dict[str, Any]) -> None:
        """Append a trace step to SQLite."""
        conn = self._get_connection()

        if isinstance(step, TraceStep):
            step_id = step.step_id
            run_id = step.run_id
            sequence = step.sequence
            step_type = step.step_type
            status = step.status
            created_at = step.created_at.isoformat()
            duration_ms = step.duration_ms
            content = step.content
            tool_name = step.tool_name
            arguments_json = safe_json_dumps(step.arguments)
            observation_json = safe_json_dumps(step.observation)
            error_json = safe_json_dumps(step.error)
            metadata_json = safe_json_dumps(step.metadata)
        else:
            step_id = step.get("step_id", str(uuid.uuid4()))
            run_id = step["run_id"]
            sequence = step.get("sequence", step.get("step_number", 1))
            step_type = step["step_type"]
            status = step.get("status", "SUCCESS")
            created_at = step.get("created_at", step.get("timestamp", datetime.now(UTC).isoformat()))
            duration_ms = step.get("duration_ms")
            content = step.get("content", step.get("description", ""))
            tool_name = step.get("tool_name")
            arguments_json = safe_json_dumps(step.get("arguments"))
            observation_json = safe_json_dumps(step.get("observation"))
            error_json = safe_json_dumps(step.get("error"))
            metadata_json = safe_json_dumps(step.get("metadata", step.get("data")))

        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO trace_steps (
                        step_id, run_id, sequence, step_type, status,
                        created_at, duration_ms, content, tool_name,
                        arguments_json, observation_json, error_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        step_id,
                        run_id,
                        sequence,
                        step_type,
                        status,
                        created_at,
                        duration_ms,
                        content,
                        tool_name,
                        arguments_json,
                        observation_json,
                        error_json,
                        metadata_json,
                    ),
                )
        except sqlite3.IntegrityError as e:
            raise PersistenceError(
                f"Duplicate step sequence {sequence} for run {run_id}: {e}",
                original_exception=e,
            )
        except Exception as e:
            raise PersistenceError(f"Failed to append step to run {run_id}: {e}", original_exception=e)

    def update_run(self, run_id: str, update: RunUpdate) -> None:
        """Update an existing run record."""
        conn = self._get_connection()
        fields: list[str] = []
        params: list[Any] = []

        if update.status is not None:
            fields.append("status = ?")
            params.append(update.status)
        if update.updated_at is not None:
            fields.append("updated_at = ?")
            params.append(update.updated_at)
        if update.completed_at is not None:
            fields.append("completed_at = ?")
            params.append(update.completed_at)
        if update.final_answer is not None:
            fields.append("final_answer = ?")
            params.append(update.final_answer)
        if update.metadata_json is not None:
            fields.append("metadata_json = ?")
            params.append(safe_json_dumps(update.metadata_json))
        if update.error_json is not None:
            fields.append("error_json = ?")
            params.append(safe_json_dumps(update.error_json))

        if not fields:
            return

        params.append(run_id)
        query = f"UPDATE runs SET {', '.join(fields)} WHERE run_id = ?"

        try:
            with conn:
                conn.execute(query, tuple(params))
        except Exception as e:
            raise PersistenceError(f"Failed to update run {run_id}: {e}", original_exception=e)

    def get_run(self, run_id: str) -> RunRecord | None:
        """Fetch a run by run_id."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT run_id, user_query, status, created_at, updated_at,
                   completed_at, final_answer, metadata_json, error_json
            FROM runs WHERE run_id = ?
            """,
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        return RunRecord(
            run_id=row["run_id"],
            user_query=row["user_query"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            final_answer=row["final_answer"],
            metadata_json=row["metadata_json"],
            error_json=row["error_json"],
        )

    def get_steps(self, run_id: str) -> list[TraceStep]:
        """Retrieve ordered trace steps for a given run."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT step_id, run_id, sequence, step_type, status, created_at,
                   duration_ms, content, tool_name, arguments_json,
                   observation_json, error_json, metadata_json
            FROM trace_steps
            WHERE run_id = ?
            ORDER BY sequence ASC
            """,
            (run_id,),
        )
        rows = cur.fetchall()
        steps: list[TraceStep] = []
        for r in rows:
            step = TraceStep(
                step_id=r["step_id"],
                run_id=r["run_id"],
                sequence=r["sequence"],
                step_type=r["step_type"],
                status=r["status"],
                created_at=ensure_utc(r["created_at"]),
                duration_ms=r["duration_ms"],
                content=r["content"] or "",
                tool_name=r["tool_name"],
                arguments=safe_json_loads(r["arguments_json"]),
                observation=safe_json_loads(r["observation_json"]),
                error=safe_json_loads(r["error_json"]),
                metadata=safe_json_loads(r["metadata_json"]) or {},
            )
            steps.append(step)
        return steps

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        """Return recent runs without loading their trace payloads."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("limit must be an integer from 1 to 200")
        rows = self._get_connection().execute(
            """
            SELECT run_id, user_query, status, created_at, updated_at,
                   completed_at, final_answer, metadata_json, error_json
            FROM runs ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [RunRecord(**dict(row)) for row in rows]

    def close(self) -> None:
        """Close all cached connections."""
        with self._lock:
            for conn in self._all_connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_connections.clear()
        if hasattr(self._local, "connection"):
            self._local.connection = None
