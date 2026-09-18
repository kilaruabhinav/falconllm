"""Unit tests for SQLiteTraceStore and Persistence Layer."""

import concurrent.futures
from datetime import datetime, UTC
import os
import tempfile
import unittest

from backend.core.errors import PersistenceError
from backend.core.persistence import SQLiteTraceStore
from backend.core.protocols import RunRecord, RunUpdate
from backend.core.trace import TraceStep, ExecutionTrace


class TestPersistence(unittest.TestCase):
    """Test suite covering SQLite schema, transactions, foreign keys, sequence uniqueness, and persistence across restarts."""

    def setUp(self) -> None:
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)
        self.store = SQLiteTraceStore(database_path=self.temp_db_path)

    def tearDown(self) -> None:
        self.store.close()
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except Exception:
                pass

    def test_schema_creation(self) -> None:
        conn = self.store._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        self.assertIn("runs", tables)
        self.assertIn("trace_steps", tables)

    def test_create_and_get_run(self) -> None:
        now_str = datetime.now(UTC).isoformat()
        rec = RunRecord(
            run_id="run-test-01",
            user_query="How does SQLite persistence work?",
            status="RUNNING",
            created_at=now_str,
            updated_at=now_str,
            metadata_json='{"env": "test"}',
        )
        self.store.create_run(rec)

        fetched = self.store.get_run("run-test-01")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.run_id, "run-test-01")
        self.assertEqual(fetched.user_query, "How does SQLite persistence work?")
        self.assertEqual(fetched.status, "RUNNING")

        # Nonexistent run returns None
        self.assertIsNone(self.store.get_run("nonexistent-id"))

    def test_step_append_and_ordered_retrieval(self) -> None:
        now_str = datetime.now(UTC).isoformat()
        self.store.create_run(
            RunRecord(
                run_id="run-test-02",
                user_query="Query",
                status="RUNNING",
                created_at=now_str,
                updated_at=now_str,
            )
        )

        trace = ExecutionTrace(run_id="run-test-02")
        s1 = trace.add_step("RUN_STARTED", content="Started")
        s2 = trace.add_plan(content="Planning")
        s3 = trace.add_action(tool_name="search", arguments={"q": "rag"})

        self.store.append_step(s1)
        self.store.append_step(s2)
        self.store.append_step(s3)

        steps = self.store.get_steps("run-test-02")
        self.assertEqual(len(steps), 3)
        self.assertEqual([s.sequence for s in steps], [1, 2, 3])
        self.assertEqual(steps[0].content, "Started")
        self.assertEqual(steps[1].step_type, "PLAN")
        self.assertEqual(steps[2].tool_name, "search")
        self.assertEqual(steps[2].arguments, {"q": "rag"})

    def test_update_run(self) -> None:
        now_str = datetime.now(UTC).isoformat()
        self.store.create_run(
            RunRecord(
                run_id="run-test-03",
                user_query="Query",
                status="RUNNING",
                created_at=now_str,
                updated_at=now_str,
            )
        )

        completed_str = datetime.now(UTC).isoformat()
        self.store.update_run(
            "run-test-03",
            RunUpdate(
                status="COMPLETED",
                completed_at=completed_str,
                final_answer="Final answer text",
            ),
        )

        updated = self.store.get_run("run-test-03")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, "COMPLETED")
        self.assertEqual(updated.final_answer, "Final answer text")
        self.assertEqual(updated.completed_at, completed_str)

    def test_foreign_key_enforcement(self) -> None:
        # Appending step with non-existent run_id must raise PersistenceError
        step = TraceStep(
            run_id="non-existent-run",
            sequence=1,
            step_type="PLAN",
            status="SUCCESS",
            created_at=datetime.now(UTC),
            content="Plan without run",
        )
        with self.assertRaises(PersistenceError):
            self.store.append_step(step)

    def test_duplicate_sequence_rejection(self) -> None:
        now_str = datetime.now(UTC).isoformat()
        self.store.create_run(
            RunRecord(
                run_id="run-test-04",
                user_query="Query",
                status="RUNNING",
                created_at=now_str,
                updated_at=now_str,
            )
        )

        s1 = TraceStep(
            run_id="run-test-04",
            sequence=1,
            step_type="PLAN",
            status="SUCCESS",
            created_at=datetime.now(UTC),
            content="Step 1",
        )
        s2 = TraceStep(
            run_id="run-test-04",
            sequence=1,  # Duplicate sequence!
            step_type="ACTION",
            status="STARTED",
            created_at=datetime.now(UTC),
            content="Step 1 duplicate",
        )

        self.store.append_step(s1)
        with self.assertRaises(PersistenceError):
            self.store.append_step(s2)

    def test_database_restart_and_persistence(self) -> None:
        # 1. Write run and steps
        now_str = datetime.now(UTC).isoformat()
        self.store.create_run(
            RunRecord(
                run_id="persist-run-01",
                user_query="Query for restart test",
                status="COMPLETED",
                created_at=now_str,
                updated_at=now_str,
                final_answer="Restart persisted answer",
            )
        )
        s1 = TraceStep(
            run_id="persist-run-01",
            sequence=1,
            step_type="RUN_STARTED",
            status="SUCCESS",
            created_at=datetime.now(UTC),
            content="Persisted step 1",
        )
        self.store.append_step(s1)

        # 2. Close store
        self.store.close()

        # 3. Re-open against same file
        reopened_store = SQLiteTraceStore(database_path=self.temp_db_path)
        try:
            run = reopened_store.get_run("persist-run-01")
            self.assertIsNotNone(run)
            self.assertEqual(run.final_answer, "Restart persisted answer")

            steps = reopened_store.get_steps("persist-run-01")
            self.assertEqual(len(steps), 1)
            self.assertEqual(steps[0].content, "Persisted step 1")
        finally:
            reopened_store.close()

    def test_concurrent_writes(self) -> None:
        now_str = datetime.now(UTC).isoformat()
        num_runs = 10

        def create_and_populate_run(idx: int):
            rid = f"concurrent-run-{idx}"
            self.store.create_run(
                RunRecord(
                    run_id=rid,
                    user_query=f"Query {idx}",
                    status="RUNNING",
                    created_at=now_str,
                    updated_at=now_str,
                )
            )
            step = TraceStep(
                run_id=rid,
                sequence=1,
                step_type="RUN_STARTED",
                status="SUCCESS",
                created_at=datetime.now(UTC),
                content=f"Concurrent step {idx}",
            )
            self.store.append_step(step)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_and_populate_run, i) for i in range(num_runs)]
            concurrent.futures.wait(futures)

        for i in range(num_runs):
            run = self.store.get_run(f"concurrent-run-{i}")
            self.assertIsNotNone(run)
            steps = self.store.get_steps(f"concurrent-run-{i}")
            self.assertEqual(len(steps), 1)


if __name__ == "__main__":
    unittest.main()
