"""Unit tests for ExecutionTrace and TraceManager."""

import concurrent.futures
from datetime import datetime, UTC
import json
import unittest

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = ROOT_DIR / "backend" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from trace import (
    ExecutionTrace,
    TraceStep,
    TraceStepType,
    TraceManager,
)


class TestExecutionTrace(unittest.TestCase):
    """Test suite covering ExecutionTrace step appending, sequencing, thread safety, and redaction."""

    def test_unique_run_ids_and_defaults(self) -> None:
        t1 = ExecutionTrace()
        t2 = ExecutionTrace()
        self.assertNotEqual(t1.run_id, t2.run_id)
        self.assertEqual(t1.status, "RUNNING")
        self.assertEqual(len(t1.steps), 0)
        self.assertEqual(t1.started_at.tzinfo, UTC)

    def test_ordered_monotonic_steps(self) -> None:
        trace = ExecutionTrace()
        s1 = trace.add_step(TraceStepType.RUN_STARTED, content="Start")
        s2 = trace.add_plan(content="Plan something")
        s3 = trace.add_action(tool_name="calculator", arguments={"expr": "1+1"})
        s4 = trace.add_observation(observation={"res": 2})
        s5 = trace.add_recovery(description="Handled")
        s6 = trace.add_error(error="Small issue")
        s7 = trace.complete(final_output="2")

        self.assertEqual([s.sequence for s in trace.steps], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(s1.step_type, TraceStepType.RUN_STARTED.value)
        self.assertEqual(s2.step_type, TraceStepType.PLAN.value)
        self.assertEqual(s3.step_type, TraceStepType.ACTION.value)
        self.assertEqual(s4.step_type, TraceStepType.OBSERVATION.value)
        self.assertEqual(s5.step_type, TraceStepType.RECOVERY.value)
        self.assertEqual(s6.step_type, TraceStepType.ERROR.value)
        self.assertEqual(s7.step_type, TraceStepType.RUN_COMPLETED.value)
        self.assertEqual(trace.status, "COMPLETED")
        self.assertEqual(trace.final_output, "2")
        self.assertIsNotNone(trace.completed_at)

    def test_duplicate_sequence_rejection(self) -> None:
        trace = ExecutionTrace()
        trace.add_step(TraceStepType.RUN_STARTED, content="Start", sequence=1)
        with self.assertRaises(ValueError):
            trace.add_step(TraceStepType.PLAN, content="Duplicate seq", sequence=1)

    def test_thread_safe_concurrent_appends(self) -> None:
        trace = ExecutionTrace()
        num_threads = 20

        def append_step(idx: int):
            trace.add_step(
                step_type=TraceStepType.ACTION,
                content=f"Thread action {idx}",
                metadata={"worker_id": idx},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(append_step, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        self.assertEqual(len(trace.steps), num_threads)
        sequences = [s.sequence for s in trace.steps]
        self.assertEqual(len(sequences), len(set(sequences)), "Sequence numbers must be unique")
        self.assertEqual(min(sequences), 1)
        self.assertEqual(max(sequences), num_threads)

    def test_credential_redaction_in_trace(self) -> None:
        trace = ExecutionTrace()
        step = trace.add_action(
            tool_name="api_caller",
            arguments={"api_key": "secret-val-123", "param": "public"},
            data={"token": "bearer-token-abc"},
        )
        data = step.to_dict()
        self.assertEqual(data["arguments"]["api_key"], "[REDACTED]")
        self.assertEqual(data["arguments"]["param"], "public")
        self.assertEqual(data["metadata"]["token"], "[REDACTED]")

    def test_serialization_roundtrip(self) -> None:
        trace = ExecutionTrace()
        trace.add_plan(content="Initial plan")
        trace.add_action(tool_name="search", arguments={"query": "test"})
        trace.complete(final_output="Done")

        trace_dict = trace.to_dict()
        trace_json = trace.to_json()

        self.assertIsInstance(trace_dict, dict)
        self.assertEqual(trace_dict["total_steps"], 3)
        self.assertIsInstance(trace_json, str)

        reconstructed = ExecutionTrace.from_dict(json.loads(trace_json))
        self.assertEqual(reconstructed.run_id, trace.run_id)
        self.assertEqual(reconstructed.status, "COMPLETED")
        self.assertEqual(len(reconstructed.steps), 3)
        self.assertEqual(reconstructed.steps[0].content, "Initial plan")
        self.assertEqual(reconstructed.steps[1].tool_name, "search")

    def test_trace_manager(self) -> None:
        tm = TraceManager()
        trace = tm.start_run(state_or_query="Sample query", run_id="run-100")
        self.assertEqual(trace.run_id, "run-100")
        self.assertEqual(len(trace.steps), 1)

        s_plan = tm.record_plan("run-100", content="Need search")
        self.assertEqual(s_plan.step_type, TraceStepType.PLAN.value)

        s_act = tm.record_action("run-100", tool_name="search", arguments={"q": "rag"})
        self.assertEqual(s_act.tool_name, "search")

        s_obs = tm.record_observation("run-100", observation={"text": "RAG info"})
        self.assertEqual(s_obs.observation, {"text": "RAG info"})

        s_done = tm.complete_run("run-100", final_output="Final result")
        self.assertEqual(s_done.step_type, TraceStepType.RUN_COMPLETED.value)

        fetched = tm.get_trace("run-100")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.status, "COMPLETED")
        self.assertEqual(len(fetched.steps), 5)


if __name__ == "__main__":
    unittest.main()
