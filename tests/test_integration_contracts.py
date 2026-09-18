"""Integration contract tests simulating end-to-end agent lifecycle.

Simulates the full agent flow:
PLAN -> ACT -> TOOL TIMEOUT -> RECOVERY OBSERVATION -> RETRY/ALTERNATE ACTION -> FINAL ANSWER -> SQLITE RETRIEVAL
without invoking real external LLM APIs or external tools.
"""

from datetime import datetime, UTC
import os
import tempfile
import unittest

from backend.core.errors import (
    ErrorType,
    ToolTimeoutError,
    UnknownToolError,
    RepeatedActionError,
    AgentLoopDetectedError,
    MaxIterationsExceededError,
)
from backend.core.guards import GuardManager
from backend.core.persistence import SQLiteTraceStore
from backend.core.recovery import RecoveryManager
from backend.core.state import AgentState, AgentStatus, StateManager
from backend.core.trace import TraceManager, TraceStepType


class TestIntegrationContracts(unittest.TestCase):
    """End-to-end integration contracts validating orchestration across all core brain subsystems."""

    def setUp(self) -> None:
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.temp_db_fd)
        self.store = SQLiteTraceStore(database_path=self.temp_db_path)
        self.trace_manager = TraceManager(store=self.store)
        self.guard_manager = GuardManager(maximum_iterations=5, max_repeated_actions=3)
        self.recovery_manager = RecoveryManager(
            max_retries=2,
            base_delay=0.1,
            enable_jitter=False,
            idempotent_tools={"search", "calculator"},
        )

    def tearDown(self) -> None:
        self.store.close()
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except Exception:
                pass

    def test_end_to_end_timeout_recovery_success_lifecycle(self) -> None:
        """Simulate:
        1. Run creation
        2. Plan recording
        3. Tool action recording
        4. Tool timeout
        5. Recovery observation
        6. Retry decision
        7. Final completion
        8. Trace retrieval from SQLite
        """
        user_query = "What is Retrieval Augmented Generation?"

        # 1. Run creation
        state = StateManager.create(
            user_query=user_query,
            maximum_iterations=5,
            metadata={"session": "integration_test"},
        )
        self.trace_manager.start_run(state)
        self.assertEqual(state.status, AgentStatus.CREATED)

        # 2. Plan recording
        StateManager.update_iteration(state, 1)
        self.trace_manager.record_plan(
            run_id=state.run_id,
            content="Search the knowledge base for RAG definition",
            data={"iteration": state.current_iteration},
        )

        # 3. Tool action recording (with pre-action guard check)
        tool_name = "search"
        tool_args = {"query": "What is RAG?"}

        guard_result = self.guard_manager.check_action(tool_name, tool_args)
        self.assertTrue(guard_result.passed)

        state.add_action(tool_name=tool_name, arguments=tool_args)
        tool_call = state.add_tool_call(tool_name=tool_name, arguments=tool_args)

        self.trace_manager.record_action(
            run_id=state.run_id,
            tool_name=tool_name,
            arguments=tool_args,
        )

        # 4. Simulate tool timeout failure
        timeout_err = ToolTimeoutError(
            message="Search endpoint timed out after 5000ms",
            retry_after_ms=200,
        )
        tool_call.status = "FAILED"
        tool_call.error = timeout_err.message

        state.add_error(
            error_type=timeout_err.error_type.value,
            message=timeout_err.message,
            retryable=timeout_err.retryable,
            error_code=timeout_err.error_code,
        )
        self.trace_manager.record_error(
            run_id=state.run_id,
            error=timeout_err,
        )

        # 5. Recovery decision
        recovery_decision = self.recovery_manager.recover(timeout_err, tool_name=tool_name)
        self.assertEqual(recovery_decision.action, "RETRY")
        self.assertTrue(recovery_decision.retry)
        self.assertFalse(recovery_decision.terminate_run)

        self.trace_manager.record_recovery(
            run_id=state.run_id,
            recovery_decision=recovery_decision,
        )

        # Convert recovery into an observation for the agent state
        obs = StateManager.update_observation(state, recovery_decision.observation)
        self.assertFalse(obs.success)
        self.assertEqual(obs.error_type, "TOOL_TIMEOUT")

        # 6. Retry execution succeeds
        retry_result = "RAG combines retrieval systems with generative LLMs."
        tool_call_retry = state.add_tool_call(tool_name=tool_name, arguments=tool_args)
        tool_call_retry.status = "COMPLETED"
        tool_call_retry.result = retry_result

        StateManager.update_observation(
            state,
            {"success": True, "result": retry_result},
        )
        self.trace_manager.record_observation(
            run_id=state.run_id,
            observation={"result": retry_result},
            content="Search tool retry succeeded",
        )

        # 7. Final completion
        final_answer = "Retrieval Augmented Generation (RAG) is an AI architecture combining information retrieval with LLM generation."
        StateManager.set_final_answer(state, final_answer)
        self.trace_manager.complete_run(run_id=state.run_id, final_output=final_answer)

        self.assertEqual(state.status, AgentStatus.COMPLETED)
        self.assertEqual(state.final_answer, final_answer)

        # 8. Trace retrieval and validation from SQLite
        persisted_run = self.store.get_run(state.run_id)
        self.assertIsNotNone(persisted_run)
        self.assertEqual(persisted_run.status, "COMPLETED")
        self.assertEqual(persisted_run.final_answer, final_answer)

        persisted_steps = self.store.get_steps(state.run_id)
        self.assertGreaterEqual(len(persisted_steps), 6)

        step_types = [s.step_type for s in persisted_steps]
        self.assertIn(TraceStepType.RUN_STARTED.value, step_types)
        self.assertIn(TraceStepType.PLAN.value, step_types)
        self.assertIn(TraceStepType.ACTION.value, step_types)
        self.assertIn(TraceStepType.ERROR.value, step_types)
        self.assertIn(TraceStepType.RECOVERY.value, step_types)
        self.assertIn(TraceStepType.OBSERVATION.value, step_types)
        self.assertIn(TraceStepType.RUN_COMPLETED.value, step_types)

        # Verify strict sequence numbering
        sequences = [s.sequence for s in persisted_steps]
        self.assertEqual(sequences, list(range(1, len(persisted_steps) + 1)))

    def test_loop_guard_intervention_lifecycle(self) -> None:
        """Verify that loop guard triggers failure recording when an agent enters an infinite loop."""
        state = StateManager.create("Looping query", maximum_iterations=10)
        self.trace_manager.start_run(state)

        # Simulate consecutive repeated actions
        tool_name = "fetch"
        arguments = {"page": "home"}

        # First 2 times are allowed (max_repeated_actions=3)
        res1 = self.guard_manager.check_action(tool_name, arguments)
        self.assertTrue(res1.passed)
        res2 = self.guard_manager.check_action(tool_name, arguments)
        self.assertTrue(res2.passed)

        # 3rd consecutive attempt fails guard
        res3 = self.guard_manager.check_action(tool_name, arguments)
        self.assertFalse(res3.passed)
        self.assertEqual(res3.violation_type, ErrorType.REPEATED_ACTION.value)

        # Recovery decides to replan / break loop
        err = RepeatedActionError(res3.message or "Repeated action", metadata=res3.details)
        decision = self.recovery_manager.recover(err, tool_name=tool_name)
        self.assertEqual(decision.action, "REPLAN")
        self.assertEqual(decision.recommended_action, "break_loop_replan")

    def test_max_iterations_intervention_lifecycle(self) -> None:
        """Verify that maximum iteration guard halts the run when exceeded."""
        gm = GuardManager(maximum_iterations=3)
        state = StateManager.create("Long query", maximum_iterations=3)
        self.trace_manager.start_run(state)

        # Iterations 1 and 2 pass
        self.assertTrue(gm.check_iteration(1).passed)
        self.assertTrue(gm.check_iteration(2).passed)

        # Iteration 3 triggers violation
        iter_check = gm.check_iteration(3)
        self.assertFalse(iter_check.passed)

        self.assertEqual(iter_check.violation_type, ErrorType.MAX_ITERATIONS_EXCEEDED.value)

        # Recovery halts run
        err = MaxIterationsExceededError(iter_check.message or "Max iterations")
        decision = self.recovery_manager.recover(err)
        self.assertEqual(decision.action, "STOP")
        self.assertTrue(decision.terminate_run)

        StateManager.mark_failed(state, err.message)
        self.trace_manager.fail_run(state.run_id, err.message)

        persisted = self.store.get_run(state.run_id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, "FAILED")


if __name__ == "__main__":
    unittest.main()
