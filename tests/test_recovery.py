"""Unit tests for Failure Recovery, Backoff Policies, and Decision Engine."""

import unittest

from backend.core.errors import (
    AgentError,
    ErrorType,
    ToolTimeoutError,
    UnknownToolError,
    InvalidToolArgumentsError,
    EmptyToolResponseError,
    MalformedLLMOutputError,
    LLMTimeoutError,
    LLMRateLimitError,
    MaxIterationsExceededError,
)
from backend.core.recovery import RecoveryManager, RecoveryDecision


class TestRecovery(unittest.TestCase):
    """Test suite covering recovery decisions, exponential backoff, retry bounds, and idempotency."""

    def setUp(self) -> None:
        # Disable jitter for deterministic delay assertion in unit tests
        self.recovery = RecoveryManager(
            max_retries=2,
            base_delay=1.0,
            max_delay=10.0,
            enable_jitter=False,
            idempotent_tools={"search", "read_file"},
        )

    def test_unknown_tool_decision(self) -> None:
        err = UnknownToolError("Tool 'fly_drone' does not exist")
        decision = self.recovery.recover(err, tool_name="fly_drone")

        self.assertEqual(decision.action, "STOP")
        self.assertFalse(decision.retry)
        self.assertTrue(decision.recoverable)
        self.assertFalse(decision.terminate_run)
        self.assertEqual(decision.observation["recommended_action"], "switch_tool")

    def test_invalid_arguments_decision(self) -> None:
        err = InvalidToolArgumentsError("Query parameter missing")
        decision = self.recovery.recover(err, tool_name="search")

        self.assertEqual(decision.action, "REPLAN")
        self.assertFalse(decision.retry)
        self.assertTrue(decision.recoverable)
        self.assertEqual(decision.observation["recommended_action"], "replan")

    def test_empty_tool_response_decision(self) -> None:
        err = EmptyToolResponseError("Search returned 0 bytes")
        decision = self.recovery.recover(err, tool_name="search")

        self.assertEqual(decision.action, "REPLAN")
        self.assertFalse(decision.retry)

    def test_bounded_retry_and_exponential_backoff(self) -> None:
        err = ToolTimeoutError("Search timed out")

        # Attempt 1
        d1 = self.recovery.recover(err, tool_name="search")
        self.assertEqual(d1.action, "RETRY")
        self.assertTrue(d1.retry)
        self.assertEqual(d1.retry_delay, 1.0)  # 1.0 * (2^0) = 1.0

        # Attempt 2
        d2 = self.recovery.recover(err, tool_name="search")
        self.assertEqual(d2.action, "RETRY")
        self.assertTrue(d2.retry)
        self.assertEqual(d2.retry_delay, 2.0)  # 1.0 * (2^1) = 2.0

        # Attempt 3 (exceeded max_retries = 2)
        d3 = self.recovery.recover(err, tool_name="search")
        self.assertEqual(d3.action, "STOP")
        self.assertFalse(d3.retry)
        self.assertTrue(d3.terminate_run)

    def test_non_idempotent_tool_retry_suppressed(self) -> None:
        err = ToolTimeoutError("API call timed out")
        # 'send_payment' is not in idempotent_tools
        decision = self.recovery.recover(err, tool_name="send_payment")

        self.assertEqual(decision.action, "REPLAN")
        self.assertFalse(decision.retry)
        self.assertIn("Automatic retry suppressed", decision.observation["message"])

    def test_llm_rate_limit_with_retry_after(self) -> None:
        err = LLMRateLimitError("Rate limit exceeded", retry_after_ms=3500)
        decision = self.recovery.recover(err)

        self.assertEqual(decision.action, "RETRY")
        self.assertTrue(decision.retry)
        self.assertEqual(decision.retry_delay, 3.5)

    def test_max_iterations_terminate_run(self) -> None:
        err = MaxIterationsExceededError("Limit reached")
        decision = self.recovery.recover(err)

        self.assertEqual(decision.action, "STOP")
        self.assertFalse(decision.retry)
        self.assertFalse(decision.recoverable)
        self.assertTrue(decision.terminate_run)

    def test_handle_tool_result_integration(self) -> None:
        # Success result
        res_ok = self.recovery.handle_tool_result("r1", None, "search", {"results": [1, 2]})
        self.assertEqual(res_ok.action, "CONTINUE")
        self.assertTrue(res_ok.observation["success"])

        # Tool failure dict
        res_fail = self.recovery.handle_tool_result(
            "r1",
            None,
            "search",
            {"success": False, "error_type": ErrorType.TOOL_TIMEOUT, "message": "Search slow"},
        )
        self.assertEqual(res_fail.action, "RETRY")
        self.assertTrue(res_fail.retry)


if __name__ == "__main__":
    unittest.main()
