"""Unit tests for Failure Taxonomy, Error Hierarchy, and Observation Conversion."""

import unittest

from backend.core.errors import (
    ErrorType,
    AgentError,
    ToolTimeoutError,
    UnknownToolError,
    ToolExecutionError,
    InvalidToolArgumentsError,
    EmptyToolResponseError,
    MalformedLLMOutputError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMProviderError,
    RepeatedActionError,
    AgentLoopDetectedError,
    MaxIterationsExceededError,
    PersistenceError,
    SerializationError,
    AgentCancelledError,
)


class TestErrors(unittest.TestCase):
    """Test suite covering ErrorType taxonomy, AgentError properties, and observation conversion."""

    def test_complete_taxonomy_coverage(self) -> None:
        expected_types = [
            "UNKNOWN_TOOL",
            "TOOL_EXCEPTION",
            "TOOL_TIMEOUT",
            "INVALID_TOOL_ARGUMENTS",
            "EMPTY_TOOL_RESPONSE",
            "MALFORMED_LLM_OUTPUT",
            "LLM_API_TIMEOUT",
            "LLM_RATE_LIMIT",
            "LLM_PROVIDER_ERROR",
            "REPEATED_ACTION",
            "AGENT_LOOP_DETECTED",
            "MAX_ITERATIONS_EXCEEDED",
            "PERSISTENCE_ERROR",
            "SERIALIZATION_ERROR",
            "CANCELLED",
            "UNKNOWN_ERROR",
        ]
        for t_name in expected_types:
            self.assertIn(t_name, ErrorType.__members__)

    def test_backward_compatibility_aliases(self) -> None:
        self.assertEqual(ErrorType.INVALID_ARGUMENTS, ErrorType.INVALID_TOOL_ARGUMENTS)
        self.assertEqual(ErrorType.EMPTY_RESPONSE, ErrorType.EMPTY_TOOL_RESPONSE)
        self.assertEqual(ErrorType.MALFORMED_LLM_JSON, ErrorType.MALFORMED_LLM_OUTPUT)
        self.assertEqual(ErrorType.LLM_TIMEOUT, ErrorType.LLM_API_TIMEOUT)
        self.assertEqual(ErrorType.RATE_LIMIT, ErrorType.LLM_RATE_LIMIT)
        self.assertEqual(ErrorType.STUCK_LOOP, ErrorType.AGENT_LOOP_DETECTED)
        self.assertEqual(ErrorType.MAX_ITERATIONS, ErrorType.MAX_ITERATIONS_EXCEEDED)

    def test_agent_error_base_attributes(self) -> None:
        err = AgentError(
            error_type=ErrorType.TOOL_TIMEOUT,
            message="Search timed out",
            retryable=True,
            metadata={"tool": "search", "api_key": "secret123"},
            retry_after_ms=1500,
        )
        self.assertEqual(err.error_type, ErrorType.TOOL_TIMEOUT)
        self.assertEqual(err.message, "Search timed out")
        self.assertTrue(err.retryable)
        self.assertEqual(err.error_code, "ERR_TOOL_TIMEOUT")
        self.assertEqual(err.recommended_action, "retry_with_backoff")
        self.assertEqual(err.retry_after_ms, 1500)
        self.assertEqual(err.metadata["api_key"], "[REDACTED]")
        self.assertEqual(err.metadata["tool"], "search")

    def test_typed_subclasses(self) -> None:
        timeout_err = ToolTimeoutError("Timeout in tool")
        self.assertEqual(timeout_err.error_type, ErrorType.TOOL_TIMEOUT)
        self.assertTrue(timeout_err.retryable)

        unknown_err = UnknownToolError("No such tool")
        self.assertEqual(unknown_err.error_type, ErrorType.UNKNOWN_TOOL)
        self.assertFalse(unknown_err.retryable)

        loop_err = AgentLoopDetectedError("Loop detected")
        self.assertEqual(loop_err.error_type, ErrorType.AGENT_LOOP_DETECTED)
        self.assertFalse(loop_err.retryable)

        max_iter_err = MaxIterationsExceededError("Hit limit")
        self.assertEqual(max_iter_err.error_type, ErrorType.MAX_ITERATIONS_EXCEEDED)
        self.assertFalse(max_iter_err.retryable)

    def test_to_dict_and_redaction(self) -> None:
        cause = ValueError("Original root cause")
        err = AgentError(
            error_type=ErrorType.TOOL_EXCEPTION,
            message="Failure during fetch",
            original_exception=cause,
            metadata={"password": "pwd", "normal_key": "abc"},
        )
        d = err.to_dict()
        self.assertEqual(d["error_type"], "TOOL_EXCEPTION")
        self.assertEqual(d["original_exception"], "ValueError")
        self.assertEqual(d["metadata"]["password"], "[REDACTED]")
        self.assertEqual(d["metadata"]["normal_key"], "abc")

    def test_to_observation_conversion(self) -> None:
        err = ToolTimeoutError("Search tool timed out", retry_after_ms=2000)
        obs = err.to_observation()

        self.assertFalse(obs["success"])
        self.assertEqual(obs["error_type"], "TOOL_TIMEOUT")
        self.assertEqual(obs["message"], "Search tool timed out")
        self.assertTrue(obs["retryable"])
        self.assertEqual(obs["recommended_action"], "retry_with_backoff")
        self.assertEqual(obs["retry_after_ms"], 2000)


if __name__ == "__main__":
    unittest.main()
