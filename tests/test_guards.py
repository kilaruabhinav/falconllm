"""Unit tests for Guards, Loop Protection, and Payload Limits."""

import unittest

from backend.core.errors import (
    RepeatedActionError,
    AgentLoopDetectedError,
    MaxIterationsExceededError,
)
from backend.core.guards import (
    IterationGuard,
    RepeatedActionGuard,
    LoopGuard,
    PayloadGuard,
    GuardManager,
    canonicalize_arguments,
)


class TestGuards(unittest.TestCase):
    """Test suite covering iteration guards, canonical repeated-action checks, loop cycles, and payload limits."""

    def test_iteration_guard(self) -> None:
        guard = IterationGuard(maximum_iterations=5, raise_on_violation=True)
        self.assertTrue(guard.check_iteration(1))
        self.assertTrue(guard.check_iteration(4))

        with self.assertRaises(MaxIterationsExceededError):
            guard.check_iteration(5)

        # Structured non-raising evaluation
        safe_guard = IterationGuard(maximum_iterations=5, raise_on_violation=False)
        res = safe_guard.evaluate(5)
        self.assertFalse(res.passed)
        self.assertEqual(res.violation_type, "MAX_ITERATIONS_EXCEEDED")

    def test_canonical_argument_ordering(self) -> None:
        dict1 = {"query": "rag", "limit": 10, "nested": {"b": 2, "a": 1}}
        dict2 = {"limit": 10, "nested": {"a": 1, "b": 2}, "query": "rag"}

        self.assertEqual(
            canonicalize_arguments(dict1),
            canonicalize_arguments(dict2),
            "Key ordering must produce identical canonical representation",
        )

    def test_repeated_action_detection(self) -> None:
        guard = RepeatedActionGuard(max_repeated_actions=3, raise_on_violation=True)

        guard.check_action("search", {"q": "rag", "p": 1})
        guard.check_action("search", {"p": 1, "q": "rag"})  # key order shuffled

        # Third identical action triggers violation
        with self.assertRaises(RepeatedActionError):
            guard.check_action("search", {"q": "rag", "p": 1})

    def test_different_arguments_do_not_trigger_repeated_action(self) -> None:
        guard = RepeatedActionGuard(max_repeated_actions=3, raise_on_violation=True)

        for page in range(1, 10):
            self.assertTrue(guard.check_action("search", {"query": "rag", "page": page}))

    def test_multi_step_loop_cycle_detection(self) -> None:
        # LoopGuard detecting cycle of period 2: tool_a -> tool_b -> tool_a -> tool_b
        guard = LoopGuard(
            max_repeated_actions=5,
            history_window=10,
            cycle_threshold=2,
            raise_on_violation=True,
        )

        guard.check_action("tool_a", {"step": 1})
        guard.check_action("tool_b", {"step": 2})
        guard.check_action("tool_a", {"step": 1})

        # Completing the 2nd cycle of (tool_a, tool_b) triggers AgentLoopDetectedError
        with self.assertRaises(AgentLoopDetectedError):
            guard.check_action("tool_b", {"step": 2})

    def test_payload_guard_truncation(self) -> None:
        guard = PayloadGuard(
            max_observation_chars=50,
            max_argument_chars=50,
            max_trace_steps=5,
        )

        # Observation truncation
        big_obs = "A" * 100
        sanitized = guard.sanitize_observation(big_obs)
        self.assertIsInstance(sanitized, dict)
        self.assertTrue(sanitized["truncated"])
        self.assertEqual(sanitized["original_length"], 100)

        # Capacity check
        cap_ok = guard.check_trace_steps(4)
        self.assertTrue(cap_ok.passed)

        cap_fail = guard.check_trace_steps(5)
        self.assertFalse(cap_fail.passed)

    def test_guard_manager(self) -> None:
        gm = GuardManager(maximum_iterations=3, max_repeated_actions=2)

        iter_res = gm.check_iteration(2)
        self.assertTrue(iter_res.passed)

        iter_fail = gm.check_iteration(3)
        self.assertFalse(iter_fail.passed)

        act1 = gm.check_action("search", {"q": "ai"})
        self.assertTrue(act1.passed)

        act2 = gm.check_action("search", {"q": "ai"})
        self.assertFalse(act2.passed)
        self.assertEqual(act2.violation_type, "REPEATED_ACTION")


if __name__ == "__main__":
    unittest.main()
