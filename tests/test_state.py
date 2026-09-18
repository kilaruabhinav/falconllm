"""Unit tests for AgentState and StateManager."""

from datetime import datetime, UTC
import json
import unittest

from backend.core.state import (
    AgentState,
    AgentStatus,
    Action,
    ToolCall,
    Observation,
    ErrorRecord,
    StateManager,
    redact_credentials,
    truncate_payload,
)


class TestAgentState(unittest.TestCase):
    """Test suite covering AgentState lifecycle, validation, and serialization."""

    def test_state_creation_and_defaults(self) -> None:
        state = AgentState(user_query="Hello world")
        self.assertEqual(state.user_query, "Hello world")
        self.assertIsNotNone(state.run_id)
        self.assertEqual(state.current_iteration, 0)
        self.assertEqual(state.maximum_iterations, 10)
        self.assertEqual(state.status, AgentStatus.CREATED)
        self.assertEqual(len(state.messages), 0)
        self.assertEqual(len(state.actions_taken), 0)
        self.assertEqual(len(state.tool_calls), 0)
        self.assertEqual(len(state.observations), 0)
        self.assertEqual(len(state.errors), 0)
        self.assertIsNone(state.final_answer)
        self.assertEqual(state.created_at.tzinfo, UTC)
        self.assertEqual(state.updated_at.tzinfo, UTC)

    def test_mutable_default_isolation(self) -> None:
        state1 = AgentState(user_query="Query 1")
        state2 = AgentState(user_query="Query 2")

        state1.add_message("user", "Msg 1")
        state1.add_action("tool1", {"arg": "val"})

        self.assertEqual(len(state1.messages), 1)
        self.assertEqual(len(state1.actions_taken), 1)
        self.assertEqual(len(state2.messages), 0)
        self.assertEqual(len(state2.actions_taken), 0)
        self.assertIsNot(state1.messages, state2.messages)
        self.assertIsNot(state1.actions_taken, state2.actions_taken)

    def test_validation_constraints(self) -> None:
        with self.assertRaises(ValueError):
            AgentState(user_query="test", maximum_iterations=0)
        with self.assertRaises(ValueError):
            AgentState(user_query="test", maximum_iterations=-5)
        with self.assertRaises(ValueError):
            AgentState(user_query="test", current_iteration=-1)

    def test_status_transitions(self) -> None:
        state = AgentState(user_query="Query")
        self.assertEqual(state.status, AgentStatus.CREATED)

        state.set_status(AgentStatus.RUNNING)
        self.assertEqual(state.status, AgentStatus.RUNNING)

        state.set_status(AgentStatus.WAITING_FOR_TOOL)
        self.assertEqual(state.status, AgentStatus.WAITING_FOR_TOOL)

        state.set_status(AgentStatus.COMPLETED)
        self.assertEqual(state.status, AgentStatus.COMPLETED)

    def test_credential_redaction(self) -> None:
        state = AgentState(user_query="Query", metadata={"api_key": "sk-secret-12345", "safe_val": 42})
        state.add_action("call_api", {"api_key": "sk-secret", "token": "bearer xyz", "param": "ok"})
        state.add_tool_call("call_api", {"authorization": "Bearer 123", "input": "test"})

        data = state.to_dict()
        self.assertEqual(data["metadata"]["api_key"], "[REDACTED]")
        self.assertEqual(data["metadata"]["safe_val"], 42)
        self.assertEqual(data["actions"][0]["arguments"]["api_key"], "[REDACTED]")
        self.assertEqual(data["actions"][0]["arguments"]["token"], "[REDACTED]")
        self.assertEqual(data["actions"][0]["arguments"]["param"], "ok")
        self.assertEqual(data["tool_calls"][0]["arguments"]["authorization"], "[REDACTED]")

    def test_payload_truncation(self) -> None:
        huge_str = "x" * 1000
        truncated = truncate_payload(huge_str, max_chars=100)
        self.assertIsInstance(truncated, dict)
        self.assertTrue(truncated["truncated"])
        self.assertEqual(truncated["original_length"], 1000)
        self.assertTrue(truncated["preview"].endswith("[TRUNCATED]"))

        normal_str = "short"
        self.assertEqual(truncate_payload(normal_str, max_chars=100), "short")

    def test_serialization_and_deserialization(self) -> None:
        state = AgentState(
            user_query="What is RAG?",
            maximum_iterations=5,
            metadata={"source": "unit_test"},
        )
        state.add_message("user", "What is RAG?")
        state.add_action("search", {"query": "RAG"})
        state.add_tool_call("search", {"query": "RAG"})
        state.add_observation(success=True, result="Retrieval Augmented Generation")
        state.add_error("TEST_ERROR", "Minor failure", retryable=True)
        state.update_token_usage(input_tokens=100, output_tokens=50, estimated_cost=0.002)
        state.final_answer = "Retrieval Augmented Generation"
        state.status = AgentStatus.COMPLETED

        # to_dict & to_json
        state_dict = state.to_dict()
        state_json = state.to_json()

        self.assertIsInstance(state_dict, dict)
        self.assertIsInstance(state_json, str)

        # from_dict & from_json
        reconstructed = AgentState.from_json(state_json)
        self.assertEqual(reconstructed.run_id, state.run_id)
        self.assertEqual(reconstructed.user_query, state.user_query)
        self.assertEqual(reconstructed.status, AgentStatus.COMPLETED.value)
        self.assertEqual(len(reconstructed.messages), 1)
        self.assertEqual(len(reconstructed.actions_taken), 1)
        self.assertEqual(len(reconstructed.tool_calls), 1)
        self.assertEqual(len(reconstructed.observations), 1)
        self.assertEqual(len(reconstructed.errors), 1)
        self.assertEqual(reconstructed.final_answer, state.final_answer)
        self.assertEqual(reconstructed.total_tokens, 150)
        self.assertEqual(reconstructed.estimated_cost, 0.002)
        self.assertEqual(reconstructed.created_at.tzinfo, UTC)

    def test_state_manager_facade(self) -> None:
        state = StateManager.create("Explain AI", maximum_iterations=12, env="test")
        self.assertEqual(state.user_query, "Explain AI")
        self.assertEqual(state.maximum_iterations, 12)

        new_iter = StateManager.update_iteration(state, 2)
        self.assertEqual(new_iter, 2)
        self.assertEqual(state.current_iteration, 2)

        obs = StateManager.update_observation(
            state,
            {"success": True, "result": "AI explanation", "retryable": False},
        )
        self.assertTrue(obs.success)
        self.assertEqual(len(state.observations), 1)

        StateManager.set_final_answer(state, "AI is artificial intelligence.")
        self.assertEqual(state.status, AgentStatus.COMPLETED)
        self.assertEqual(state.final_answer, "AI is artificial intelligence.")


if __name__ == "__main__":
    unittest.main()
