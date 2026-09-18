import pytest

from backend.agent.engine import AgentEngine
from backend.agent.mock_llm import (
    MockLLMClient,
)
from backend.agent.planner import Planner
from backend.tools.mock_registry import (
    MockToolRegistry,
)


class TestConfig:
    MAX_AGENT_ITERATIONS = 5


@pytest.mark.asyncio
async def test_calculator_agent():

    llm = MockLLMClient(
        responses=[
            {
                "type": "tool",
                "plan": (
                    "Need to calculate "
                    "the expression."
                ),
                "tool": "calculator",
                "arguments": {
                    "expression": "40 + 2"
                },
            },
            {
                "type": "final",
                "plan": (
                    "The calculation "
                    "is complete."
                ),
                "answer": "The answer is 42.",
            },
        ]
    )

    tools = MockToolRegistry()

    agent = AgentEngine(
        llm=llm,
        tool_registry=tools,
        config=TestConfig,
    )

    result = await agent.run(
        "What is 40 + 2?"
    )

    assert (
        result.status
        == "completed"
    )

    assert (
        result.answer
        == "The answer is 42."
    )

    assert result.iterations == 2

    tool_calls = [
        item
        for item in result.trace
        if item.type == "tool_call"
    ]

    assert len(tool_calls) == 1

    assert (
        tool_calls[0].tool
        == "calculator"
    )


def tool_action(expression="40 + 2", tool="calculator"):
    return {"type": "tool", "plan": "Need a calculation before answering.",
            "tool": tool, "arguments": {"expression": expression}}


def final_action():
    return {"type": "final", "plan": "Enough information has been gathered.", "answer": "42"}


@pytest.mark.asyncio
async def test_failed_tool_observation_and_retry():
    llm = MockLLMClient([tool_action("1 / 0"), tool_action(), final_action()])
    result = await AgentEngine(llm, MockToolRegistry(), config=TestConfig).run("Calculate")
    assert result.status == "completed"
    assert result.iterations == 3
    assert [s.type for s in result.trace].count("tool_error") == 1
    messages = llm.calls[1]["messages"]
    assert any("Success: False" in m["content"] for m in messages)
    assert any(m["role"] == "assistant" and "1 / 0" in m["content"] for m in messages)
    assert any("Result: 42" in m["content"] for m in llm.calls[2]["messages"])
    assert [s.step for s in result.trace] == list(range(1, len(result.trace) + 1))


@pytest.mark.asyncio
async def test_unknown_tool():
    llm = MockLLMClient([tool_action(tool="missing"), final_action()])
    result = await AgentEngine(llm, MockToolRegistry()).run("test")
    assert result.status == "completed"
    assert any(s.type == "tool_error" and "Unknown tool" in s.error for s in result.trace)
    assert any("Unknown tool" in m["content"] for m in llm.calls[1]["messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["not JSON", "", '{"type":"tool"}'])
async def test_malformed_output_recovery(bad):
    llm = MockLLMClient([bad, final_action()])
    result = await AgentEngine(llm, MockToolRegistry()).run("test")
    assert result.status == "completed"
    assert any(step.type == "parse_error" for step in result.trace)
    assert any("invalid" in m["content"].lower() for m in llm.calls[1]["messages"])


@pytest.mark.asyncio
async def test_iteration_limit():
    llm = MockLLMClient([tool_action()] * 5)
    result = await AgentEngine(llm, MockToolRegistry(), config=TestConfig).run("test")
    assert result.status == "max_iterations"
    assert result.iterations == llm.call_count == 5
    assert result.answer is None


@pytest.mark.asyncio
async def test_llm_exception_recovery():
    llm = MockLLMClient([RuntimeError("temporarily unavailable"), final_action()])
    result = await AgentEngine(llm, MockToolRegistry()).run("test")
    assert result.status == "completed"
    assert any(step.type == "llm_error" for step in result.trace)
    assert any("temporarily unavailable" in m["content"] for m in llm.calls[1]["messages"])


@pytest.mark.asyncio
async def test_persistent_llm_failure_is_bounded():
    llm = MockLLMClient([RuntimeError("unavailable")] * 5)
    result = await AgentEngine(llm, MockToolRegistry(), config=TestConfig).run("test")
    assert result.status == "max_iterations"
    assert [s.type for s in result.trace].count("llm_error") == 5
    assert result.trace[0].type == "run_started"
    assert result.trace[-1].type == "run_failed"


@pytest.mark.asyncio
async def test_registry_exception_becomes_observation():
    class BrokenRegistry(MockToolRegistry):
        async def execute(self, tool_name, arguments):
            raise RuntimeError("tool unavailable")

    llm = MockLLMClient([tool_action(), final_action()])
    result = await AgentEngine(llm, BrokenRegistry()).run("test")
    assert result.status == "completed"
    assert any("tool unavailable" in m["content"] for m in llm.calls[1]["messages"])


@pytest.mark.asyncio
async def test_dynamic_registry_and_repeated_runs():
    from backend.agent.schemas import ToolResult

    class Registry:
        def __init__(self):
            self.used = False

        def get_tool_schemas(self):
            return [] if self.used else [{"name": "teammate_tool", "description": "New tool",
                                        "parameters": {"type": "object"}}]

        async def execute(self, tool_name, arguments):
            assert tool_name == "teammate_tool"
            self.used = True
            return ToolResult(success=True, tool=tool_name, result=42)

    llm = MockLLMClient([tool_action(tool="teammate_tool"), final_action(), final_action()])
    engine = AgentEngine(llm, Registry())
    result = await engine.run("test")
    assert result.status == "completed"
    assert llm.calls[0]["tools"][0]["name"] == "teammate_tool"
    assert llm.calls[1]["tools"] == []
    result2 = await engine.run("new request")
    assert result2.iterations == 1
    assert result2.trace[0].step == 1
    assert not any("TOOL OBSERVATION" in m["content"] for m in llm.calls[2]["messages"])


def test_planner_bounds_history_and_observation_size():
    history = [
        {"type": "tool_result", "tool": "search", "success": True, "result": "x" * 10_000}
        for _ in range(12)
    ]
    messages = Planner().build_messages("query", history, [], 3)
    observations = [message for message in messages if "TOOL OBSERVATION" in message["content"]]
    assert len(observations) == Planner.MAX_HISTORY_ITEMS
    assert all("[TRUNCATED]" in message["content"] for message in observations)
    assert all(len(message["content"]) < 2_600 for message in observations)
