import pytest

from backend.agent.engine import AgentEngine
from backend.agent.mock_llm import (
    MockLLMClient,
)
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
