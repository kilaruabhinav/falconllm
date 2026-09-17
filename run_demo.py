import asyncio

from backend.agent.engine import AgentEngine
from backend.agent.mock_llm import MockLLMClient
from backend.tools.mock_registry import MockToolRegistry


class DemoConfig:
    MAX_AGENT_ITERATIONS = 5


async def main():

    llm = MockLLMClient(
        responses=[
            {
                "type": "tool",
                "plan": "Need to calculate the value.",
                "tool": "calculator",
                "arguments": {
                    "expression": "40 + 2"
                }
            },
            {
                "type": "final",
                "plan": "Calculation completed.",
                "answer": "40 + 2 = 42"
            }
        ]
    )

    agent = AgentEngine(
        llm=llm,
        tool_registry=MockToolRegistry(),
        config=DemoConfig
    )

    result = await agent.run(
        "What is 40 + 2?"
    )

    print("\n========== AGENT RESULT ==========")

    print("Status:", result.status)
    print("Answer:", result.answer)
    print("Iterations:", result.iterations)

    print("\n========== TRACE ==========")

    for item in result.trace:

        print(
            item.model_dump(
                exclude_none=True
            )
        )


asyncio.run(main())
