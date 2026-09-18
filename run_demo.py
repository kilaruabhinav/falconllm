import asyncio

from backend.agent.engine import AgentEngine
from backend.agent.mock_llm import MockLLMClient
from backend.agent.config import AgentConfig
from backend.tools.factory import create_tool_registry


class DemoConfig:
    MAX_AGENT_ITERATIONS = 5


async def main():

    llm = MockLLMClient(
        responses=[
            {
                "type": "tool",
                "plan": "Read the local product price.",
                "tool": "file_reader",
                "arguments": {
                    "path": "demo_price.txt"
                }
            },
            {
                "type": "tool",
                "plan": "Calculate the price after a 20% discount.",
                "tool": "calculator",
                "arguments": {"expression": "125 * 0.8"}
            },
            {
                "type": "final",
                "plan": "The discounted price is ready.",
                "answer": "The product costs 100 after a 20% discount."
            }
        ]
    )

    agent = AgentEngine(
        llm=llm,
        tool_registry=create_tool_registry(AgentConfig(env_file=None)),
        config=DemoConfig
    )

    result = await agent.run(
        "Read the demo product price and apply a 20% discount."
    )

    print("\n========== AGENT RESULT ==========")

    print("Status:", result.status)
    print("Answer:", result.answer)
    print("Iterations:", result.iterations)

    print("\n========== TRACE ==========")

    for item in result.trace:

        print(item.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
