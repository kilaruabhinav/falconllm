"""Run the real Gemini decision loop with temporary mock tools."""
import argparse
import asyncio

from backend.agent.config import AgentConfig
from backend.agent.engine import AgentEngine
from backend.agent.llm_factory import create_llm
from backend.tools.mock_registry import MockToolRegistry


async def main(query: str | None = None) -> int:
    try:
        config = AgentConfig()
        llm = create_llm(config)
    except ValueError as exc:
        print(f"Setup error: {exc}")
        return 1

    try:
        if query is None:
            try:
                query = input("Ask the agent (tools are mocks): ")
            except EOFError:
                print("Provide a query interactively or with --query.")
                return 1
        if not query.strip():
            print("Please provide a non-empty query.")
            return 1
        agent = AgentEngine(llm=llm, tool_registry=MockToolRegistry(), config=config)
        result = await agent.run(query)
        print("Status:", result.status)
        print("Answer:", result.answer)
        print("Iterations:", result.iterations)
        print("Trace:")
        for step in result.trace:
            print(step.model_dump_json(exclude_none=True))
        return 0 if result.status == "completed" else 1
    finally:
        await llm.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Non-interactive query; otherwise prompt in the terminal.")
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(main(args.query)))
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
