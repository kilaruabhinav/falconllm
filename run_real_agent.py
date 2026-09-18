"""Run the real Gemini decision loop with the production tool registry."""
import argparse
import asyncio

from backend.agent.config import AgentConfig
from backend.agent.engine import AgentEngine
from backend.agent.llm_factory import create_llm
from backend.tools.factory import create_tool_registry


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
                query = input("Ask the agent: ")
            except EOFError:
                print("Provide a query interactively or with --query.")
                return 1
        if not query.strip():
            print("Please provide a non-empty query.")
            return 1
        agent = AgentEngine(llm=llm, tool_registry=create_tool_registry(config), config=config)
        result = await agent.run(query)
        print("Status:", result.status)
        print("Answer:", result.answer)
        print("Iterations:", result.iterations)
        print("Trace:")
        for step in result.trace:
            print(step.to_dict())
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
