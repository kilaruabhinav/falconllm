"""Measure deterministic dispatch and live-trace latency without external API calls."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent.engine import AgentEngine
from backend.agent.mock_llm import MockLLMClient
from backend.api.service import AgentService
from backend.core.persistence import SQLiteTraceStore
from backend.tools.calculator_tool import CalculatorTool
from backend.tools.registry import ToolRegistry


class TimedMockLLM(MockLLMClient):
    async def generate(self, messages, tools):
        await asyncio.sleep(0.05)
        return await super().generate(messages, tools)


async def measure() -> None:
    with tempfile.TemporaryDirectory(prefix="falconllm-latency-") as directory:
        store = SQLiteTraceStore(str(Path(directory) / "trace.db"))

        def engine_factory(manager):
            registry = ToolRegistry()
            registry.register(CalculatorTool())
            llm = TimedMockLLM([
                {"type": "tool", "plan": "Calculate.", "tool": "calculator", "arguments": {"expression": "35 + 46 + 14"}},
                {"type": "final", "plan": "Done.", "answer": "95"},
            ])
            return AgentEngine(llm, registry, trace_manager=manager)

        service = AgentService(store, engine_factory)
        started = time.perf_counter()
        response = await service.create_run("What is 35 + 46 and add 14?")
        dispatched_ms = (time.perf_counter() - started) * 1000
        observed: dict[str, float] = {}
        async for event in service.stream_events(response["run_id"]):
            if event is None:
                continue
            observed.setdefault(event["step_type"], (time.perf_counter() - started) * 1000)
        print(f"POST/service dispatch: {dispatched_ms:.3f} ms")
        print(f"RUN_STARTED visible: {observed['RUN_STARTED']:.3f} ms")
        print(f"first PLAN visible: {observed['PLAN']:.3f} ms")
        print(f"total mock run: {observed['RUN_COMPLETED']:.3f} ms")
        store.close()


if __name__ == "__main__":
    asyncio.run(measure())
