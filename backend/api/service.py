"""Application service connecting HTTP handlers to the custom agent framework."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import time
from typing import Any
import uuid

from backend.agent.config import AgentConfig
from backend.agent.engine import AgentEngine
from backend.agent.llm_factory import create_llm
from backend.core.persistence import SQLiteTraceStore
from backend.core.trace import TraceManager
from backend.core.trace_events import TERMINAL_TRACE_TYPES, TraceEventBroker
from backend.tools.factory import create_tool_registry


def sqlite_path(database_url: str) -> str:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("Only sqlite:/// DATABASE_URL values are supported.")
    path = database_url[len(prefix):]
    if not path:
        raise ValueError("DATABASE_URL must include a SQLite path.")
    return path


class AgentService:
    def __init__(
        self,
        store: SQLiteTraceStore,
        engine_factory: Callable[[TraceManager], AgentEngine],
        event_broker: TraceEventBroker | None = None,
    ) -> None:
        self.store = store
        self.engine_factory = engine_factory
        self.event_broker = event_broker or TraceEventBroker()
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def create_run(self, prompt: str) -> dict[str, Any]:
        started = time.perf_counter()
        run_id = str(uuid.uuid4())
        self.event_broker.create_run(run_id)
        task = asyncio.create_task(
            self._execute_run(run_id, prompt), name=f"falconllm-run-{run_id}"
        )
        self._tasks[run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(run_id, None))
        return {
            "run_id": run_id,
            "status": "running",
            "dispatch_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    async def _execute_run(self, run_id: str, prompt: str) -> None:
        manager = TraceManager(self.store, publisher=self.event_broker.publish)
        try:
            engine = self.engine_factory(manager)
            await engine.run(prompt, run_id=run_id)
        except asyncio.CancelledError:
            trace = manager.get_trace(run_id)
            if trace is None:
                trace = manager.start_run(prompt, run_id=run_id)
            if trace.status == "RUNNING":
                manager.cancel_run(run_id)
            raise
        except Exception as exc:
            trace = manager.get_trace(run_id)
            if trace is None:
                manager.start_run(prompt, run_id=run_id)
                trace = manager.get_trace(run_id)
            if trace is not None and trace.status == "RUNNING":
                manager.fail_run(run_id, f"Unexpected agent failure: {type(exc).__name__}")
        finally:
            self.event_broker.finish(run_id)

    async def stream_events(self, run_id: str, after_sequence: int = 0):
        """Yield persisted catch-up followed by live events without duplicates."""
        last_sequence = max(0, after_sequence)
        persisted = await asyncio.to_thread(self.get_trace, run_id)
        for event in persisted or []:
            sequence = int(event.get("sequence", 0))
            if sequence <= last_sequence:
                continue
            last_sequence = sequence
            yield event
            if event.get("step_type") in TERMINAL_TRACE_TYPES:
                return

        subscription, backlog = self.event_broker.subscribe(run_id, last_sequence)
        try:
            for event in backlog:
                sequence = int(event.get("sequence", 0))
                if sequence <= last_sequence:
                    continue
                last_sequence = sequence
                yield event
                if event.get("step_type") in TERMINAL_TRACE_TYPES:
                    return

            if subscription is None:
                return
            while True:
                try:
                    event = await asyncio.wait_for(subscription.queue.get(), timeout=12)
                except asyncio.TimeoutError:
                    yield None
                    continue
                sequence = int(event.get("sequence", 0))
                if sequence <= last_sequence:
                    continue
                last_sequence = sequence
                yield event
                if event.get("step_type") in TERMINAL_TRACE_TYPES:
                    return
        finally:
            self.event_broker.unsubscribe(subscription)

    def has_run(self, run_id: str) -> bool:
        return (
            run_id in self._tasks
            or self.event_broker.has_run(run_id)
            or self.store.get_run(run_id) is not None
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.store.get_run(run_id)
        return run.__dict__ if run else None

    def get_trace(self, run_id: str) -> list[dict[str, Any]] | None:
        if self.store.get_run(run_id) is None:
            return None
        return [step.to_dict() for step in self.store.get_steps(run_id)]

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return [run.__dict__ for run in self.store.list_runs(limit)]


def create_default_service(config: AgentConfig | None = None) -> AgentService:
    config = config or AgentConfig()
    store = SQLiteTraceStore(sqlite_path(config.DATABASE_URL))
    shared_llm = create_llm(config)
    shared_registry = create_tool_registry(config, file_root=Path("data"))

    def engine_factory(manager: TraceManager) -> AgentEngine:
        llm = shared_llm.spawn_router() if hasattr(shared_llm, "spawn_router") else shared_llm
        return AgentEngine(
            llm=llm,
            tool_registry=shared_registry,
            trace_manager=manager,
            config=config,
        )

    return AgentService(store, engine_factory)
