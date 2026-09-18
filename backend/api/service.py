"""Application service connecting HTTP handlers to the custom agent framework."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.agent.config import AgentConfig
from backend.agent.engine import AgentEngine
from backend.agent.llm_factory import create_llm
from backend.core.persistence import SQLiteTraceStore
from backend.core.trace import TraceManager
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
    ) -> None:
        self.store = store
        self.engine_factory = engine_factory

    async def create_run(self, prompt: str) -> dict[str, Any]:
        manager = TraceManager(self.store)
        result = await self.engine_factory(manager).run(prompt)
        return {
            "run_id": result.run_id,
            "status": result.status,
            "result": result.answer,
            "iterations": result.iterations,
            "trace": [step.to_dict() for step in result.trace],
        }

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

    def engine_factory(manager: TraceManager) -> AgentEngine:
        return AgentEngine(
            llm=create_llm(config),
            tool_registry=create_tool_registry(config, file_root=Path("data")),
            trace_manager=manager,
            config=config,
        )

    return AgentService(store, engine_factory)
