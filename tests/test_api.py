"""End-to-end HTTP contract tests using the real engine, registry, and SQLite."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from backend.agent.engine import AgentEngine
from backend.agent.mock_llm import MockLLMClient
from backend.api.app import app
from backend.api.routes import get_service
from backend.api.service import AgentService
from backend.core.persistence import SQLiteTraceStore
from backend.tools.calculator_tool import CalculatorTool
from backend.tools.registry import ToolRegistry


class SlowMockLLMClient(MockLLMClient):
    async def generate(self, messages, tools):
        await asyncio.sleep(0.03)
        return await super().generate(messages, tools)


@pytest.fixture
def client(tmp_path: Path):
    store = SQLiteTraceStore(str(tmp_path / "api.db"))

    def engine_factory(manager):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        llm = SlowMockLLMClient([
            {
                "type": "tool",
                "plan": "Calculate the requested expression.",
                "tool": "calculator",
                "arguments": {"expression": "40 + 2"},
            },
            {
                "type": "final",
                "plan": "The calculation is complete.",
                "answer": "40 + 2 = 42",
            },
        ])
        return AgentEngine(llm, registry, trace_manager=manager)

    service = AgentService(store, engine_factory)
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    store.close()


def test_health_endpoint(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_run_trace_and_history_contract(client):
    response = client.post("/api/runs", json={"prompt": "What is 40 + 2?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["dispatch_ms"] >= 0

    run_id = payload["run_id"]
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "RUNNING"
    stream = client.get(f"/api/runs/{run_id}/stream")
    assert stream.status_code == 200
    events = [
        json.loads(line[6:])
        for line in stream.text.splitlines()
        if line.startswith("data: ") and line != "data: {}"
    ]
    types = [event["step_type"] for event in events]
    assert types[0] == "RUN_STARTED"
    assert types.index("PLAN") < types.index("TOOL_CALL")
    assert types.index("TOOL_RESULT") < types.index("PLAN", types.index("PLAN") + 1)
    assert types[-2:] == ["FINAL", "RUN_COMPLETED"]
    plan = next(event for event in events if event["step_type"] == "PLAN")
    completed = events[-1]
    assert plan["duration_ms"] >= 25
    assert completed["duration_ms"] >= plan["duration_ms"]
    assert completed["metadata"]["timing"]["database_writes"] > 0

    assert client.get(f"/api/runs/{run_id}").json()["final_answer"] == "40 + 2 = 42"
    trace = client.get(f"/api/runs/{run_id}/trace").json()["trace"]
    assert [step["sequence"] for step in trace] == list(range(1, len(trace) + 1))
    assert client.get("/api/runs").json()["runs"][0]["run_id"] == run_id

    resume = client.get(f"/api/runs/{run_id}/stream?after=4")
    resumed_events = [
        json.loads(line[6:]) for line in resume.text.splitlines() if line.startswith("data: ")
    ]
    assert resumed_events
    assert all(event["sequence"] > 4 for event in resumed_events)
    assert len({event["step_id"] for event in resumed_events}) == len(resumed_events)


def test_validation_missing_and_cors(client):
    assert client.post("/api/runs", json={"prompt": ""}).status_code == 422
    assert client.get("/api/runs/missing").status_code == 404
    origin = "http://localhost:5175"
    response = client.options(
        "/api/runs",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_failed_run_and_provider_events_terminate_stream(tmp_path: Path):
    store = SQLiteTraceStore(str(tmp_path / "fail.db"))

    class EventfulFailureLLM(MockLLMClient):
        def start_run(self, run_id, event_handler=None):
            self.handler = event_handler

        async def generate(self, messages, tools):
            self.handler("LLM_PROVIDER_SELECTED", {"provider": "gemini", "model": "primary"})
            self.handler("LLM_PROVIDER_ERROR", {
                "provider": "gemini", "model": "primary", "code": "QUOTA_EXCEEDED"
            })
            self.handler("LLM_FALLBACK", {
                "from": "gemini/primary", "to": "openai/fallback", "reason": "QUOTA_EXCEEDED"
            })
            raise RuntimeError("simulated provider failure")

    def engine_factory(manager):
        return AgentEngine(
            EventfulFailureLLM([]),
            ToolRegistry(),
            trace_manager=manager,
            config=SimpleNamespace(MAX_AGENT_ITERATIONS=1),
        )

    service = AgentService(store, engine_factory)
    app.dependency_overrides[get_service] = lambda: service
    try:
        with TestClient(app) as failure_client:
            run_id = failure_client.post("/api/runs", json={"prompt": "fail"}).json()["run_id"]
            response = failure_client.get(f"/api/runs/{run_id}/stream")
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: {}"
            ]
            types = [event["step_type"] for event in events]
            assert "LLM_PROVIDER_SELECTED" in types
            assert "LLM_PROVIDER_ERROR" in types
            assert "LLM_FALLBACK" in types
            assert types[-1] == "RUN_FAILED"
    finally:
        app.dependency_overrides.clear()
        store.close()


def test_simultaneous_runs_are_isolated_and_channels_cleanup(client):
    first = client.post("/api/runs", json={"prompt": "first"}).json()["run_id"]
    second = client.post("/api/runs", json={"prompt": "second"}).json()["run_id"]
    first_events = [
        json.loads(line[6:])
        for line in client.get(f"/api/runs/{first}/stream").text.splitlines()
        if line.startswith("data: ") and line != "data: {}"
    ]
    second_events = [
        json.loads(line[6:])
        for line in client.get(f"/api/runs/{second}/stream").text.splitlines()
        if line.startswith("data: ") and line != "data: {}"
    ]
    assert {event["run_id"] for event in first_events} == {first}
    assert {event["run_id"] for event in second_events} == {second}
    service = app.dependency_overrides[get_service]()
    assert service.event_broker.active_channel_count == 0
