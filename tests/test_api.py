"""End-to-end HTTP contract tests using the real engine, registry, and SQLite."""

from pathlib import Path

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


@pytest.fixture
def client(tmp_path: Path):
    store = SQLiteTraceStore(str(tmp_path / "api.db"))

    def engine_factory(manager):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        llm = MockLLMClient([
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
    assert payload["status"] == "completed"
    assert payload["result"] == "40 + 2 = 42"
    assert any(step["step_type"] == "TOOL_CALL" for step in payload["trace"])
    assert any(step["step_type"] == "TOOL_RESULT" for step in payload["trace"])

    run_id = payload["run_id"]
    assert client.get(f"/api/runs/{run_id}").json()["final_answer"] == "40 + 2 = 42"
    trace = client.get(f"/api/runs/{run_id}/trace").json()["trace"]
    assert [step["sequence"] for step in trace] == list(range(1, len(trace) + 1))
    assert client.get("/api/runs").json()["runs"][0]["run_id"] == run_id


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
