"""Contract tests for the Teammate 3 API boundary."""

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from main import app  # noqa: E402


class FalconApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_run_returns_result_and_trace(self) -> None:
        response = self.client.post("/api/runs", json={"prompt": "Plan a demo"})
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["mode"], "demo_adapter")
        self.assertIn("Plan a demo", payload["result"])
        self.assertEqual(len(payload["trace"]), 2)
        self.assertEqual(payload["trace"][0]["status"], "completed")

    def test_empty_prompt_is_rejected(self) -> None:
        response = self.client.post("/api/runs", json={"prompt": ""})
        self.assertEqual(response.status_code, 422)

    def test_cors_allows_vite_fallback_port(self) -> None:
        origin = "http://localhost:5175"
        response = self.client.options(
            "/api/runs",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], origin)


if __name__ == "__main__":
    unittest.main()
