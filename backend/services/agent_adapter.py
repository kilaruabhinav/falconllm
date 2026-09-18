"""Temporary API boundary for agent execution.

Replace ``run_task`` with a call to the shared agent-core once its public
execution contract is available. Keeping the fallback here prevents UI code
from carrying demo-specific behavior.
"""

from datetime import datetime, timezone
from uuid import uuid4


def run_task(prompt: str) -> dict:
    """Return a deterministic demo run compatible with the frontend contract."""
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(uuid4())
    return {
        "run_id": run_id,
        "mode": "demo_adapter",
        "result": (
            "FalconLLM's agent adapter received your task. "
            "The shared agent-core can replace this adapter without changing the UI.\n\n"
            f"Task: {prompt}"
        ),
        "trace": [
            {
                "step": 1,
                "action": "validate_task",
                "status": "completed",
                "input": prompt,
                "output": "Task accepted by the API adapter.",
                "timestamp": started_at,
            },
            {
                "step": 2,
                "action": "agent_execution",
                "status": "completed",
                "input": "Dispatch to configured agent implementation.",
                "output": "Demo adapter response returned; agent-core integration pending.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ],
    }
