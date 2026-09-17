import json
from typing import Any, Dict, List

from backend.agent.llm_client import BaseLLMClient


class MockLLMClient(BaseLLMClient):

    def __init__(
        self,
        responses: List[Dict[str, Any]],
    ):
        self.responses = responses
        self.call_count = 0

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> str:

        if self.call_count >= len(
            self.responses
        ):
            raise RuntimeError(
                "Mock LLM has no more responses."
            )

        response = self.responses[
            self.call_count
        ]

        self.call_count += 1

        return json.dumps(response)
