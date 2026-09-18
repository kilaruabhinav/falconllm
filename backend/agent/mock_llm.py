import json
from copy import deepcopy
from typing import Any, Dict, List

from backend.agent.llm_client import BaseLLMClient


class MockLLMClient(BaseLLMClient):

    def __init__(
        self,
        responses: List[Dict[str, Any] | str | Exception],
    ):
        self.responses = responses
        self.call_count = 0
        self.calls = []

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> str:

        self.calls.append(deepcopy({"messages": messages, "tools": tools}))
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

        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)
