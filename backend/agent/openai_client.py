"""OpenAI Responses API transport; orchestration remains in AgentEngine."""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from backend.agent.llm_client import BaseLLMClient
from backend.agent.provider_errors import (
    LLMProviderFailure,
    ProviderErrorCode,
    classify_provider_exception,
)
from backend.agent.schemas import AgentAction


class OpenAILLMClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, api_key: str, model: str, timeout: float = 30) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env.")
        if not model or not model.strip():
            raise ValueError("OPENAI_MODEL must not be empty.")
        self.model = model.strip()
        self.timeout = timeout
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    async def generate(self, messages: list[dict], tools: list[dict]) -> str:
        instructions = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        inputs = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message["role"] != "system"
        ]
        try:
            response = await asyncio.wait_for(
                self.client.responses.parse(
                    model=self.model,
                    instructions=instructions,
                    input=inputs,
                    text_format=AgentAction,
                    store=False,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise LLMProviderFailure(
                provider=self.provider,
                model=self.model,
                code=ProviderErrorCode.TIMEOUT,
                message="OpenAI request timed out.",
                retryable=True,
            ) from None
        except Exception as exc:
            raise classify_provider_exception(
                exc, provider=self.provider, model=self.model
            ) from None
        action = response.output_parsed
        if action is None:
            raise LLMProviderFailure(
                provider=self.provider,
                model=self.model,
                code=ProviderErrorCode.UNKNOWN_PROVIDER_ERROR,
                message="OpenAI returned no structured action.",
                retryable=False,
            )
        return action.model_dump_json(exclude_none=True)

    async def aclose(self) -> None:
        await self.client.close()
