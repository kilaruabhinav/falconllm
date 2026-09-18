from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.agent.openai_client import OpenAILLMClient
from backend.agent.provider_errors import LLMProviderFailure, ProviderErrorCode
from backend.agent.schemas import AgentAction


@pytest.fixture
def sdk(monkeypatch):
    fake = Mock()
    fake.responses.parse = AsyncMock(
        return_value=SimpleNamespace(
            output_parsed=AgentAction(
                type="final", plan="Enough information is available.", answer="42"
            )
        )
    )
    fake.close = AsyncMock()
    constructor = Mock(return_value=fake)
    monkeypatch.setattr("backend.agent.openai_client.AsyncOpenAI", constructor)
    return fake, constructor


@pytest.mark.asyncio
async def test_openai_responses_structured_output_contract(sdk):
    fake, constructor = sdk
    client = OpenAILLMClient("test-key", "gpt-5-mini", timeout=4)
    output = await client.generate(
        [
            {"role": "system", "content": "Return an AgentAction."},
            {"role": "user", "content": "What is 40+2?"},
        ],
        [{"name": "calculator"}],
    )
    assert '"answer":"42"' in output
    assert constructor.call_args.kwargs["max_retries"] == 0
    kwargs = fake.responses.parse.call_args.kwargs
    assert kwargs["model"] == "gpt-5-mini"
    assert kwargs["text_format"] is AgentAction
    assert kwargs["store"] is False
    assert kwargs["instructions"] == "Return an AgentAction."
    await client.aclose()
    fake.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_rate_limit_is_sanitized(sdk):
    fake, _ = sdk
    error = RuntimeError("raw secret response")
    error.status_code = 429
    fake.responses.parse.side_effect = error
    with pytest.raises(LLMProviderFailure) as caught:
        await OpenAILLMClient("test-key", "gpt-5-mini").generate([], [])
    assert caught.value.code == ProviderErrorCode.RATE_LIMITED
    assert "secret" not in str(caught.value)
