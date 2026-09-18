"""Provider routing, circuit breaker, tracing, and state-preservation tests."""

from __future__ import annotations

import pytest

from backend.agent.config import AgentConfig, DEFAULT_OPENAI_MODEL
from backend.agent.engine import AgentEngine
from backend.agent.fallback_llm import FallbackLLMClient
from backend.agent.llm_client import BaseLLMClient
from backend.agent.llm_factory import create_llm
from backend.agent.provider_errors import LLMProviderFailure, ProviderErrorCode
from backend.tools.mock_registry import MockToolRegistry


class StubClient(BaseLLMClient):
    def __init__(self, provider, model, responses):
        self.provider = provider
        self.model = model
        self.responses = list(responses)
        self.calls = []

    async def generate(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def failure(provider, model, code=ProviderErrorCode.QUOTA_EXCEEDED):
    return LLMProviderFailure(
        provider=provider,
        model=model,
        code=code,
        message=f"{provider} unavailable",
        retryable=False,
    )


@pytest.fixture(autouse=True)
def clear_breaker():
    FallbackLLMClient.clear_cooldowns()
    yield
    FallbackLLMClient.clear_cooldowns()


@pytest.mark.asyncio
async def test_primary_success_does_not_call_fallback():
    primary = StubClient("gemini", "primary", ['{"type":"final","plan":"done","answer":"ok"}'])
    openai = StubClient("openai", "gpt", ['{"type":"final","plan":"done","answer":"unused"}'])
    router = FallbackLLMClient([primary, openai])
    assert "ok" in await router.generate([], [])
    assert len(primary.calls) == 1
    assert openai.calls == []


@pytest.mark.asyncio
async def test_primary_429_calls_gemini_lite():
    primary = StubClient("gemini", "primary", [failure("gemini", "primary")])
    lite = StubClient("gemini", "lite", ['{"type":"final","plan":"done","answer":"lite"}'])
    router = FallbackLLMClient([primary, lite])
    assert "lite" in await router.generate([], [])
    assert len(lite.calls) == 1


@pytest.mark.asyncio
async def test_two_gemini_quota_errors_reach_openai_and_emit_events():
    primary = StubClient("gemini", "primary", [failure("gemini", "primary")])
    lite = StubClient("gemini", "lite", [failure("gemini", "lite")])
    openai = StubClient("openai", "gpt", ['{"type":"final","plan":"done","answer":"openai"}'])
    events = []
    router = FallbackLLMClient([primary, lite, openai])
    router.start_run("run", lambda event, data: events.append((event, data)))
    assert "openai" in await router.generate([], [])
    assert [event for event, _ in events].count("LLM_FALLBACK") == 2
    assert events[-1][0] == "LLM_PROVIDER_SELECTED"
    assert events[-1][1]["provider"] == "openai"


@pytest.mark.asyncio
async def test_all_providers_fail_with_clean_error():
    clients = [
        StubClient("gemini", "primary", [failure("gemini", "primary")]),
        StubClient("openai", "gpt", [failure("openai", "gpt")]),
    ]
    with pytest.raises(LLMProviderFailure) as caught:
        await FallbackLLMClient(clients).generate([], [])
    assert caught.value.provider == "openai"
    assert "key" not in str(caught.value).lower()


def config(monkeypatch, **values):
    for name in ["GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_MODEL", "LLM_PROVIDER_MODE"]:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return AgentConfig(None)


def test_factory_skips_missing_provider_keys(monkeypatch):
    openai_only = create_llm(config(monkeypatch, LLM_PROVIDER_MODE="fallback", OPENAI_API_KEY="test"))
    assert openai_only.chain == [f"openai/{DEFAULT_OPENAI_MODEL}"]
    gemini_only = create_llm(config(monkeypatch, LLM_PROVIDER_MODE="fallback", GEMINI_API_KEY="test"))
    assert len(gemini_only.chain) == 2
    assert all(item.startswith("gemini/") for item in gemini_only.chain)
    with pytest.raises(ValueError, match="No LLM provider"):
        create_llm(config(monkeypatch, LLM_PROVIDER_MODE="fallback"))


@pytest.mark.asyncio
async def test_cooldown_skips_exhausted_model_on_next_run():
    now = [100.0]
    primary = StubClient("gemini", "primary", [failure("gemini", "primary")])
    fallback = StubClient("openai", "gpt", [
        '{"type":"final","plan":"done","answer":"one"}',
        '{"type":"final","plan":"done","answer":"two"}',
    ])
    router = FallbackLLMClient([primary, fallback], cooldown_seconds=60, clock=lambda: now[0])
    router.start_run("one")
    await router.generate([], [])
    router.start_run("two")
    await router.generate([], [])
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 2


@pytest.mark.asyncio
async def test_state_and_tool_observation_survive_provider_switch():
    primary = StubClient("gemini", "primary", [
        '{"type":"tool","plan":"calculate","tool":"calculator","arguments":{"expression":"40+2"}}',
        failure("gemini", "primary"),
    ])
    openai = StubClient("openai", "gpt", [
        '{"type":"final","plan":"use observation","answer":"42"}',
    ])
    router = FallbackLLMClient([primary, openai])
    result = await AgentEngine(router, MockToolRegistry()).run("What is 40+2?")
    assert result.answer == "42"
    assert any("Result: 42" in message["content"] for message in openai.calls[0]["messages"])
    assert any(step.type == "llm_fallback" for step in result.trace)
    assert any(step.type == "tool_result" and step.result == 42 for step in result.trace)


@pytest.mark.asyncio
async def test_malformed_json_stays_with_primary_parser_recovery():
    primary = StubClient("gemini", "primary", [
        "not-json",
        '{"type":"final","plan":"fixed","answer":"ok"}',
    ])
    fallback = StubClient("openai", "gpt", ['{"type":"final","plan":"unused","answer":"bad"}'])
    result = await AgentEngine(FallbackLLMClient([primary, fallback]), MockToolRegistry()).run("test")
    assert result.answer == "ok"
    assert len(primary.calls) == 2
    assert fallback.calls == []
    assert any(step.type == "parse_error" for step in result.trace)


def test_spawned_routers_reuse_clients_but_isolate_run_state():
    client = StubClient("gemini", "primary", [])
    template = FallbackLLMClient([client], cooldown_seconds=7, max_retries=2)
    first = template.spawn_router()
    second = template.spawn_router()
    assert first is not second
    assert first.clients[0] is client
    assert second.clients[0] is client
    first.start_run("one")
    second.start_run("two")
    assert first.run_id == "one"
    assert second.run_id == "two"
