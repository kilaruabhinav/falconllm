import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.agent.config import AgentConfig, DEFAULT_GEMINI_MODEL
from backend.agent.gemini_client import GeminiLLMClient
from backend.agent.llm_factory import create_llm
from backend.agent.parser import AgentParseError, parse_agent_response
from backend.tools.mock_registry import MockToolRegistry


@pytest.mark.parametrize("raw", ["", " ", "[]", "null", '{"type":"other"}',
    '{"type":"final","plan":"done","answer":""}',
    '{"type":"tool","plan":"calculate","tool":"calculator"}',
    '{"type":"tool","plan":"calculate","tool":"calculator","arguments":[],"answer":"42"}',
    '{"type":"final","plan":"done","answer":"42","tool":"calculator"}',
    '{"type":"final","plan":"done","answer":"42","extra":true}',
    '```json\n{"type":"final","plan":"done","answer":"42"}',
])
def test_parser_rejects_invalid_actions(raw):
    with pytest.raises(AgentParseError):
        parse_agent_response(raw)


@pytest.mark.parametrize("fence", ["", "json", "JSON"])
def test_parser_accepts_fenced_json(fence):
    action = parse_agent_response(f'```{fence}\n{{"type":"final","plan":"done","answer":"42"}}\n```')
    assert action.answer == "42"


@pytest.fixture
def clean_env(monkeypatch):
    for name in ["GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "GEMINI_MODEL",
                 "LLM_TIMEOUT", "MAX_AGENT_ITERATIONS"]:
        monkeypatch.delenv(name, raising=False)


def test_config_reads_at_instantiation_and_env_overrides_dotenv(clean_env, monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=test-placeholder\nMAX_AGENT_ITERATIONS=3\n")
    monkeypatch.setenv("GEMINI_API_KEY", "environment-placeholder")
    config = AgentConfig(env)
    assert config.GEMINI_API_KEY == "environment-placeholder"
    assert config.MAX_AGENT_ITERATIONS == 3
    assert config.GEMINI_MODEL == DEFAULT_GEMINI_MODEL
    monkeypatch.setenv("MAX_AGENT_ITERATIONS", "4")
    assert AgentConfig(None).MAX_AGENT_ITERATIONS == 4


@pytest.mark.parametrize("name,value", [("MAX_AGENT_ITERATIONS", "0"), ("MAX_AGENT_ITERATIONS", "x"),
    ("LLM_TIMEOUT", "0"), ("LLM_TIMEOUT", "nan"), ("LLM_TIMEOUT", "inf"), ("GEMINI_MODEL", " ")])
def test_invalid_config(clean_env, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        AgentConfig(None)


def test_missing_key_and_unsupported_provider(clean_env, monkeypatch):
    with pytest.raises(ValueError, match="GEMINI_API_KEY is missing. Add it to .env."):
        create_llm(AgentConfig(None))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm(AgentConfig(None))


def test_real_cli_missing_key_exits_without_traceback():
    env = dict(os.environ, GEMINI_API_KEY="", LLM_PROVIDER="gemini", GEMINI_MODEL=DEFAULT_GEMINI_MODEL,
               MAX_AGENT_ITERATIONS="2", LLM_TIMEOUT="5")
    result = subprocess.run([sys.executable, "run_real_agent.py", "--query", "test"],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert "GEMINI_API_KEY is missing. Add it to .env." in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("expression,expected", [("40+2", 42), ("(6 * 7) / 2", 21), ("-2 + 4.5", 2.5)])
async def test_mock_calculator_arithmetic(expression, expected):
    result = await MockToolRegistry().execute("calculator", {"expression": expression})
    assert result.success and result.result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", ["__import__('os')", "2 ** 999999", "1 / 0", "True + 2", "1e309", "x.y"])
async def test_mock_calculator_rejects_unsafe_or_invalid_expression(expression):
    result = await MockToolRegistry().execute("calculator", {"expression": expression})
    assert not result.success and result.error


@pytest.fixture
def sdk(monkeypatch):
    fake = Mock()
    fake.aio.models.generate_content = AsyncMock(return_value=SimpleNamespace(text='{"type":"final"}'))
    fake.aio.aclose = AsyncMock()
    constructor = Mock(return_value=fake)
    monkeypatch.setattr("backend.agent.gemini_client.genai.Client", constructor)
    return fake, constructor


@pytest.mark.asyncio
async def test_gemini_async_sdk_contract(sdk):
    fake, constructor = sdk
    client = GeminiLLMClient("test-placeholder", timeout=2)
    response = await client.generate([
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "previous action"},
    ], [])
    assert json.loads(response)["type"] == "final"
    assert constructor.call_args.kwargs["http_options"].timeout == 2000
    kwargs = fake.aio.models.generate_content.call_args.kwargs
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].automatic_function_calling.disable is True
    assert kwargs["config"].system_instruction == "system prompt"
    assert [m.role for m in kwargs["contents"]] == ["user", "model"]
    await client.aclose()
    fake.aio.aclose.assert_awaited_once()
    fake.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,match", [(asyncio.TimeoutError(), "timed out"),
                                         (RuntimeError("secret-request-details"), "request failed")])
async def test_gemini_safe_errors(sdk, failure, match):
    fake, _ = sdk
    fake.aio.models.generate_content.side_effect = failure
    client = GeminiLLMClient("test-placeholder")
    with pytest.raises(RuntimeError, match=match) as caught:
        await client.generate([], [])
    assert "secret-request-details" not in str(caught.value)


@pytest.mark.asyncio
async def test_gemini_empty_response(sdk):
    fake, _ = sdk
    fake.aio.models.generate_content.return_value = SimpleNamespace(text=" ")
    with pytest.raises(RuntimeError, match="empty response"):
        await GeminiLLMClient("test-placeholder").generate([], [])
