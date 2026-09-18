"""Validate the LLM fallback chain without exposing credentials."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent.config import AgentConfig
from backend.agent.fallback_llm import FallbackLLMClient
from backend.agent.llm_client import BaseLLMClient
from backend.agent.openai_client import OpenAILLMClient
from backend.agent.parser import parse_agent_response
from backend.agent.provider_errors import LLMProviderFailure, ProviderErrorCode


class SimulatedClient(BaseLLMClient):
    def __init__(self, provider: str, model: str, response=None, error=None):
        self.provider = provider
        self.model = model
        self.response = response
        self.error = error

    async def generate(self, messages, tools):
        if self.error:
            raise self.error
        return self.response


def quota(model: str) -> LLMProviderFailure:
    return LLMProviderFailure(
        provider="gemini",
        model=model,
        code=ProviderErrorCode.QUOTA_EXCEEDED,
        message="Simulated Gemini quota exhaustion.",
        retryable=False,
    )


def print_configuration(config: AgentConfig) -> None:
    print(f"Gemini primary configured: {'YES' if config.GEMINI_API_KEY else 'NO'}")
    print(f"Gemini fallback configured: {'YES' if config.GEMINI_API_KEY else 'NO'}")
    print(f"OpenAI configured: {'YES' if config.OPENAI_API_KEY else 'NO'}")
    print(f"Mode: {config.LLM_PROVIDER_MODE}")
    print("Configured chain:")
    if config.LLM_PROVIDER_MODE in {"gemini", "fallback"} and config.GEMINI_API_KEY:
        print(f"  gemini/{config.GEMINI_PRIMARY_MODEL}")
        if config.LLM_PROVIDER_MODE == "fallback":
            print(f"  gemini/{config.GEMINI_FALLBACK_MODEL}")
    if config.LLM_PROVIDER_MODE in {"openai", "fallback"} and config.OPENAI_API_KEY:
        print(f"  openai/{config.OPENAI_MODEL}")


async def simulate(config: AgentConfig) -> int:
    clients = [
        SimulatedClient("gemini", config.GEMINI_PRIMARY_MODEL, error=quota(config.GEMINI_PRIMARY_MODEL)),
        SimulatedClient("gemini", config.GEMINI_FALLBACK_MODEL, error=quota(config.GEMINI_FALLBACK_MODEL)),
        SimulatedClient(
            "openai",
            config.OPENAI_MODEL,
            response='{"type":"final","plan":"Fallback completed.","answer":"42"}',
        ),
    ]
    events = []
    router = FallbackLLMClient(clients, cooldown_seconds=0)
    router.start_run("simulation", lambda event, data: events.append((event, data)))
    action = parse_agent_response(await router.generate([], []))
    fallbacks = [data for event, data in events if event == "LLM_FALLBACK"]
    print("Simulation: Gemini primary → Gemini fallback → OpenAI")
    print(f"Fallback events: {len(fallbacks)}")
    print(f"Final provider: {events[-1][1]['provider']}")
    print(f"Valid AgentAction: {'PASS' if action.type == 'final' else 'FAIL'}")
    return 0 if action.type == "final" and len(fallbacks) == 2 else 1


async def live_openai(config: AgentConfig) -> int:
    if not config.OPENAI_API_KEY:
        print("Live OpenAI: SKIPPED (OPENAI_API_KEY is not configured)")
        return 0
    client = OpenAILLMClient(config.OPENAI_API_KEY, config.OPENAI_MODEL, config.LLM_TIMEOUT)
    try:
        raw = await client.generate(
            [
                {
                    "role": "system",
                    "content": "Return one valid AgentAction JSON object. Use a concise operational plan and do not expose hidden reasoning.",
                },
                {"role": "user", "content": "What is 18 + 24? Answer directly."},
            ],
            [],
        )
        action = parse_agent_response(raw)
        print(f"Live OpenAI ({config.OPENAI_MODEL}): PASS ({action.type})")
        return 0
    except LLMProviderFailure as exc:
        status = f", HTTP {exc.status_code}" if exc.status_code else ""
        print(f"Live OpenAI ({config.OPENAI_MODEL}): FAIL ({exc.code.value}{status})")
        return 1
    except Exception as exc:
        print(f"Live OpenAI ({config.OPENAI_MODEL}): FAIL ({type(exc).__name__})")
        return 1
    finally:
        await client.aclose()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate-gemini-429", action="store_true")
    parser.add_argument("--live-openai", action="store_true")
    args = parser.parse_args()
    config = AgentConfig()
    print_configuration(config)
    if args.simulate_gemini_429:
        return await simulate(config)
    if args.live_openai:
        return await live_openai(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
