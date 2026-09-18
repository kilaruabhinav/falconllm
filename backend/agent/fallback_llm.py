"""Monotonic provider routing with run-scoped failover and process cooldowns."""

from __future__ import annotations

import time
import threading
from collections.abc import Callable
from typing import Any

from backend.agent.llm_client import BaseLLMClient, LLMEventHandler
from backend.agent.provider_errors import (
    LLMProviderFailure,
    ProviderErrorCode,
    classify_provider_exception,
)


class FallbackLLMClient(BaseLLMClient):
    """Try an ordered provider/model chain without ever moving backwards."""

    _cooldowns: dict[str, float] = {}
    _cooldown_lock = threading.Lock()

    def __init__(
        self,
        clients: list[BaseLLMClient],
        *,
        cooldown_seconds: float = 60,
        max_retries: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not clients:
            raise ValueError("At least one LLM provider must be configured.")
        self.clients = clients
        self.cooldown_seconds = cooldown_seconds
        self.max_retries = max_retries
        self.clock = clock
        self._current_index = 0
        self._run_unavailable: set[str] = set()
        self._event_handler: LLMEventHandler | None = None
        self.run_id: str | None = None

    @property
    def chain(self) -> list[str]:
        return [self._identity(client) for client in self.clients]

    def start_run(self, run_id: str, event_handler: LLMEventHandler | None = None) -> None:
        self.run_id = run_id
        self._current_index = 0
        self._run_unavailable = set()
        self._event_handler = event_handler

    def end_run(self) -> None:
        self.run_id = None
        self._event_handler = None
        self._run_unavailable = set()
        self._current_index = 0

    async def generate(self, messages: list[dict], tools: list[dict]) -> str:
        last_error: LLMProviderFailure | None = None
        previous_identity: str | None = None
        for index in range(self._current_index, len(self.clients)):
            client = self.clients[index]
            identity = self._identity(client)
            if identity in self._run_unavailable or self._in_cooldown(identity):
                error = LLMProviderFailure(
                    provider=self._provider(client),
                    model=self._model(client),
                    code=ProviderErrorCode.COOLDOWN,
                    message="Provider/model is temporarily in cooldown.",
                    retryable=False,
                )
                self._emit("LLM_PROVIDER_ERROR", error.to_dict())
                last_error = error
                if index + 1 < len(self.clients):
                    self._emit_fallback(identity, self._identity(self.clients[index + 1]), error.code)
                previous_identity = identity
                continue

            if previous_identity is not None:
                self._current_index = index
            retries = 0
            while True:
                self._emit(
                    "LLM_PROVIDER_SELECTED",
                    {"provider": self._provider(client), "model": self._model(client)},
                )
                attempt_started = self.clock()
                try:
                    result = await client.generate(messages, tools)
                    self._current_index = index
                    return result
                except LLMProviderFailure as exc:
                    error = exc
                except Exception as exc:
                    error = classify_provider_exception(
                        exc, provider=self._provider(client), model=self._model(client)
                    )

                error_data = error.to_dict()
                error_data["duration_ms"] = max(0, (self.clock() - attempt_started) * 1000)
                self._emit("LLM_PROVIDER_ERROR", error_data)
                last_error = error
                can_retry = (
                    error.retryable
                    and error.code in {ProviderErrorCode.TIMEOUT, ProviderErrorCode.SERVICE_UNAVAILABLE}
                    and retries < self.max_retries
                )
                if can_retry:
                    retries += 1
                    continue

                self._run_unavailable.add(identity)
                self._set_cooldown(identity)
                if not error.fallback_allowed or index + 1 >= len(self.clients):
                    raise error
                next_identity = self._identity(self.clients[index + 1])
                self._emit_fallback(identity, next_identity, error.code)
                previous_identity = identity
                break

        if last_error is not None:
            raise last_error
        raise LLMProviderFailure(
            provider="router",
            model="none",
            code=ProviderErrorCode.MODEL_UNAVAILABLE,
            message="No configured LLM provider is available.",
            retryable=False,
            fallback_allowed=False,
        )

    async def aclose(self) -> None:
        for client in self.clients:
            await client.aclose()

    def spawn_router(self) -> FallbackLLMClient:
        """Create isolated per-run routing state while reusing SDK transports."""
        return FallbackLLMClient(
            self.clients,
            cooldown_seconds=self.cooldown_seconds,
            max_retries=self.max_retries,
            clock=self.clock,
        )

    def _in_cooldown(self, identity: str) -> bool:
        with self._cooldown_lock:
            expires_at = self._cooldowns.get(identity, 0)
            if expires_at <= self.clock():
                self._cooldowns.pop(identity, None)
                return False
            return True

    def _set_cooldown(self, identity: str) -> None:
        if self.cooldown_seconds > 0:
            with self._cooldown_lock:
                self._cooldowns[identity] = self.clock() + self.cooldown_seconds

    def _emit_fallback(
        self, source: str, destination: str, code: ProviderErrorCode
    ) -> None:
        self._emit(
            "LLM_FALLBACK",
            {"from": source, "to": destination, "reason": code.value},
        )

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self._event_handler is not None:
            self._event_handler(event_type, data)

    @staticmethod
    def _provider(client: BaseLLMClient) -> str:
        return str(getattr(client, "provider", type(client).__name__)).lower()

    @staticmethod
    def _model(client: BaseLLMClient) -> str:
        return str(getattr(client, "model", "unknown"))

    @classmethod
    def _identity(cls, client: BaseLLMClient) -> str:
        return f"{cls._provider(client)}/{cls._model(client)}"

    @classmethod
    def clear_cooldowns(cls) -> None:
        """Test/operations hook for clearing process-local breaker state."""
        with cls._cooldown_lock:
            cls._cooldowns.clear()
