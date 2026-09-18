"""Safe, provider-independent failure taxonomy for LLM transports."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ProviderErrorCode(str, Enum):
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    AUTH_ERROR = "AUTH_ERROR"
    TIMEOUT = "TIMEOUT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    UNKNOWN_PROVIDER_ERROR = "UNKNOWN_PROVIDER_ERROR"
    COOLDOWN = "COOLDOWN"


class LLMProviderFailure(Exception):
    """A sanitized provider failure suitable for routing and tracing."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        code: ProviderErrorCode,
        message: str,
        retryable: bool,
        fallback_allowed: bool = True,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.code = code
        self.message = message
        self.retryable = retryable
        self.fallback_allowed = fallback_allowed
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "fallback_allowed": self.fallback_allowed,
            "status_code": self.status_code,
        }


def status_code_from_exception(exc: Exception) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(value, int):
            return value
    return None


def classify_provider_exception(
    exc: Exception, *, provider: str, model: str
) -> LLMProviderFailure:
    """Classify common HTTP/provider failures without exposing raw responses."""
    status = status_code_from_exception(exc)
    name = type(exc).__name__.lower()
    raw = str(exc).lower()
    if status == 429:
        quota = "quota" in raw or "resource_exhausted" in raw
        code = ProviderErrorCode.QUOTA_EXCEEDED if quota else ProviderErrorCode.RATE_LIMITED
        message = "Provider quota is exhausted." if quota else "Provider rate limit reached."
        return LLMProviderFailure(provider=provider, model=model, code=code, message=message, retryable=False, status_code=status)
    if status in {401, 403} or "authentication" in name or "permissiondenied" in name:
        return LLMProviderFailure(provider=provider, model=model, code=ProviderErrorCode.AUTH_ERROR, message="Provider authentication failed.", retryable=False, status_code=status)
    if status == 404 or "notfound" in name:
        return LLMProviderFailure(provider=provider, model=model, code=ProviderErrorCode.MODEL_UNAVAILABLE, message="Provider model is unavailable.", retryable=False, status_code=status)
    if status in {408, 504} or "timeout" in name:
        return LLMProviderFailure(provider=provider, model=model, code=ProviderErrorCode.TIMEOUT, message="Provider request timed out.", retryable=True, status_code=status)
    if status in {500, 502, 503} or "connection" in name or "unavailable" in raw:
        return LLMProviderFailure(provider=provider, model=model, code=ProviderErrorCode.SERVICE_UNAVAILABLE, message="Provider service is unavailable.", retryable=True, status_code=status)
    return LLMProviderFailure(provider=provider, model=model, code=ProviderErrorCode.UNKNOWN_PROVIDER_ERROR, message="Provider request failed.", retryable=False, status_code=status)
