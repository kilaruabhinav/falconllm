"""Runtime settings; environment variables take precedence over the root .env."""
import math
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class AgentConfig:
    def __init__(self, env_file: Path | str | None = ENV_PATH):
        if env_file is not None:
            load_dotenv(env_file, override=False)
        self.LLM_PROVIDER_MODE = os.getenv("LLM_PROVIDER_MODE", "fallback").strip().lower()
        self.LLM_PROVIDER = self.LLM_PROVIDER_MODE
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
        self.GEMINI_PRIMARY_MODEL = os.getenv(
            "GEMINI_PRIMARY_MODEL",
            os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        ).strip()
        self.GEMINI_FALLBACK_MODEL = os.getenv(
            "GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL
        ).strip()
        self.GEMINI_MODEL = self.GEMINI_PRIMARY_MODEL
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
        configured_openai_model = os.getenv("OPENAI_MODEL", "").strip()
        if configured_openai_model.upper() in {
            "YOUR_CHOSEN_OPENAI_MODEL",
            "YOUR_OPENAI_MODEL",
        }:
            configured_openai_model = ""
        self.OPENAI_MODEL = configured_openai_model or DEFAULT_OPENAI_MODEL
        self.SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "").strip().lower()
        self.SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "").strip()
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", self.SEARCH_API_KEY).strip()
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///agent.db").strip()
        self.BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1").strip()
        self.BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
        if self.LLM_PROVIDER_MODE not in {"gemini", "openai", "fallback"}:
            raise ValueError("LLM_PROVIDER_MODE must be gemini, openai, or fallback.")
        if not self.GEMINI_PRIMARY_MODEL or not self.GEMINI_FALLBACK_MODEL:
            raise ValueError("Gemini model names must not be empty.")
        if not self.OPENAI_MODEL:
            raise ValueError("OPENAI_MODEL must not be empty.")
        try:
            self.MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
            self.LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
            self.MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", "1"))
            self.LLM_PROVIDER_COOLDOWN_SECONDS = float(
                os.getenv("LLM_PROVIDER_COOLDOWN_SECONDS", "60")
            )
            self.SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "10"))
        except ValueError:
            raise ValueError("Agent/LLM limits contain an invalid number.") from None
        if self.MAX_AGENT_ITERATIONS < 1:
            raise ValueError("MAX_AGENT_ITERATIONS must be positive.")
        if not math.isfinite(self.LLM_TIMEOUT) or self.LLM_TIMEOUT <= 0:
            raise ValueError("LLM_TIMEOUT must be a finite positive number of seconds.")
        if self.MAX_LLM_RETRIES < 0:
            raise ValueError("MAX_LLM_RETRIES must not be negative.")
        if (
            not math.isfinite(self.LLM_PROVIDER_COOLDOWN_SECONDS)
            or self.LLM_PROVIDER_COOLDOWN_SECONDS < 0
        ):
            raise ValueError("LLM_PROVIDER_COOLDOWN_SECONDS must be finite and non-negative.")
        if not math.isfinite(self.SEARCH_TIMEOUT) or self.SEARCH_TIMEOUT <= 0:
            raise ValueError("SEARCH_TIMEOUT must be a finite positive number of seconds.")
