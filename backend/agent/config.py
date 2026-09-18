"""Runtime settings; environment variables take precedence over the root .env."""
import math
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class AgentConfig:
    def __init__(self, env_file: Path | str | None = ENV_PATH):
        if env_file is not None:
            load_dotenv(env_file, override=False)
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        self.SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "").strip().lower()
        self.SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "").strip()
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", self.SEARCH_API_KEY).strip()
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///agent.db").strip()
        self.BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1").strip()
        self.BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
        if not self.GEMINI_MODEL:
            raise ValueError("GEMINI_MODEL must not be empty.")
        try:
            self.MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
            self.LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
        except ValueError:
            raise ValueError("MAX_AGENT_ITERATIONS must be an integer and LLM_TIMEOUT a number.") from None
        if self.MAX_AGENT_ITERATIONS < 1:
            raise ValueError("MAX_AGENT_ITERATIONS must be positive.")
        if not math.isfinite(self.LLM_TIMEOUT) or self.LLM_TIMEOUT <= 0:
            raise ValueError("LLM_TIMEOUT must be a finite positive number of seconds.")
