from backend.agent.config import AgentConfig
from backend.agent.llm_client import BaseLLMClient


def create_llm(config: AgentConfig | None = None) -> BaseLLMClient:
    config = config or AgentConfig()
    provider = config.LLM_PROVIDER.strip().lower()
    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing. Add it to .env.")
        from backend.agent.gemini_client import GeminiLLMClient

        return GeminiLLMClient(
            api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            timeout=config.LLM_TIMEOUT,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}. Only gemini is implemented.")
