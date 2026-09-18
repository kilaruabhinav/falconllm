from backend.agent.config import AgentConfig
from backend.agent.fallback_llm import FallbackLLMClient
from backend.agent.llm_client import BaseLLMClient


def create_llm(config: AgentConfig | None = None) -> BaseLLMClient:
    config = config or AgentConfig()
    mode = config.LLM_PROVIDER_MODE
    clients: list[BaseLLMClient] = []

    if mode in {"gemini", "fallback"} and config.GEMINI_API_KEY:
        from backend.agent.gemini_client import GeminiLLMClient
        models = [config.GEMINI_PRIMARY_MODEL]
        if mode == "fallback" and config.GEMINI_FALLBACK_MODEL not in models:
            models.append(config.GEMINI_FALLBACK_MODEL)
        clients.extend(
            GeminiLLMClient(
                api_key=config.GEMINI_API_KEY,
                model=model,
                timeout=config.LLM_TIMEOUT,
            )
            for model in models
        )

    if mode in {"openai", "fallback"} and config.OPENAI_API_KEY:
        from backend.agent.openai_client import OpenAILLMClient
        clients.append(
            OpenAILLMClient(
                api_key=config.OPENAI_API_KEY,
                model=config.OPENAI_MODEL,
                timeout=config.LLM_TIMEOUT,
            )
        )

    if not clients:
        required = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "fallback": "GEMINI_API_KEY or OPENAI_API_KEY",
        }[mode]
        raise ValueError(f"No LLM provider is configured. Set {required} in .env.")

    return FallbackLLMClient(
        clients,
        cooldown_seconds=config.LLM_PROVIDER_COOLDOWN_SECONDS,
        max_retries=config.MAX_LLM_RETRIES,
    )
