import os
from dotenv import load_dotenv

load_dotenv()


class AgentConfig:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    MAX_AGENT_ITERATIONS = int(
        os.getenv("MAX_AGENT_ITERATIONS", "10")
    )

    LLM_TIMEOUT = int(
        os.getenv("LLM_TIMEOUT", "30")
    )
