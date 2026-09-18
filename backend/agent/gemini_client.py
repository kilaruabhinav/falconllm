"""Direct Gemini transport. Agent orchestration belongs in AgentEngine."""
import asyncio

from google import genai
from google.genai import types

from backend.agent.config import DEFAULT_GEMINI_MODEL
from backend.agent.llm_client import BaseLLMClient


class GeminiLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL, timeout: float = 30):
        if not api_key or not api_key.strip():
            raise ValueError("GEMINI_API_KEY is missing. Add it to .env.")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )
        self.model = model
        self.timeout = timeout

    async def generate(self, messages: list[dict], tools: list[dict]) -> str:
        # Tool definitions are injected by Planner, not executed by the SDK.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages if m["role"] != "system"
        ]
        try:
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Gemini request timed out after {self.timeout:g} seconds.") from None
        except Exception as exc:
            # SDK exceptions can contain request details. Never put them in traces.
            code = getattr(exc, "code", None)
            label = f" (HTTP {code})" if isinstance(code, int) else ""
            raise RuntimeError(
                f"Gemini request failed{label}. Check API key, model access, quota, and network."
            ) from None
        if not response.text or not response.text.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return response.text

    async def aclose(self):
        await self.client.aio.aclose()
        self.client.close()
