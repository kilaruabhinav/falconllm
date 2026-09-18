from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLLMClient(ABC):
    """Provider transport only; the engine owns decisions and tool execution."""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> str:
        """Return one textual next action for the supplied conversation."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release transport resources, if any. In-memory clients need no cleanup."""
