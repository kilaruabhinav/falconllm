from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import List, Dict, Any

LLMEventHandler = Callable[[str, dict[str, Any]], None]


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

    def start_run(self, run_id: str, event_handler: LLMEventHandler | None = None) -> None:
        """Bind optional run-scoped infrastructure tracing; transports may ignore it."""

    def end_run(self) -> None:
        """Release run-scoped routing state without closing the transport."""
