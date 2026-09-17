from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLLMClient(ABC):

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> str:
        pass
