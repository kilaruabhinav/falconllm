"""Construct the production tool registry from application configuration."""

from pathlib import Path

from backend.agent.config import AgentConfig

from .calculator_tool import CalculatorTool
from .file_tool import FileReaderTool
from .providers import TavilySearchProvider
from .registry import ToolRegistry
from .search_tool import SearchTool


def create_tool_registry(
    config: AgentConfig | None = None, file_root: str | Path = "data"
) -> ToolRegistry:
    config = config or AgentConfig()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileReaderTool(root_directory=file_root))
    provider = None
    if config.SEARCH_PROVIDER in {"tavily", ""} and config.TAVILY_API_KEY:
        provider = TavilySearchProvider(api_key=config.TAVILY_API_KEY)
    registry.register(SearchTool(provider))
    return registry
