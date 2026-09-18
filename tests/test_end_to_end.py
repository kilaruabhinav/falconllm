"""High-level agent flows over the production tool registry."""

import pytest

from backend.agent.engine import AgentEngine
from backend.agent.mock_llm import MockLLMClient
from backend.tools.calculator_tool import CalculatorTool
from backend.tools.providers import SearchProvider
from backend.tools.registry import ToolRegistry
from backend.tools.search_tool import SearchTool


class PriceSearchProvider(SearchProvider):
    def search(self, query, max_results):
        return [{"title": "Product", "content": "Current price: 125", "url": "https://example.test"}]


def registry(provider=None):
    result = ToolRegistry()
    result.register(SearchTool(provider))
    result.register(CalculatorTool())
    return result


@pytest.mark.asyncio
async def test_search_then_calculator_flow_uses_llm_selected_tools():
    llm = MockLLMClient([
        {"type": "tool", "plan": "Find the current price.", "tool": "search", "arguments": {"query": "product price"}},
        {"type": "tool", "plan": "Apply the requested discount.", "tool": "calculator", "arguments": {"expression": "125 * 0.8"}},
        {"type": "final", "plan": "The discounted price is ready.", "answer": "The discounted price is 100."},
    ])
    result = await AgentEngine(llm, registry(PriceSearchProvider())).run("Find the price and discount it")
    assert result.status == "completed"
    assert result.answer == "The discounted price is 100."
    calls = [step.tool for step in result.trace if step.type == "tool_call"]
    assert calls == ["search", "calculator"]
    assert any(step.type == "tool_result" and step.result == 100 for step in result.trace)


@pytest.mark.asyncio
async def test_unconfigured_search_failure_is_observed_then_recovered():
    llm = MockLLMClient([
        {"type": "tool", "plan": "Try the configured search.", "tool": "search", "arguments": {"query": "price"}},
        {"type": "tool", "plan": "Use the known local value instead.", "tool": "calculator", "arguments": {"expression": "50 * 2"}},
        {"type": "final", "plan": "An alternate tool produced the answer.", "answer": "The result is 100; live search was unavailable."},
    ])
    result = await AgentEngine(llm, registry()).run("Search, then recover if unavailable")
    assert result.status == "completed"
    assert any(step.type == "tool_error" and "No search provider" in step.error for step in result.trace)
    assert any(step.type == "recovery" for step in result.trace)
    assert any("No search provider is configured" in message["content"] for message in llm.calls[1]["messages"])
