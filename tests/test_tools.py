from pathlib import Path

import pytest

from backend.tools.calculator_tool import CalculatorTool
from backend.tools.file_tool import FileReaderTool
from backend.tools.providers import SearchProvider
from backend.tools.registry import ToolRegistry
from backend.tools.search_tool import SearchTool


class StubSearchProvider(SearchProvider):
    def search(self, query, max_results):
        return [{"title": "Result", "url": "https://example.test", "content": query, "limit": max_results}]


def test_calculator_success_and_safe_failures():
    tool = CalculatorTool()
    assert tool.execute({"expression": "(25 * 4) / 5"}).result == 20
    assert tool.execute({"expression": "10 / 0"}).error == "Cannot divide by zero."
    assert tool.execute({"expression": "__import__('os').system('bad')"}).success is False
    assert tool.execute({"expression": "2 ** 1001"}).success is False


def test_registry_discovery_and_structured_errors():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    assert registry.get_tool_descriptions()[0]["name"] == "calculator"
    assert registry.execute("calculator", {"expression": "2 + 2"})["result"] == 4
    assert registry.execute("missing", {})["error"] == "Unknown tool: missing"
    assert registry.execute("calculator", None)["error"] == "Tool arguments must be an object."
    with pytest.raises(ValueError, match="already registered"):
        registry.register(CalculatorTool())


def test_search_provider_adapter_and_no_provider_failure():
    result = SearchTool(StubSearchProvider()).execute({"query": "agents", "max_results": 1})
    assert result.success is True
    assert result.result[0]["content"] == "agents"
    assert SearchTool().execute({"query": "agents"}).success is False
    assert SearchTool(StubSearchProvider()).execute({"query": "agents", "max_results": 11}).success is False


def test_file_reader_success_and_required_failures(tmp_path: Path):
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"png")
    (tmp_path / "large.txt").write_text("abcdef", encoding="utf-8")
    tool = FileReaderTool(tmp_path, max_bytes=5)
    assert tool.execute({"path": "note.txt"}).result == "hello"
    assert tool.execute({"path": "missing.txt"}).error == "File does not exist."
    assert tool.execute({"path": "image.png"}).error == "Unsupported file type."
    assert tool.execute({"path": "empty.txt"}).error == "File is empty."
    assert tool.execute({"path": "large.txt"}).error == "File is too large."
    assert tool.execute({"path": "../outside.txt"}).error == "Access to this file is not allowed."
