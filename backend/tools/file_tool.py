"""Safe, bounded reading of UTF-8 text files beneath one configured root."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .base_tool import BaseTool, ToolResult


class FileReaderTool(BaseTool):
    name = "file_reader"
    description = "Read a UTF-8 text document from the configured data directory."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path beneath the configured data directory."}
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    _ALLOWED_SUFFIXES = frozenset({".txt", ".md", ".csv", ".json", ".log"})

    def __init__(self, root_directory: str | Path = "data", max_bytes: int = 1_000_000) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        self.root_directory = Path(root_directory).resolve()
        self.max_bytes = max_bytes

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        value = arguments.get("path")
        if not isinstance(value, str) or not value.strip():
            return ToolResult(success=False, tool=self.name, error="File path must be a non-empty string.")
        supplied_path = Path(value)
        if supplied_path.is_absolute():
            return ToolResult(success=False, tool=self.name, error="File path must be relative to the data directory.")

        try:
            target = (self.root_directory / supplied_path).resolve()
            if not target.is_relative_to(self.root_directory):
                return ToolResult(success=False, tool=self.name, error="Access to this file is not allowed.")
            if not target.exists():
                return ToolResult(success=False, tool=self.name, error="File does not exist.")
            if not target.is_file():
                return ToolResult(success=False, tool=self.name, error="The path is not a file.")
            if target.suffix.lower() not in self._ALLOWED_SUFFIXES:
                return ToolResult(success=False, tool=self.name, error="Unsupported file type.")
            if target.stat().st_size > self.max_bytes:
                return ToolResult(success=False, tool=self.name, error="File is too large.")
            content = target.read_text(encoding="utf-8")
            if not content.strip():
                return ToolResult(success=False, tool=self.name, error="File is empty.")
            return ToolResult(success=True, tool=self.name, result=content)
        except UnicodeDecodeError:
            return ToolResult(success=False, tool=self.name, error="File is not valid UTF-8 text.")
        except OSError:
            return ToolResult(success=False, tool=self.name, error="File could not be read.")
