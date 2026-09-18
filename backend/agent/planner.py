import json

from typing import Any, Dict, List

from backend.agent.prompts import AGENT_SYSTEM_PROMPT


class Planner:
    MAX_HISTORY_ITEMS = 8
    MAX_OBSERVATION_CHARS = 2_400

    def build_messages(
        self,
        user_query: str,
        history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        iteration: int,
    ) -> List[Dict[str, Any]]:

        tool_text = self._format_tools(
            available_tools
        )

        messages = [
            {
                "role": "system",
                "content": (
                    AGENT_SYSTEM_PROMPT
                    + "\n\nAVAILABLE TOOLS:\n"
                    + tool_text
                ),
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]

        for item in history[-self.MAX_HISTORY_ITEMS:]:

            messages.append(
                {
                    "role": "assistant" if item.get("type") == "action" else "user",
                    "content": self._format_history_item(
                        item
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Current iteration: {iteration}. "
                    "Decide the next action. "
                    "Return only valid JSON."
                ),
            }
        )

        return messages

    def _format_tools(
        self,
        tools: List[Dict[str, Any]],
    ) -> str:

        if not tools:
            return "No tools available."

        return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))

    def _format_history_item(
        self,
        item: Dict[str, Any],
    ) -> str:

        item_type = item.get(
            "type",
            "observation"
        )

        if item_type == "action":
            return json.dumps(item["action"], ensure_ascii=False)

        if item_type == "tool_result":
            result = json.dumps(item.get("result"), ensure_ascii=False, default=str)
            if len(result) > self.MAX_OBSERVATION_CHARS:
                result = result[: self.MAX_OBSERVATION_CHARS] + "... [TRUNCATED]"
            return (
                "TOOL OBSERVATION:\n"
                f"Tool: {item.get('tool')}\n"
                f"Success: {item.get('success')}\n"
                f"Result: {result}\n"
                f"Error: {item.get('error')}"
            )

        if item_type == "error":

            return (
                "FRAMEWORK ERROR:\n"
                f"{item.get('error')}"
            )

        return str(item)
