from typing import Any, Dict, List

from backend.agent.prompts import AGENT_SYSTEM_PROMPT


class Planner:

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

        for item in history:

            messages.append(
                {
                    "role": "user",
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

        lines = []

        for tool in tools:

            lines.append(
                f"- {tool.get('name')}: "
                f"{tool.get('description')}\n"
                f"  Parameters: "
                f"{tool.get('parameters', {})}"
            )

        return "\n".join(lines)

    def _format_history_item(
        self,
        item: Dict[str, Any],
    ) -> str:

        item_type = item.get(
            "type",
            "observation"
        )

        if item_type == "tool_result":

            return (
                "TOOL OBSERVATION:\n"
                f"Tool: {item.get('tool')}\n"
                f"Success: {item.get('success')}\n"
                f"Result: {item.get('result')}\n"
                f"Error: {item.get('error')}"
            )

        if item_type == "error":

            return (
                "FRAMEWORK ERROR:\n"
                f"{item.get('error')}"
            )

        return str(item)
