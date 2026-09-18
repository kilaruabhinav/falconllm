import json
import re

from pydantic import ValidationError

from backend.agent.schemas import AgentAction


class AgentParseError(Exception):
    pass


def parse_agent_response(response: str) -> AgentAction:

    if not isinstance(response, str) or not response.strip():
        raise AgentParseError(
            "LLM returned an empty response."
        )

    cleaned = response.strip()

    # Handle models that accidentally wrap JSON in markdown fences.
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError as exc:

        raise AgentParseError(
            f"LLM returned invalid JSON: {exc}"
        ) from exc

    try:
        return AgentAction.model_validate(data)

    except ValidationError as exc:

        raise AgentParseError(
            f"LLM response does not match AgentAction schema: {exc}"
        ) from exc
