import json

from pydantic import ValidationError

from backend.agent.schemas import AgentAction


class AgentParseError(Exception):
    pass


def parse_agent_response(response: str) -> AgentAction:

    if not response or not response.strip():
        raise AgentParseError(
            "LLM returned an empty response."
        )

    cleaned = response.strip()

    # Handle models that accidentally wrap JSON in markdown fences.
    if cleaned.startswith("```"):

        cleaned = cleaned.strip("`")

        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

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
