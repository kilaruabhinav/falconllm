import json

from backend.agent.schemas import AgentAction


def parse_agent_response(response: str) -> AgentAction:

    try:
        data = json.loads(response)
        return AgentAction(**data)

    except Exception as exc:
        raise ValueError(
            f"Invalid agent response: {exc}"
        )
