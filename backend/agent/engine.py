from backend.agent.schemas import AgentAction


class AgentEngine:

    def __init__(
        self,
        llm,
        tool_registry,
        trace_manager=None,
        config=None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.trace_manager = trace_manager
        self.config = config

    async def run(self, user_query: str):

        print(f"Starting agent for: {user_query}")

        # Agent loop will be implemented here next.

        return {
            "status": "initialized",
            "query": user_query
        }
