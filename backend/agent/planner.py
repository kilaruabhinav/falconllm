class Planner:

    def build_context(
        self,
        user_query,
        history,
        available_tools
    ):
        return {
            "query": user_query,
            "history": history,
            "tools": available_tools
        }
