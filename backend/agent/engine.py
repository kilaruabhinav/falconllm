from typing import Any, Dict, List

from backend.agent.llm_client import BaseLLMClient
from backend.tools.registry import ToolRegistry
from backend.agent.parser import (
    AgentParseError,
    parse_agent_response,
)
from backend.agent.planner import Planner
from backend.agent.schemas import (
    AgentResult,
    ToolResult,
    TraceStep,
)


class AgentEngine:

    def __init__(
        self,
        llm: BaseLLMClient,
        tool_registry: ToolRegistry,
        trace_manager=None,
        config=None,
    ):

        self.llm = llm
        self.tool_registry = tool_registry
        self.trace_manager = trace_manager
        self.config = config

        self.planner = Planner()

        self.max_iterations = getattr(
            config,
            "MAX_AGENT_ITERATIONS",
            10,
        )

        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("MAX_AGENT_ITERATIONS must be a positive integer.")

    async def run(
        self,
        user_query: str,
    ) -> AgentResult:

        history: List[
            Dict[str, Any]
        ] = []

        trace: List[
            TraceStep
        ] = []

        step_counter = 0

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):

            available_tools = self.tool_registry.get_tool_schemas()

            messages = (
                self.planner.build_messages(
                    user_query=user_query,
                    history=history,
                    available_tools=available_tools,
                    iteration=iteration,
                )
            )

            try:

                raw_response = (
                    await self.llm.generate(
                        messages=messages,
                        tools=available_tools,
                    )
                )

            except Exception as exc:

                step_counter += 1

                trace.append(
                    TraceStep(
                        step=step_counter,
                        type="llm_error",
                        error=str(exc),
                    )
                )

                history.append(
                    {
                        "type": "error",
                        "error": (
                            f"LLM error: {exc}"
                        ),
                    }
                )

                continue

            try:

                action = (
                    parse_agent_response(
                        raw_response
                    )
                )

            except AgentParseError as exc:

                step_counter += 1

                trace.append(
                    TraceStep(
                        step=step_counter,
                        type="parse_error",
                        error=str(exc),
                    )
                )

                history.append(
                    {
                        "type": "error",
                        "error": (
                            "Your previous response "
                            "was invalid. "
                            f"{exc}"
                        ),
                    }
                )

                continue

            history.append({"type": "action", "action": action.model_dump(exclude_none=True)})

            step_counter += 1

            trace.append(
                TraceStep(
                    step=step_counter,
                    type="plan",
                    content=action.plan,
                )
            )

            if action.type == "final":

                step_counter += 1

                trace.append(
                    TraceStep(
                        step=step_counter,
                        type="final",
                        content=action.answer,
                    )
                )

                return AgentResult(
                    status="completed",
                    answer=action.answer,
                    iterations=iteration,
                    trace=trace,
                )

            step_counter += 1

            trace.append(
                TraceStep(
                    step=step_counter,
                    type="tool_call",
                    tool=action.tool,
                    arguments=(
                        action.arguments
                        or {}
                    ),
                )
            )

            try:

                tool_result = (
                    await self.tool_registry
                    .execute(
                        action.tool,
                        action.arguments
                        or {},
                    )
                )

                tool_result = ToolResult.model_validate(tool_result)

            except Exception as exc:

                tool_result = None

                step_counter += 1

                trace.append(
                    TraceStep(
                        step=step_counter,
                        type="tool_error",
                        tool=action.tool,
                        error=str(exc),
                    )
                )

                history.append(
                    {
                        "type": "tool_result",
                        "tool": action.tool,
                        "success": False,
                        "result": None,
                        "error": str(exc),
                    }
                )

                continue

            step_counter += 1

            trace.append(
                TraceStep(
                    step=step_counter,
                    type=(
                        "tool_result"
                        if tool_result.success
                        else "tool_error"
                    ),
                    tool=tool_result.tool,
                    result=tool_result.result,
                    error=tool_result.error,
                )
            )

            history.append(
                {
                    "type": "tool_result",
                    "tool": tool_result.tool,
                    "success": (
                        tool_result.success
                    ),
                    "result": (
                        tool_result.result
                    ),
                    "error": (
                        tool_result.error
                    ),
                }
            )

        return AgentResult(
            status="max_iterations",
            answer=None,
            iterations=self.max_iterations,
            trace=trace,
        )
