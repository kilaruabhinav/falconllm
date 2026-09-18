"""LLM-driven PLAN → ACT → OBSERVE agent loop."""

from __future__ import annotations

import time
from typing import Any

from backend.agent.llm_client import BaseLLMClient
from backend.agent.parser import AgentParseError, parse_agent_response
from backend.agent.planner import Planner
from backend.agent.schemas import AgentResult, ToolResult
from backend.core.errors import AgentError, ErrorType
from backend.core.guards import GuardManager
from backend.core.recovery import RecoveryManager
from backend.core.state import AgentState, AgentStatus, StateManager
from backend.core.trace import TraceManager, TraceStepType
from backend.tools.registry import ToolRegistry


class AgentEngine:
    """Coordinate the LLM, tools, state, guards, recovery, and trace layers."""

    def __init__(
        self,
        llm: BaseLLMClient,
        tool_registry: ToolRegistry,
        trace_manager: TraceManager | None = None,
        config: Any = None,
    ) -> None:
        self.llm = llm
        self.tool_registry = tool_registry
        self.trace_manager = trace_manager or TraceManager()
        self.config = config
        self.planner = Planner()
        self.max_iterations = getattr(config, "MAX_AGENT_ITERATIONS", 10)
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("MAX_AGENT_ITERATIONS must be a positive integer.")

    async def run(self, user_query: str, run_id: str | None = None) -> AgentResult:
        run_started = time.perf_counter()
        state = StateManager.create(
            user_query=user_query,
            maximum_iterations=self.max_iterations,
            run_id=run_id,
        )
        state.add_message("user", user_query)
        state.set_status(AgentStatus.RUNNING)
        trace = self.trace_manager.start_run(state)
        def record_llm_event(event_type: str, data: dict[str, Any]) -> None:
            metadata = dict(data)
            duration_ms = metadata.pop("duration_ms", None)
            self.trace_manager.record_step(
                state.run_id,
                event_type,
                content=self._llm_event_content(event_type, data),
                status="FAILED" if event_type == "LLM_PROVIDER_ERROR" else "SUCCESS",
                error=data.get("code") if event_type == "LLM_PROVIDER_ERROR" else None,
                metadata=metadata,
                duration_ms=duration_ms,
            )

        self.llm.start_run(state.run_id, record_llm_event)
        guard = GuardManager(maximum_iterations=self.max_iterations)
        recovery = RecoveryManager(max_retries=0)
        history: list[dict[str, Any]] = []

        for iteration in range(1, self.max_iterations + 1):
            state.current_iteration = iteration
            available_tools = self.tool_registry.get_tool_schemas()
            messages = self.planner.build_messages(
                user_query=user_query,
                history=history,
                available_tools=available_tools,
                iteration=iteration,
            )

            try:
                llm_started = time.perf_counter()
                raw_response = await self.llm.generate(messages=messages, tools=available_tools)
                llm_duration_ms = (time.perf_counter() - llm_started) * 1000
            except Exception as exc:
                self._record_failure(
                    state,
                    TraceStepType.LLM_ERROR,
                    ErrorType.LLM_PROVIDER_ERROR,
                    f"LLM error: {exc}",
                    history,
                    recovery,
                )
                continue

            try:
                action = parse_agent_response(raw_response)
            except AgentParseError as exc:
                self._record_failure(
                    state,
                    TraceStepType.PARSE_ERROR,
                    ErrorType.MALFORMED_LLM_OUTPUT,
                    f"Invalid LLM response: {exc}",
                    history,
                    recovery,
                )
                continue

            history.append({"type": "action", "action": action.model_dump(exclude_none=True)})
            self.trace_manager.record_plan(
                state.run_id,
                action.plan,
                data={"iteration": iteration, "timing_kind": "llm"},
                duration_ms=llm_duration_ms,
            )

            if action.type == "final":
                state.final_answer = action.answer
                state.set_status(AgentStatus.COMPLETED)
                self.trace_manager.record_step(
                    state.run_id,
                    TraceStepType.FINAL,
                    content=action.answer or "",
                )
                self.trace_manager.complete_run(
                    state.run_id,
                    action.answer or "",
                    duration_ms=(time.perf_counter() - run_started) * 1000,
                )
                result = self._result(state, trace, "completed")
                self.llm.end_run()
                return result

            arguments = action.arguments or {}
            guard_result = guard.check_action(action.tool or "", arguments)
            if not guard_result.passed:
                self._record_failure(
                    state,
                    TraceStepType.TOOL_ERROR,
                    ErrorType.REPEATED_ACTION,
                    guard_result.message or "Repeated action detected",
                    history,
                    recovery,
                    tool_name=action.tool,
                    arguments=arguments,
                )
                continue

            state.add_action(action.tool or "", arguments)
            tool_call = state.add_tool_call(action.tool or "", arguments)
            self.trace_manager.record_step(
                state.run_id,
                TraceStepType.TOOL_CALL,
                content=f"Calling tool '{action.tool}'",
                status="STARTED",
                tool_name=action.tool,
                arguments=arguments,
            )

            try:
                tool_started = time.perf_counter()
                tool_result = ToolResult.model_validate(
                    await self.tool_registry.execute(action.tool or "", arguments)
                )
            except Exception as exc:
                tool_result = ToolResult(
                    success=False, tool=action.tool or "", error=str(exc)
                )
            tool_duration_ms = (time.perf_counter() - tool_started) * 1000

            tool_call.status = "COMPLETED" if tool_result.success else "FAILED"
            tool_call.result = tool_result.result
            tool_call.error = tool_result.error
            state.add_observation(
                success=tool_result.success,
                result=tool_result.result,
                error_type=None if tool_result.success else ErrorType.TOOL_EXCEPTION.value,
                message=tool_result.error,
                retryable=False,
            )
            if not tool_result.success:
                state.add_error(
                    ErrorType.TOOL_EXCEPTION.value,
                    tool_result.error or "Tool failed",
                    retryable=False,
                )

            self.trace_manager.record_step(
                state.run_id,
                TraceStepType.TOOL_RESULT if tool_result.success else TraceStepType.TOOL_ERROR,
                content="Tool completed" if tool_result.success else "Tool failed",
                status="SUCCESS" if tool_result.success else "FAILED",
                tool_name=tool_result.tool,
                observation=tool_result.result,
                error=tool_result.error,
                duration_ms=tool_duration_ms,
                metadata={"timing_kind": "search" if tool_result.tool == "search" else "tool"},
            )
            history.append({"type": "tool_result", **tool_result.model_dump()})
            if not tool_result.success:
                decision = recovery.handle_tool_result(
                    state.run_id, state, action.tool or "", tool_result.model_dump()
                )
                self.trace_manager.record_recovery(state.run_id, decision)

        state.set_status(AgentStatus.MAX_ITERATIONS_EXCEEDED)
        message = f"Maximum of {self.max_iterations} iterations reached."
        state.add_error(ErrorType.MAX_ITERATIONS_EXCEEDED.value, message)
        self.trace_manager.fail_run(state.run_id, message)
        result = self._result(state, trace, "max_iterations")
        self.llm.end_run()
        return result

    @staticmethod
    def _llm_event_content(event_type: str, data: dict[str, Any]) -> str:
        if event_type == "LLM_PROVIDER_SELECTED":
            return f"Selected {data.get('provider')}/{data.get('model')}"
        if event_type == "LLM_PROVIDER_ERROR":
            return f"{data.get('provider')}/{data.get('model')} failed: {data.get('code')}"
        if event_type == "LLM_FALLBACK":
            return f"Falling back from {data.get('from')} to {data.get('to')}"
        return event_type

    def _record_failure(
        self,
        state: AgentState,
        step_type: TraceStepType,
        error_type: ErrorType,
        message: str,
        history: list[dict[str, Any]],
        recovery: RecoveryManager,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        state.add_error(error_type.value, message, retryable=False)
        state.add_observation(False, error_type=error_type.value, message=message)
        self.trace_manager.record_step(
            state.run_id,
            step_type,
            content=message,
            status="FAILED",
            tool_name=tool_name,
            arguments=arguments,
            error=message,
        )
        history.append({"type": "error", "error": message})
        decision = recovery.recover(
            AgentError(error_type=error_type, message=message, retryable=False),
            tool_name=tool_name,
        )
        self.trace_manager.record_recovery(state.run_id, decision)

    @staticmethod
    def _result(state: AgentState, trace: Any, status: str) -> AgentResult:
        return AgentResult(
            run_id=state.run_id,
            status=status,
            answer=state.final_answer,
            iterations=state.current_iteration,
            trace=list(trace.steps),
        )
