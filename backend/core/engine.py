from state import AgentState
from trace import ExecutionTrace
from errors import AgentError, ErrorType
from recovery import RecoveryManager
from guards import LoopGuard, IterationGuard
from Storage.database import SQLiteStorage


class AgentEngine:

    def __init__(self, maximum_iterations=10):

        self.maximum_iterations = maximum_iterations

        self.recovery = RecoveryManager(
            max_retries=2
        )

        self.loop_guard = LoopGuard(
            max_repeated_actions=3
        )

        self.iteration_guard = IterationGuard(
            maximum_iterations=maximum_iterations
        )

        self.storage = SQLiteStorage(
            database_path="agent.db"
        )

    # =========================================
    # TEMPORARY TOOL EXECUTOR
    # =========================================

    def execute_tool(self, tool_name, arguments):

        # TEMPORARY TIMEOUT TEST
        if tool_name == "search":

            raise AgentError(
                error_type=ErrorType.TOOL_TIMEOUT,
                message="Search tool timed out.",
                retryable=True
            )

        raise AgentError(
            error_type=ErrorType.UNKNOWN_TOOL,
            message=f"Unknown tool: {tool_name}",
            retryable=False
        )

    # =========================================
    # DATABASE SAVE
    # =========================================

    def save_to_database(self, state, trace):

        self.storage.save_run(
            run_id=trace.run_id,
            user_query=state.user_query,
            started_at=trace.started_at.isoformat(),
            completed_at=(
                trace.completed_at.isoformat()
                if trace.completed_at
                else None
            ),
            status=trace.status,
            error=trace.error,
            final_output=trace.final_output
        )

        for step in trace.steps:

            self.storage.save_step(
                run_id=trace.run_id,
                step_number=step.step_number,
                step_type=step.step_type,
                description=step.description,
                data=step.data,
                timestamp=step.timestamp.isoformat()
            )

    # =========================================
    # MAIN EXECUTION
    # =========================================

    def run(self, user_query):

        state = AgentState(
            user_query=user_query,
            maximum_iterations=self.maximum_iterations
        )

        trace = ExecutionTrace()

        try:

            # START
            state.add_message(
                role="user",
                content=user_query
            )

            trace.add_step(
                step_type="START",
                description="Agent execution started",
                data={
                    "user_query": user_query
                }
            )

            # ITERATION
            state.current_iteration += 1

            self.iteration_guard.check_iteration(
                state.current_iteration
            )

            # PLAN
            tool_name = "search"

            arguments = {
                "query": user_query
            }

            trace.add_plan(
                description="Agent decided to use search tool",
                data={
                    "iteration": state.current_iteration,
                    "tool": tool_name
                }
            )

            # ACTION
            state.add_action(
                tool_name=tool_name,
                arguments=arguments
            )

            tool_call = state.add_tool_call(
                tool_name=tool_name,
                arguments=arguments
            )

            # GUARD
            self.loop_guard.check_action(
                tool_name,
                arguments
            )

            trace.add_action(
                description="Agent selected tool",
                data={
                    "tool": tool_name,
                    "arguments": arguments
                }
            )

            # TOOL EXECUTION
            try:

                result = self.execute_tool(
                    tool_name,
                    arguments
                )

                tool_call.status = "COMPLETED"
                tool_call.result = result

            except AgentError:

                tool_call.status = "FAILED"
                raise

            # OBSERVE
            state.add_observation(
                success=True,
                result=result
            )

            trace.add_observation(
                description="Agent received tool result",
                data={
                    "result": result
                }
            )

            # FINAL
            final_answer = result

            state.final_answer = final_answer
            state.status = "COMPLETED"

            trace.complete(
                final_output=final_answer
            )

            # SAVE
            self.save_to_database(
                state,
                trace
            )

            return {
                "state": state.to_dict(),
                "trace": trace.to_dict()
            }

        except AgentError as error:

            # ERROR
            state.status = "FAILED"

            state.add_error(
                error_type=error.error_type.value,
                message=error.message,
                retryable=error.retryable
            )

            trace.add_error(
                description="Agent encountered an error",
                data=error.to_dict()
            )

            # RECOVERY
            decision = self.recovery.recover(
                error
            )

            trace.add_recovery(
                description="Recovery decision created",
                data={
                    "action": decision.action,
                    "reason": decision.reason,
                    "retry": decision.retry
                }
            )

            # FAIL FOR THIS TEST
            trace.fail(
                error.message
            )

            # SAVE
            self.save_to_database(
                state,
                trace
            )

            return {
                "state": state.to_dict(),
                "trace": trace.to_dict()
            }


# =============================================
# TEST
# =============================================

if __name__ == "__main__":

    engine = AgentEngine(
        maximum_iterations=10
    )

    result = engine.run(
        "What is RAG?"
    )

    print("\n====================================")
    print("TOOL TIMEOUT FAILURE TEST")
    print("====================================")

    print("\nTRACE:")

    for step in result["trace"]["steps"]:

        print(
            f"{step['step_number']}. "
            f"{step['step_type']} -> "
            f"{step['description']}"
        )

    print("\nSTATUS:")
    print(result["trace"]["status"])

    print("\nERROR:")
    print(result["trace"]["error"])

    print("\nFULL TRACE:")
    print(result["trace"])