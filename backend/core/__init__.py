"""AI Agent Framework Core Brain Module.

Exports the core infrastructure components:
- State Management (AgentState, StateManager, AgentStatus)
- Execution Tracing (ExecutionTrace, TraceStep, TraceManager)
- Failure Taxonomy (ErrorType, AgentError and typed subclasses)
- Failure Recovery (RecoveryManager, RecoveryDecision)
- Guards & Loop Protection (IterationGuard, RepeatedActionGuard, LoopGuard, GuardManager)
- Persistence (TraceStore, SQLiteTraceStore, RunRecord, RunUpdate)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package directory is on sys.path for direct imports
_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from protocols import (
    TraceStore,
    RunRecord,
    RunUpdate,
    ToolProtocol,
    ClockProtocol,
    SleeperProtocol,
)

from errors import (
    ErrorType,
    AgentError,
    ToolTimeoutError,
    UnknownToolError,
    ToolExecutionError,
    InvalidToolArgumentsError,
    EmptyToolResponseError,
    MalformedLLMOutputError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMProviderError,
    RepeatedActionError,
    AgentLoopDetectedError,
    MaxIterationsExceededError,
    PersistenceError,
    SerializationError,
    AgentCancelledError,
)

from state import (
    AgentState,
    AgentStatus,
    Action,
    ToolCall,
    Observation,
    ErrorRecord,
    TokenUsage,
    StateManager,
)

from trace import (
    ExecutionTrace,
    TraceStep,
    TraceStepType,
    TraceManager,
)

from recovery import (
    RecoveryDecision,
    RecoveryManager,
)

from guards import (
    IterationGuard,
    RepeatedActionGuard,
    LoopGuard,
    PayloadGuard,
    GuardManager,
    GuardResult,
)

from persistence import (
    SQLiteTraceStore,
)

__all__ = [
    # Protocols
    "TraceStore",
    "RunRecord",
    "RunUpdate",
    "ToolProtocol",
    "ClockProtocol",
    "SleeperProtocol",
    # Errors
    "ErrorType",
    "AgentError",
    "ToolTimeoutError",
    "UnknownToolError",
    "ToolExecutionError",
    "InvalidToolArgumentsError",
    "EmptyToolResponseError",
    "MalformedLLMOutputError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMProviderError",
    "RepeatedActionError",
    "AgentLoopDetectedError",
    "MaxIterationsExceededError",
    "PersistenceError",
    "SerializationError",
    "AgentCancelledError",
    # State
    "AgentState",
    "AgentStatus",
    "Action",
    "ToolCall",
    "Observation",
    "ErrorRecord",
    "TokenUsage",
    "StateManager",
    # Trace
    "ExecutionTrace",
    "TraceStep",
    "TraceStepType",
    "TraceManager",
    # Recovery
    "RecoveryDecision",
    "RecoveryManager",
    # Guards
    "IterationGuard",
    "RepeatedActionGuard",
    "LoopGuard",
    "PayloadGuard",
    "GuardManager",
    "GuardResult",
    # Persistence
    "SQLiteTraceStore",
]
