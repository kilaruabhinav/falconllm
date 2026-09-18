"""In-process live trace fan-out for active agent runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import threading
from typing import Any

from .trace import TraceStep, TraceStepType


TERMINAL_TRACE_TYPES = {
    TraceStepType.RUN_COMPLETED.value,
    TraceStepType.RUN_FAILED.value,
    TraceStepType.RUN_CANCELLED.value,
}


@dataclass(eq=False)
class TraceSubscription:
    run_id: str
    queue: asyncio.Queue[dict[str, Any]]
    loop: asyncio.AbstractEventLoop


@dataclass
class _RunChannel:
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[TraceSubscription] = field(default_factory=set)
    terminal: bool = False


class TraceEventBroker:
    """Keep active-run events long enough for race-free SSE subscription."""

    def __init__(self) -> None:
        self._channels: dict[str, _RunChannel] = {}
        self._lock = threading.Lock()

    def create_run(self, run_id: str) -> None:
        with self._lock:
            self._channels.setdefault(run_id, _RunChannel())

    def has_run(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._channels

    def publish(self, step: TraceStep) -> None:
        event = step.to_dict()
        with self._lock:
            channel = self._channels.setdefault(step.run_id, _RunChannel())
            channel.events.append(event)
            if step.step_type in TERMINAL_TRACE_TYPES:
                channel.terminal = True
            subscribers = tuple(channel.subscribers)

        for subscription in subscribers:
            if subscription.loop.is_closed():
                continue
            subscription.loop.call_soon_threadsafe(subscription.queue.put_nowait, event)

    def subscribe(
        self, run_id: str, after_sequence: int = 0
    ) -> tuple[TraceSubscription | None, list[dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        subscription = TraceSubscription(run_id, asyncio.Queue(), loop)
        with self._lock:
            channel = self._channels.get(run_id)
            if channel is None:
                return None, []
            backlog = [
                event for event in channel.events
                if int(event.get("sequence", 0)) > after_sequence
            ]
            if not channel.terminal:
                channel.subscribers.add(subscription)
                return subscription, backlog
            return None, backlog

    def unsubscribe(self, subscription: TraceSubscription | None) -> None:
        if subscription is None:
            return
        with self._lock:
            channel = self._channels.get(subscription.run_id)
            if channel is not None:
                channel.subscribers.discard(subscription)
                if channel.terminal and not channel.subscribers:
                    self._channels.pop(subscription.run_id, None)

    def finish(self, run_id: str) -> None:
        """Release completed channels once their connected streams drain."""
        with self._lock:
            channel = self._channels.get(run_id)
            if channel is None:
                return
            channel.terminal = True
            if not channel.subscribers:
                self._channels.pop(run_id, None)

    @property
    def active_channel_count(self) -> int:
        with self._lock:
            return len(self._channels)
