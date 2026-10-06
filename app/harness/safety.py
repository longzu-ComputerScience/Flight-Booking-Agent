"""Bounded retries and progress checks; this is not a semantic oracle."""
import json
from collections import deque
from typing import Any

from app.config import Settings
from app.state import AgentState


class SafetyGuard:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.recent: deque[str] = deque(maxlen=8)
        self.last_progress: tuple | None = None
        self.stalled = 0
        self.consecutive_errors = 0

    def before_model(self, state: AgentState) -> str | None:
        if state.metrics.model_calls >= self.settings.max_model_calls:
            return "model_budget_exceeded"
        return None

    def before_tool(self, name: str, args: dict[str, Any], state: AgentState) -> str | None:
        if state.metrics.tool_calls >= self.settings.max_tool_calls:
            return "tool_budget_exceeded"
        fingerprint = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        if self.recent.count(fingerprint) + 1 >= self.settings.repeat_limit:
            state.metrics.loop_detected = True
            return f"loop_detected:{name}:same_args_{self.settings.repeat_limit}_times"
        self.recent.append(fingerprint)
        return None

    def after_tool(self, state: AgentState, result: dict) -> str | None:
        progress = (
            len(state.searched_flights), len(state.flight_details),
            len(state.rejected_flights), bool(state.hold), bool(state.booking),
        )
        self.stalled = self.stalled + 1 if progress == self.last_progress else 0
        self.last_progress = progress
        self.consecutive_errors = self.consecutive_errors + 1 if result.get("status") == "error" else 0
        if self.consecutive_errors >= self.settings.consecutive_error_limit:
            return "consecutive_tool_errors"
        if self.stalled >= self.settings.stall_limit:
            state.metrics.stall_detected = True
            return "progress_stalled"
        return None
