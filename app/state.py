"""State owns facts; the model can propose actions, not overwrite authority."""
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TravelRequest(StrictSchema):
    origin: str
    destination: str
    travel_date: str
    time_preference: Literal["morning", "afternoon", "evening", "any"] = "morning"
    max_price: int = 2_000_000
    min_baggage_kg: int = 15
    passenger_name: str = ""


class PlanStep(StrictSchema):
    tool: str
    args: dict[str, Any]
    purpose: str


class ActionPlan(StrictSchema):
    selection_rule: Literal["cheapest", "earliest"] = "cheapest"
    steps: list[PlanStep] = Field(min_length=1, max_length=10)


class Strategy(StrictSchema):
    stages: list[Literal["discover", "reserve", "finalize"]] = Field(
        min_length=1, max_length=5
    )
    rationale: str


class BookingSummary(StrictSchema):
    booking_id: str
    flight_id: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    price: int
    baggage_kg: int
    passenger_name: str


@dataclass(frozen=True)
class Confirmation:
    """Receipt supplied by the user/CLI, never accepted from model output."""
    confirmed: bool = False
    origin: str = "SGN"
    destination: str = "HAN"
    travel_date: str = ""
    max_price: int = 2_000_000
    passenger_name: str = "Nguyen Van An"


@dataclass
class Metrics:
    model_calls: int = 0
    tool_calls: int = 0
    total_steps: int = 0
    blocked_actions: int = 0
    handoff_count: int = 0
    replans: int = 0
    loop_detected: bool = False
    stall_detected: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    token_usage_reported: bool = False
    elapsed_seconds: float = 0.0


@dataclass
class AgentState:
    user_request: str
    confirmation: Confirmation
    passenger_input: str = "Nguyen Van An"
    hold_allowed: bool = True
    request: TravelRequest | None = None
    searched_flights: dict[str, dict[str, Any]] = field(default_factory=dict)
    flight_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_flight: dict[str, Any] | None = None
    rejected_flights: list[str] = field(default_factory=list)
    hold: dict[str, Any] | None = None
    booking: dict[str, Any] | None = None
    final_summary: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    attempted_actions: list[dict[str, Any]] = field(default_factory=list)
    messages: list[BaseMessage] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    status: str = "running"
    handoff: dict[str, Any] | None = None
    completion_missing: list[str] = field(default_factory=list)
    plan: list[PlanStep] = field(default_factory=list)
    plan_history: list[dict[str, Any]] = field(default_factory=list)
    plan_index: int = 0
    selection_rule: str = "cheapest"
    stages: list[str] = field(default_factory=list)
    stage_index: int = 0
    needs_replan: bool = False
    pending: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        """Compact model context; excludes authority-changing setters."""
        return {
            "request": self.request.model_dump() if self.request else None,
            "searched_flights": list(self.searched_flights.values()),
            "flight_details": list(self.flight_details.values()),
            "selected_flight": self.selected_flight,
            "rejected_flights": self.rejected_flights,
            "hold": self.hold,
            "booking": self.booking,
            "confirmation": asdict(self.confirmation),
            "last_observation": self.observations[-1] if self.observations else None,
            "completion_missing": self.completion_missing,
            "status": self.status,
        }


class GraphState(TypedDict):
    """LangGraph carries one authoritative domain state between nodes."""
    domain: AgentState
