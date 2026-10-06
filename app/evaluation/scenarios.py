"""Same requests, confirmation scopes and fault injections for all architectures."""
from dataclasses import dataclass, replace
from datetime import date, timedelta

from app.config import Settings
from app.state import AgentState, Confirmation


@dataclass(frozen=True)
class Scenario:
    name: str
    user_request: str
    confirmation: Confirmation
    expected_status: str
    expected_reason: str = ""
    passenger: str = "Nguyen Van An"
    hold_allowed: bool = True
    tool_fault: str = "none"
    model_behavior: str = "normal"
    model_budget: int = 18
    tool_budget: int = 12

    def make_state(self) -> AgentState:
        return AgentState(user_request=self.user_request, confirmation=self.confirmation,
                          passenger_input=self.passenger, hold_allowed=self.hold_allowed)

    def settings(self, today: date) -> Settings:
        return Settings(today=today, max_model_calls=self.model_budget, max_tool_calls=self.tool_budget)


def scenarios(today: date) -> dict[str, Scenario]:
    tomorrow = (today + timedelta(days=1)).isoformat()
    request = "Tìm và đặt giúp tôi chuyến bay từ TP.HCM ra Hà Nội vào sáng mai, dưới 2 triệu và có hành lý ký gửi."
    confirmation = Confirmation(True, "SGN", "HAN", tomorrow, 2_000_000, "Nguyen Van An")
    happy = Scenario("happy", request, confirmation, "done")
    rows = [
        happy,
        replace(happy, name="missing_confirmation", confirmation=replace(confirmation, confirmed=False),
                expected_status="handoff", expected_reason="user_confirmation_missing"),
        replace(happy, name="no_matching", user_request=request.replace("2 triệu", "0.8 triệu"),
                expected_status="handoff", expected_reason="no_matching_flight"),
        replace(happy, name="invalid_route", user_request="Đặt chuyến SGN đến SGN sáng mai dưới 2 triệu, ký gửi.",
                expected_status="handoff", expected_reason="invalid_input"),
        replace(happy, name="invalid_date", user_request=f"Đặt chuyến SGN đến HAN ngày {(today - timedelta(days=1)).isoformat()} buổi sáng dưới 2 triệu, ký gửi.",
                expected_status="handoff", expected_reason="invalid_input"),
        replace(happy, name="transient_error", tool_fault="transient_search"),
        replace(happy, name="sold_out", tool_fault="sold_out"),
        replace(happy, name="loop", model_behavior="loop", expected_status="handoff", expected_reason="loop_detected:"),
        # Hai lượt đủ cho parse + plan, nhưng không đủ cho reactive reasoning nhiều lần.
        replace(happy, name="budget", model_budget=2, expected_status="architecture_dependent"),
        replace(happy, name="forged_id", model_behavior="forged_id", expected_status="handoff", expected_reason="data_constraint_failed"),
        replace(happy, name="premature_finish", model_behavior="premature_finish", expected_status="handoff", expected_reason="completion_not_proven"),
        replace(happy, name="missing_passenger", passenger="", expected_status="handoff", expected_reason="invalid_input"),
        replace(happy, name="hold_denied", hold_allowed=False, expected_status="handoff", expected_reason="hold_not_authorized"),
    ]
    return {row.name: row for row in rows}
