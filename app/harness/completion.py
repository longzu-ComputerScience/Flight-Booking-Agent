"""DONE means the code can prove the booking, not that a model says so."""
from dataclasses import dataclass
from datetime import date

from app.harness.constraints import DataConstraints
from app.harness.permissions import PermissionGuard
from app.state import AgentState, BookingSummary


@dataclass
class CompletionResult:
    done: bool
    missing_conditions: list[str]


class CompletionChecker:
    def __init__(self, today: date):
        self.constraints = DataConstraints(today)

    def check(self, state: AgentState) -> CompletionResult:
        missing = []
        if state.request is None:
            missing.append("validated_request")
        else:
            missing.extend(self.constraints.request(state.request).errors)
        flight = state.selected_flight
        if flight is None:
            missing.append("valid_selected_flight")
        elif state.request:
            missing.extend(self.constraints.flight(flight, state.request).errors)
            if state.flight_details.get(flight["flight_id"]) != flight:
                missing.append("flight_details_provenance")
            if not any(obs["tool"] == "get_flight_details" and obs["result"].get("flight") == flight
                       for obs in state.observations):
                missing.append("flight_details_tool_provenance")
            if flight["flight_id"] not in state.searched_flights:
                missing.append("flight_search_provenance")
        if not PermissionGuard().check("book_flight", state).allowed:
            missing.append("scoped_user_confirmation")
        booking = state.booking
        if not booking or booking.get("status") != "confirmed":
            missing.append("confirmed_tool_booking")
        if not booking or not booking.get("booking_id"):
            missing.append("booking_id")
        if booking:
            if not state.hold or booking.get("hold_id") != state.hold.get("hold_id"):
                missing.append("booking_hold_match")
            sourced = any(
                obs["tool"] == "book_flight" and obs["result"] == booking
                for obs in state.observations
            )
            if not sourced:
                missing.append("booking_tool_provenance")
            if not flight or booking.get("flight_id") != flight["flight_id"]:
                missing.append("booking_flight_match")
            if state.request and booking.get("passenger_name") != state.request.passenger_name:
                missing.append("booking_passenger_match")
            if flight and any(booking.get(key) != flight[key] for key in (
                "origin", "destination", "departure_time", "arrival_time", "price", "baggage_kg"
            )):
                missing.append("booking_fare_and_route_match")
        if state.final_summary is None:
            missing.append("grounded_user_summary")
        elif not booking or any(
            state.final_summary.get(key) != booking.get(key)
            for key in BookingSummary.model_fields
        ):
            missing.append("final_summary_not_grounded")
        return CompletionResult(not missing, list(dict.fromkeys(missing)))


def format_booking(summary: dict) -> str:
    """Render only checked fields. No unchecked model prose is shown as facts."""
    return (
        f"Đã đặt vé MÔ PHỎNG {summary['booking_id']}: {summary['flight_id']} "
        f"{summary['origin']} → {summary['destination']}, "
        f"khởi hành {summary['departure_time']}, đến {summary['arrival_time']}. "
        f"Giá trọn gói {summary['price']:,} VNĐ, ký gửi {summary['baggage_kg']} kg. "
        f"Hành khách: {summary['passenger_name']}. Không có giao dịch thật."
    )
