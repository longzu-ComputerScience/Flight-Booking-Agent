"""Deterministic validation of requests, observations and proposed actions."""
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.data.flights import AIRPORTS
from app.state import AgentState, TravelRequest


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def in_time_window(departure: str, preference: str) -> bool:
    hour = datetime.fromisoformat(departure).hour
    return {
        "morning": 0 <= hour < 12,
        "afternoon": 12 <= hour < 18,
        "evening": 18 <= hour < 24,
        "any": True,
    }.get(preference, False)


class DataConstraints:
    def __init__(self, today: date):
        self.today = today

    def request(self, request: TravelRequest) -> ValidationResult:
        errors = []
        if request.origin not in AIRPORTS or request.destination not in AIRPORTS:
            errors.append("airport_code_invalid")
        if request.origin == request.destination:
            errors.append("origin_equals_destination")
        try:
            if date.fromisoformat(request.travel_date) < self.today:
                errors.append("travel_date_in_past")
        except ValueError:
            errors.append("travel_date_invalid")
        if request.max_price <= 0:
            errors.append("max_price_must_be_positive")
        if request.min_baggage_kg < 0:
            errors.append("baggage_must_be_nonnegative")
        if not request.passenger_name.strip():
            errors.append("passenger_name_missing")
        return ValidationResult(not errors, errors)

    def flight(self, flight: dict[str, Any], request: TravelRequest) -> ValidationResult:
        errors = []
        try:
            if (flight["origin"], flight["destination"]) != (request.origin, request.destination):
                errors.append("flight_route_mismatch")
            if datetime.fromisoformat(flight["departure_time"]).date().isoformat() != request.travel_date:
                errors.append("flight_date_mismatch")
            if not in_time_window(flight["departure_time"], request.time_preference):
                errors.append("flight_time_mismatch")
            if not 0 < flight["price"] <= request.max_price:
                errors.append("flight_price_out_of_budget")
            if flight["baggage_kg"] < request.min_baggage_kg:
                errors.append("flight_baggage_insufficient")
            if flight["seats_left"] <= 0:
                errors.append("flight_sold_out")
        except (KeyError, TypeError, ValueError):
            errors.append("flight_observation_invalid")
        return ValidationResult(not errors, errors)

    def action(self, name: str, args: dict[str, Any], state: AgentState) -> ValidationResult:
        if not state.request:
            return ValidationResult(False, ["request_missing"])
        errors = []
        if name == "search_flights":
            expected = {
                "origin": state.request.origin, "destination": state.request.destination,
                "date": state.request.travel_date, "time_preference": state.request.time_preference,
            }
            if args != expected:
                errors.append("search_must_preserve_validated_request")
        if name in {"get_flight_details", "hold_flight"}:
            fid = args.get("flight_id")
            if fid not in state.searched_flights:
                errors.append("flight_id_not_in_search_observations")
            if fid in state.rejected_flights:
                errors.append("flight_previously_unavailable")
        if name == "hold_flight":
            flight = state.flight_details.get(args.get("flight_id"))
            if flight is None:
                errors.append("flight_details_required_before_hold")
            else:
                errors.extend(self.flight(flight, state.request).errors)
            if args.get("passenger_name") != state.request.passenger_name:
                errors.append("passenger_mismatch")
        if name == "book_flight":
            if not state.hold or args.get("hold_id") != state.hold.get("hold_id"):
                errors.append("hold_id_not_in_observations")
            if not state.selected_flight:
                errors.append("selected_flight_missing")
            else:
                errors.extend(self.flight(state.selected_flight, state.request).errors)
        return ValidationResult(not errors, errors)
