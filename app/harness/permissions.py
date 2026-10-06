"""Side-effect policy: the model never grants its own permissions."""
from dataclasses import dataclass

from app.state import AgentState


@dataclass
class PermissionResult:
    allowed: bool
    reason: str


class PermissionGuard:
    def check(self, action: str, state: AgentState) -> PermissionResult:
        if action in {"search_flights", "get_flight_details"}:
            return PermissionResult(True, "read_only")
        if action == "hold_flight":
            # Giữ chỗ là thao tác mock, có thể đảo ngược; người dùng cho phép bằng policy.
            return PermissionResult(state.hold_allowed, "mock_hold_allowed" if state.hold_allowed else "hold_not_authorized")
        if action != "book_flight":
            return PermissionResult(False, "unknown_action_denied")
        receipt, request = state.confirmation, state.request
        if not receipt.confirmed:
            return PermissionResult(False, "user_confirmation_missing")
        if request is None:
            return PermissionResult(False, "validated_request_missing")
        if (receipt.origin, receipt.destination, receipt.travel_date, receipt.passenger_name) != (
            request.origin, request.destination, request.travel_date, request.passenger_name
        ):
            return PermissionResult(False, "confirmation_scope_mismatch")
        if request.max_price > receipt.max_price:
            return PermissionResult(False, "request_exceeds_confirmed_budget")
        if state.selected_flight and state.selected_flight["price"] > receipt.max_price:
            return PermissionResult(False, "fare_exceeds_confirmed_budget")
        return PermissionResult(True, "user_confirmation_valid_for_this_request")
