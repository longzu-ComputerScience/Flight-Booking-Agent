"""LangChain tools backed by a per-run mock inventory and defense-in-depth checks."""
from datetime import date as Date
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.harness.constraints import DataConstraints, in_time_window
from app.harness.permissions import PermissionGuard
from app.state import AgentState


def error(code: str, hint: str, **details: Any) -> dict[str, Any]:
    return {"status": "error", "error": code, "hint": hint, **details}


class FlightTools:
    """No network, payments, or shared global inventory; every run is isolated."""
    def __init__(self, state: AgentState, catalog: dict, today: Date, fault: str = "none"):
        self.state, self.catalog, self.fault = state, catalog, fault
        self.constraints = DataConstraints(today)
        self.guard = PermissionGuard()
        self.holds: dict[str, dict] = {}
        self.bookings: dict[str, dict] = {}
        self.search_attempts = 0
        self.sold_out_injected = False
        self.tools = self._build()
        self.by_name = {t.name: t for t in self.tools}

    def _preflight(self, name: str, args: dict) -> dict | None:
        permission = self.guard.check(name, self.state)
        if not permission.allowed:
            return error("permission_denied", "Obtain scoped user confirmation; do not retry blindly.", reason=permission.reason)
        validation = self.constraints.action(name, args, self.state)
        if not validation.ok:
            return error("data_constraint_failed", "Use validated request and observed identifiers.", errors=validation.errors)
        return None

    def _build(self) -> list[BaseTool]:
        @tool
        def search_flights(origin: str, destination: str, date: str, time_preference: str) -> dict:
            """Search the validated route/date/time. Prices are all-inclusive mock VND fares.

            Budget and checked baggage must be compared against the returned candidates.
            """
            args = dict(origin=origin, destination=destination, date=date, time_preference=time_preference)
            failure = self._preflight("search_flights", args)
            if failure:
                return failure
            self.search_attempts += 1
            if self.fault == "transient_search" and self.search_attempts == 1:
                return error("temporary_search_failure", "Retry the same search once.", retryable=True)
            if self.fault == "persistent_search":
                return error("temporary_search_failure", "Retry at most twice, then hand off.", retryable=True)
            results = [dict(f) for f in self.catalog.values() if (
                f["origin"] == origin and f["destination"] == destination
                and f["departure_time"][:10] == date
                and in_time_window(f["departure_time"], time_preference)
            )]
            return {"status": "ok", "count": len(results), "flights": results,
                    "hint": "Inspect details of a candidate within budget, baggage and seat constraints."}

        @tool
        def get_flight_details(flight_id: str) -> dict:
            """Inspect an ID previously returned by search_flights, including seat availability."""
            if flight_id not in self.catalog:
                return error("flight_not_found", "Call search_flights first.", queried_id=flight_id)
            failure = self._preflight("get_flight_details", {"flight_id": flight_id})
            if failure:
                return failure
            return {"status": "ok", "flight": dict(self.catalog[flight_id])}

        @tool
        def hold_flight(flight_id: str, passenger_name: str) -> dict:
            """Create an idempotent reversible mock hold for an inspected matching flight.

            A hold is not a confirmed booking and never charges money.
            """
            args = dict(flight_id=flight_id, passenger_name=passenger_name)
            failure = self._preflight("hold_flight", args)
            if failure:
                return failure
            if self.fault == "sold_out" and not self.sold_out_injected:
                self.sold_out_injected = True
                self.catalog[flight_id]["seats_left"] = 0
            flight = self.catalog[flight_id]
            if flight["seats_left"] <= 0:
                return error("flight_sold_out", "Choose another observed flight or search again.", flight_id=flight_id)
            for hold in self.holds.values():
                if hold["flight_id"] == flight_id and hold["passenger_name"] == passenger_name:
                    return dict(hold)
            hold = {"status": "held", "hold_id": f"HOLD-{len(self.holds) + 1:03}",
                    "flight_id": flight_id, "passenger_name": passenger_name}
            self.holds[hold["hold_id"]] = hold
            return dict(hold)

        @tool
        def book_flight(hold_id: str) -> dict:
            """Confirm a mock hold ONLY with trusted scoped user confirmation.

            Booking is idempotent per hold; this tool never performs real purchases.
            """
            failure = self._preflight("book_flight", {"hold_id": hold_id})
            if failure:
                return failure
            if hold_id not in self.holds:
                return error("hold_not_found", "Create a valid mock hold before booking.", queried_id=hold_id)
            if hold_id in self.bookings:
                return dict(self.bookings[hold_id])
            hold = self.holds[hold_id]
            flight = self.catalog[hold["flight_id"]]
            if flight["seats_left"] <= 0:
                return error("flight_sold_out", "Choose another flight; this hold cannot be booked.", flight_id=flight["flight_id"])
            booking = {"status": "confirmed", "booking_id": f"BK-{len(self.bookings) + 1:03}",
                       "hold_id": hold_id, "passenger_name": hold["passenger_name"],
                       **{key: flight[key] for key in (
                           "flight_id", "origin", "destination", "departure_time",
                           "arrival_time", "price", "baggage_kg",
                       )}}
            self.bookings[hold_id] = booking
            flight["seats_left"] -= 1
            return dict(booking)

        return [search_flights, get_flight_details, hold_flight, book_flight]
