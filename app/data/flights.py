"""12 flights relative to the run's reference date, with deliberately varied fares."""
from datetime import date, timedelta
from typing import Any

AIRPORTS = {"SGN", "HAN", "DAD", "CXR", "PQC"}


def build_catalog(today: date) -> dict[str, dict[str, Any]]:
    """Fresh catalog per run prevents cross-scenario inventory contamination."""
    rows = [
        ("VN201", "Vietnam Airlines", "SGN", "HAN", 1, "06:00", "08:10", 1_850_000, 23, 5, True),
        ("VJ102", "Vietjet", "SGN", "HAN", 1, "07:00", "09:10", 1_350_000, 0, 8, False),
        ("QH202", "Bamboo", "SGN", "HAN", 1, "09:00", "11:10", 1_750_000, 20, 3, False),
        ("VN203", "Vietnam Airlines", "SGN", "HAN", 1, "10:00", "12:10", 2_300_000, 23, 7, True),
        ("VJ104", "Vietjet", "SGN", "HAN", 1, "11:00", "13:10", 1_650_000, 20, 0, False),
        ("VN205", "Vietnam Airlines", "SGN", "HAN", 1, "14:00", "16:10", 1_650_000, 23, 4, True),
        ("QH206", "Bamboo", "SGN", "HAN", 1, "19:00", "21:10", 1_600_000, 20, 6, False),
        ("VN301", "Vietnam Airlines", "HAN", "SGN", 1, "08:00", "10:10", 1_800_000, 23, 4, True),
        ("VJ401", "Vietjet", "SGN", "DAD", 1, "07:30", "08:50", 950_000, 0, 10, False),
        ("VN402", "Vietnam Airlines", "SGN", "DAD", 1, "09:30", "10:50", 1_450_000, 23, 4, True),
        ("VN501", "Vietnam Airlines", "SGN", "HAN", 2, "08:30", "10:40", 1_900_000, 23, 4, True),
        ("VJ601", "Vietjet", "DAD", "HAN", 1, "10:30", "11:50", 1_100_000, 20, 2, False),
    ]
    catalog = {}
    for fid, airline, origin, dest, offset, dep, arr, price, baggage, seats, refundable in rows:
        day = (today + timedelta(days=offset)).isoformat()
        catalog[fid] = {
            "flight_id": fid, "airline": airline, "origin": origin, "destination": dest,
            "departure_time": f"{day}T{dep}:00+07:00",
            "arrival_time": f"{day}T{arr}:00+07:00", "price": price,
            "baggage_kg": baggage, "seats_left": seats, "refundable": refundable,
        }
    return catalog
