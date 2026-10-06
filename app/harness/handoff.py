"""Structured recovery context when the agent cannot safely finish."""
from app.state import AgentState


def build_handoff(state: AgentState, reason: str, missing: list[str], question: str) -> dict:
    return {
        "stop_reason": reason,
        "attempted_actions": list(state.attempted_actions),
        "current_state": {
            "request": state.request.model_dump() if state.request else None,
            "selected_flight_id": (state.selected_flight or {}).get("flight_id"),
            "hold_id": (state.hold or {}).get("hold_id"),
            "booking_id": (state.booking or {}).get("booking_id"),
            "model_calls": state.metrics.model_calls,
            "tool_calls": state.metrics.tool_calls,
            "status": state.status,
        },
        "known_facts": list(state.observations),
        "missing_information": missing,
        "question_for_user_or_human": question,
    }


def format_handoff(handoff: dict) -> str:
    actions = ", ".join(f"{a['tool']}[{a['outcome']}]" for a in handoff["attempted_actions"]) or "chưa gọi tool"
    current = handoff["current_state"]
    return (
        f"Lý do: {handoff['stop_reason']}\n"
        f"Đã thử: {actions}\n"
        f"Trạng thái: {current['model_calls']} model calls, {current['tool_calls']} tool calls; "
        f"hold={current['hold_id']}, booking={current['booking_id']}\n"
        f"Căn cứ: {len(handoff['known_facts'])} observations được giữ trong state/JSON.\n"
        f"Còn thiếu: {', '.join(handoff['missing_information']) or 'cần quyết định bước kế tiếp'}\n"
        f"Hỏi người dùng: {handoff['question_for_user_or_human']}"
    )
