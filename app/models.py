"""Both models use BaseChatModel, AIMessage tool_calls and the same graph/harness."""
import json
import os
import re
from datetime import date, timedelta
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from app.harness.constraints import DataConstraints
from app.state import BookingSummary, StrictSchema, TravelRequest


class ModelConfigurationError(ValueError):
    pass


class RequestHelp(StrictSchema):
    reason: str
    missing_information: list[str]
    question: str


class FakeChatModel(BaseChatModel):
    """Nó không phải LLM; nó chỉ mô phỏng quyết định của LLM để demo reproducible.

    Rule-based choices live in this model, not in main.py or the ReAct router.
    It reads tool observations and reacts to errors instead of returning canned results.
    """
    behavior: str = "normal"

    @property
    def _llm_type(self) -> str:
        return "assignment-fake-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        context = json.loads(messages[0].content)
        task, state = context["task"], context["state"]
        if task == "parse":
            name, args = "TravelRequest", self._parse(context)
        elif task == "plan":
            name, args = "ActionPlan", self._plan(state)
        elif task == "strategy":
            stages = (["discover"] if not state["selected_flight"] and not state["booking"] else [])
            stages += (["reserve"] if not state["booking"] else []) + ["finalize"]
            name, args = "Strategy", {"stages": stages,
                                     "rationale": "Inspect feasible candidates, reserve safely, then verify tool-grounded completion."}
        else:
            name, args = self._decide(state, context.get("stage"))
        ai = AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": f"call-{context['call_number']}", "type": "tool_call"
        }])
        return ChatResult(generations=[ChatGeneration(message=ai)])

    @staticmethod
    def _parse(context: dict) -> dict:
        """Small demonstrative parser; unknown fields stay invalid/missing, not invented."""
        text = context["user_request"].lower()
        aliases = {"sgn": "SGN", "tp.hcm": "SGN", "hồ chí minh": "SGN", "sài gòn": "SGN",
                   "han": "HAN", "hà nội": "HAN", "dad": "DAD", "đà nẵng": "DAD",
                   "cxr": "CXR", "cam ranh": "CXR", "pqc": "PQC", "phú quốc": "PQC"}
        matches = sorted((m.start(), code) for alias, code in aliases.items()
                         for m in re.finditer(re.escape(alias), text))
        places = [code for _, code in matches]
        today = date.fromisoformat(context["today"])
        explicit_date = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        day = explicit_date.group() if explicit_date else (
            (today + timedelta(days=1)).isoformat() if "mai" in text else ""
        )
        cap = re.search(r"(\d+(?:[.,]\d+)?)\s*triệu", text)
        baggage = re.search(r"(\d+)\s*kg", text)
        return {
            "origin": places[0] if places else "", "destination": places[-1] if len(places) > 1 else "",
            "travel_date": day, "time_preference": "morning" if "sáng" in text else (
                "afternoon" if "chiều" in text else "evening" if "tối" in text else "any"),
            "max_price": int(float(cap.group(1).replace(",", ".")) * 1_000_000) if cap else 2_000_000,
            "min_baggage_kg": int(baggage.group(1)) if baggage else (15 if "ký gửi" in text else 0),
            "passenger_name": context["passenger_input"],
        }

    @staticmethod
    def _candidate(state: dict) -> dict | None:
        request = TravelRequest.model_validate(state["request"])
        constraints = DataConstraints(date.fromisoformat(request.travel_date))
        candidates = [f for f in state["searched_flights"]
                      if f["flight_id"] not in state["rejected_flights"] and constraints.flight(f, request).ok]
        return min(candidates, key=lambda f: (f["price"], f["departure_time"])) if candidates else None

    @staticmethod
    def _search_args(state: dict) -> dict:
        request = state["request"]
        return {"origin": request["origin"], "destination": request["destination"],
                "date": request["travel_date"], "time_preference": request["time_preference"]}

    def _decide(self, state: dict, stage: str | None) -> tuple[str, dict]:
        if self.behavior == "premature_finish":
            return "BookingSummary", {"booking_id": "BK-INVENTED", "flight_id": "VN999",
                                      "origin": "SGN", "destination": "HAN", "departure_time": "invented",
                                      "arrival_time": "invented", "price": 1_000_000,
                                      "baggage_kg": 23, "passenger_name": state["request"]["passenger_name"]}
        if self.behavior == "loop":
            return "search_flights", self._search_args(state)
        if state["booking"]:
            return "BookingSummary", {key: state["booking"][key] for key in BookingSummary.model_fields}
        if state["hold"]:
            return "book_flight", {"hold_id": state["hold"]["hold_id"]}
        if not state["searched_flights"]:
            last = state["last_observation"]
            if last and last["tool"] == "search_flights" and last["result"].get("status") == "ok":
                return self._no_matches()
            return "search_flights", self._search_args(state)
        if self.behavior == "forged_id":
            return "get_flight_details", {"flight_id": "VN999"}
        candidate = self._candidate(state)
        if not candidate:
            return self._no_matches()
        details = next((f for f in state["flight_details"] if f["flight_id"] == candidate["flight_id"]), None)
        if details is None:
            return "get_flight_details", {"flight_id": candidate["flight_id"]}
        return "hold_flight", {"flight_id": candidate["flight_id"],
                               "passenger_name": state["request"]["passenger_name"]}

    @staticmethod
    def _no_matches() -> tuple[str, dict]:
        return "RequestHelp", {"reason": "no_matching_flight", "missing_information": ["acceptable_flight"],
                               "question": "Không có chuyến phù hợp. Bạn muốn nới ngân sách, giờ bay hay hành lý?"}

    def _plan(self, state: dict) -> dict:
        if self.behavior == "loop":
            return {"steps": [{"tool": "search_flights", "args": self._search_args(state),
                               "purpose": "Retry search"} for _ in range(4)]}
        if self.behavior == "premature_finish":
            tool, args = self._decide(state, None)
            return {"steps": [{"tool": tool, "args": args, "purpose": "Claim completion prematurely"}]}
        steps = []
        if state["booking"]:
            steps.append({"tool": "BookingSummary", "args": {"summary": "$booking"}, "purpose": "Verify completion"})
        elif state["hold"]:
            steps += [{"tool": "book_flight", "args": {"hold_id": "$hold.hold_id"}, "purpose": "Confirm mock hold"},
                      {"tool": "BookingSummary", "args": {"summary": "$booking"}, "purpose": "Verify completion"}]
        else:
            if not state["searched_flights"]:
                steps.append({"tool": "search_flights", "args": self._search_args(state), "purpose": "Find candidates"})
            steps += [
                {"tool": "get_flight_details", "args": {"flight_id": "VN999" if self.behavior == "forged_id" else "$candidate.flight_id"},
                 "purpose": "Inspect cheapest feasible flight"},
                {"tool": "hold_flight", "args": {"flight_id": "$candidate.flight_id", "passenger_name": "$request.passenger_name"},
                 "purpose": "Create reversible mock hold"},
                {"tool": "book_flight", "args": {"hold_id": "$hold.hold_id"}, "purpose": "Book only through permission guard"},
                {"tool": "BookingSummary", "args": {"summary": "$booking"}, "purpose": "Verify completion against receipt"},
            ]
        return {"selection_rule": "cheapest", "steps": steps}


def create_model(mode: str, behavior: str = "normal") -> BaseChatModel:
    if mode == "fake":
        return FakeChatModel(behavior=behavior)
    model_name = os.getenv("SE373_MODEL", "").strip()
    if not model_name or ":" not in model_name:
        raise ModelConfigurationError("Chưa cấu hình SE373_MODEL=provider:model trong .env. Dùng --model fake hoặc xem .env.example.")
    provider = model_name.split(":", 1)[0]
    providers = {"openai": ("OPENAI_API_KEY", "langchain-openai"),
                 "anthropic": ("ANTHROPIC_API_KEY", "langchain-anthropic"),
                 "google_genai": ("GOOGLE_API_KEY", "langchain-google-genai")}
    if provider not in providers:
        raise ModelConfigurationError("Demo hỗ trợ provider openai, anthropic, google_genai; chọn model hỗ trợ tool calling.")
    key, package = providers[provider]
    if not os.getenv(key, "").strip():
        raise ModelConfigurationError(f"Thiếu {key} trong .env. Không cần API key khi dùng --model fake.")
    try:
        return init_chat_model(model_name, temperature=0, timeout=30, max_retries=1)
    except (ImportError, ValueError) as exc:
        raise ModelConfigurationError(f"Không khởi tạo được model. Cài {package}, kiểm tra SE373_MODEL. Loại lỗi: {type(exc).__name__}.") from exc
