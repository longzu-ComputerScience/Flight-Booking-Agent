"""Run one isolated graph; expose state and metrics for tests/evaluation."""
from dataclasses import asdict
from datetime import date
from time import perf_counter
from typing import Any

from langgraph.errors import GraphRecursionError

from app.agents.hybrid_agent import build_hybrid
from app.agents.plan_execute_agent import build_plan_execute
from app.agents.react_agent import build_react
from app.agents.runtime import Runtime
from app.data.flights import build_catalog
from app.evaluation.scenarios import Scenario
from app.models import create_model
from app.state import AgentState
from app.tools.flight_tools import FlightTools

BUILDERS = {"react": build_react, "plan": build_plan_execute, "hybrid": build_hybrid}


def run_scenario(architecture: str, scenario: Scenario, today: date, mode: str = "fake",
                 verbose: bool = True, model_override: Any = None) -> AgentState:
    state = scenario.make_state()
    settings = scenario.settings(today)
    model = model_override if model_override is not None else create_model(mode, scenario.model_behavior)
    service = FlightTools(state, build_catalog(today), today, scenario.tool_fault)
    runtime = Runtime(state, model, service, settings, verbose)
    if verbose:
        print("=" * 76)
        print(f"ARCHITECTURE: {architecture} | MODEL: {mode} | SCENARIO: {scenario.name}")
        print("MOCK ONLY · không mua vé thật · confirmation lấy từ CLI/fixture, không từ model")
        print("=" * 76)
        runtime.log("USER", scenario.user_request)
        runtime.log("AUTHORITY", f"confirmation={scenario.confirmation.confirmed}, hold_allowed={scenario.hold_allowed}")
    started = perf_counter()
    graph = BUILDERS[architecture](runtime)
    # Trần framework cuối cùng cao hơn mọi đường chạy hợp lệ; giới hạn harness dừng trước.
    recursion_limit = max(100, 6 * (settings.max_model_calls + settings.max_tool_calls + settings.max_replans + 10))
    try:
        graph.invoke({"domain": state}, {"recursion_limit": recursion_limit})
    except GraphRecursionError:
        runtime.stop("framework_recursion_limit", ["progress"], "Harness/graph cần kiểm tra; chuyển người hỗ trợ.")
    if state.status == "running":
        runtime.stop("graph_ended_without_completion", ["proven_completion"], "Cần kiểm tra đường kết thúc graph.")
    completed = runtime.completion.check(state)
    if state.status == "done" and not completed.done:
        raise AssertionError("DONE violated completion invariant")
    state.completion_missing = completed.missing_conditions
    state.metrics.elapsed_seconds = round(perf_counter() - started, 6)
    if verbose:
        runtime.log("METRICS", str(asdict(state.metrics)))
    return state
