"""High-level planning + local ReAct; reserve failures can return to discovery via replan."""
from langgraph.graph import END, START, StateGraph

from app.agents.runtime import Runtime
from app.state import GraphState


def build_hybrid(runtime: Runtime):
    def parse(state: GraphState) -> dict:
        runtime.parse_request()
        return {"domain": runtime.state}

    def planner(state: GraphState) -> dict:
        runtime.create_plan(hybrid=True)
        return {"domain": runtime.state}

    def stage_router(state: GraphState) -> dict:
        runtime.node("stage completion checker")
        domain = runtime.state
        while domain.stage_index < len(domain.stages) and runtime.stage_complete(domain.stages[domain.stage_index]):
            runtime.log("STAGE", f"{domain.stages[domain.stage_index]} complete by code")
            domain.stage_index += 1
        if domain.stage_index >= len(domain.stages) and domain.status == "running":
            runtime.stop("strategy_exhausted_without_completion", ["proven_completion"], "Cần kiểm tra kết quả với biên nhận tool.")
        return {"domain": domain}

    def decide(state: GraphState) -> dict:
        runtime.decide(runtime.state.stages[runtime.state.stage_index])
        return {"domain": runtime.state}

    def execute(state: GraphState) -> dict:
        runtime.node("stage-local guarded action")
        domain = runtime.state
        result = runtime.invoke_action(domain.pending, domain.stages[domain.stage_index])
        domain.pending = None
        if result.get("error") == "flight_sold_out" and domain.status == "running":
            domain.needs_replan = True
            runtime.log("REPLAN", "Flight became unavailable; return to discover with updated facts.")
        return {"domain": domain}

    graph = StateGraph(GraphState)
    for name, node in [("parse", parse), ("planner", planner), ("stage_router", stage_router),
                       ("decide", decide), ("execute", execute)]:
        graph.add_node(name, node)
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", lambda s: "planner" if s["domain"].status == "running" else END)
    graph.add_conditional_edges("planner", lambda s: "stage_router" if s["domain"].status == "running" else END)
    graph.add_conditional_edges("stage_router", lambda s: "decide" if s["domain"].status == "running" else END)
    graph.add_conditional_edges("decide", lambda s: "execute" if s["domain"].pending and s["domain"].status == "running" else END)
    graph.add_conditional_edges("execute", lambda s: END if s["domain"].status != "running" else (
        "planner" if s["domain"].needs_replan else "stage_router"
    ))
    return graph.compile()
