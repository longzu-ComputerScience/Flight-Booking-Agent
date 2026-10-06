"""Planner proposes a structured tool plan; executor binds and runs it without per-step LLM calls."""
from langgraph.graph import END, START, StateGraph

from app.agents.runtime import Runtime
from app.state import GraphState


def build_plan_execute(runtime: Runtime):
    def parse(state: GraphState) -> dict:
        runtime.parse_request()
        return {"domain": runtime.state}

    def planner(state: GraphState) -> dict:
        runtime.create_plan()
        return {"domain": runtime.state}

    def executor(state: GraphState) -> dict:
        runtime.node("execute preplanned step")
        domain = runtime.state
        if domain.plan_index >= len(domain.plan):
            runtime.stop("plan_exhausted_without_completion", ["proven_completion"], "Cần bổ sung bước xác nhận kết quả hoặc chuyển người hỗ trợ.")
            return {"domain": domain}
        step = domain.plan[domain.plan_index]
        runtime.log("EXECUTOR", f"{domain.plan_index + 1}: {step.purpose}")
        args = runtime.resolve_plan_args(step.args)
        if args is not None:
            result = runtime.invoke_action({"name": step.tool, "args": args})
            if result.get("status") == "error" and domain.status == "running":
                domain.needs_replan = True
                runtime.log("REPLAN", f"Observation invalidated plan: {result.get('error')}")
            else:
                domain.plan_index += 1
        return {"domain": domain}

    def next_step(state: GraphState) -> str:
        if state["domain"].status != "running":
            return END
        return "planner" if state["domain"].needs_replan else "executor"

    graph = StateGraph(GraphState)
    graph.add_node("parse", parse)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", lambda s: "planner" if s["domain"].status == "running" else END)
    graph.add_conditional_edges("planner", lambda s: "executor" if s["domain"].status == "running" else END)
    graph.add_conditional_edges("executor", next_step)
    return graph.compile()
