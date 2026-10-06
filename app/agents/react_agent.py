"""ReAct: model -> guarded tool -> observation -> model, with no precomputed plan."""
from langgraph.graph import END, START, StateGraph

from app.agents.runtime import Runtime
from app.state import GraphState


def build_react(runtime: Runtime):
    def parse(state: GraphState) -> dict:
        runtime.parse_request()
        return {"domain": runtime.state}

    def decide(state: GraphState) -> dict:
        runtime.decide()
        return {"domain": runtime.state}

    def execute(state: GraphState) -> dict:
        runtime.node("guarded action")
        runtime.invoke_action(runtime.state.pending)
        runtime.state.pending = None
        return {"domain": runtime.state}

    graph = StateGraph(GraphState)
    graph.add_node("parse", parse)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", lambda s: "decide" if s["domain"].status == "running" else END)
    graph.add_conditional_edges("decide", lambda s: "execute" if s["domain"].pending and s["domain"].status == "running" else END)
    graph.add_conditional_edges("execute", lambda s: "decide" if s["domain"].status == "running" else END)
    return graph.compile()
