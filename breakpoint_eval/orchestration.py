from __future__ import annotations

from typing import Any, TypedDict

from breakpoint_eval.compiler import BreakPointCompiler


class CompileState(TypedDict, total=False):
    categories: list[str]
    items_per_category: int
    variants_per_item: int
    dataset_id: str
    metrics: dict[str, Any]
    trace: list[str]


def run_builtin_graph(state: CompileState) -> CompileState:
    trace = ["load_categories", "generate_candidates", "mutate_cases", "validate_judges", "filter_export"]
    compiler = BreakPointCompiler()
    bundle = compiler.compile_dataset(
        categories=state.get("categories"),
        items_per_category=state.get("items_per_category", 4),
        variants_per_item=state.get("variants_per_item", 2),
    )
    return {**state, "dataset_id": bundle.dataset_id, "metrics": bundle.metrics, "trace": trace}


def run_langgraph_if_available(state: CompileState) -> CompileState:
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return run_builtin_graph(state)

    def compile_node(inner_state: CompileState) -> CompileState:
        return run_builtin_graph(inner_state)

    graph = StateGraph(CompileState)
    graph.add_node("compile", compile_node)
    graph.set_entry_point("compile")
    graph.add_edge("compile", END)
    app = graph.compile()
    return app.invoke(state)
