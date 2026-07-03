from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from breakpoint_eval.actual_data import DEFAULT_ACTUAL_TRACES_PATH, build_actual_dataset_artifacts
from breakpoint_eval.categories import FAILURE_CATEGORIES
from breakpoint_eval.compiler import BreakPointCompiler
from breakpoint_eval.env import env_flag, load_env
from breakpoint_eval.judges import available_validation_suite
from breakpoint_eval.platform import bundle_to_product_layer, build_failure_clusters, run_local_suite, seed_human_reviews
from breakpoint_eval.specs import parse_spec, compile_spec
from breakpoint_eval.traces import RawFailureTrace, compile_trace


load_env()

app = FastAPI(
    title="BreakPoint API",
    version="0.3.0",
    description="Compile high-quality LLM eval datasets from real failure modes.",
)


class CompileRequest(BaseModel):
    categories: list[str] | None = None
    items_per_category: int = Field(default=3, ge=1, le=50)
    variants_per_item: int = Field(default=2, ge=0, le=5)
    seed: int = 7
    external_judges: bool = False


class SpecCompileRequest(BaseModel):
    yaml: str
    variants_per_item: int = Field(default=4, ge=0, le=10)
    external_judges: bool = False


class TraceCompileRequest(BaseModel):
    trace: RawFailureTrace
    suite_id: str = "suite-trace2eval-api"
    external_judges: bool = False


class ActualCompileRequest(BaseModel):
    source_path: str = str(DEFAULT_ACTUAL_TRACES_PATH)
    out_dir: str = "artifacts/actual"
    max_records: int | None = Field(default=None, ge=1, le=100)
    variants_per_item: int = Field(default=3, ge=0, le=10)
    external_judges: bool = False
    max_live_cost_usd: float = Field(default=2.0, ge=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "BreakPoint", "mode": "deterministic-local"}


@app.get("/categories")
def categories() -> list[dict[str, object]]:
    return [spec.model_dump() for spec in FAILURE_CATEGORIES.values()]


@app.post("/compile")
def compile_dataset(request: CompileRequest) -> dict[str, object]:
    compiler = BreakPointCompiler(seed=request.seed, validator=available_validation_suite(include_external=request.external_judges))
    bundle = compiler.compile_dataset(
        categories=request.categories,
        items_per_category=request.items_per_category,
        variants_per_item=request.variants_per_item,
    )
    return {
        "dataset_id": bundle.dataset_id,
        "metrics": bundle.metrics,
        "preview": [item.to_dataset_row() for item in bundle.items[:3]],
    }


@app.post("/specs/compile")
def compile_breakpoint_spec(request: SpecCompileRequest) -> dict[str, object]:
    spec = parse_spec(request.yaml)
    item = compile_spec(spec, variants_per_item=request.variants_per_item)
    report = available_validation_suite(include_external=request.external_judges).validate(item)
    return {
        "spec_id": spec.id,
        "item": item.to_dataset_row(),
        "validation_report": report.model_dump(mode="json"),
    }


@app.post("/traces/compile")
def compile_failure_trace(request: TraceCompileRequest) -> dict[str, object]:
    result = compile_trace(
        request.trace,
        suite_id=request.suite_id,
        validator=available_validation_suite(include_external=request.external_judges),
    )
    return result.model_dump(mode="json")


@app.post("/actual/compile")
def compile_actual_failure_corpus(request: ActualCompileRequest) -> dict[str, object]:
    result = build_actual_dataset_artifacts(
        source_path=request.source_path,
        output_dir=request.out_dir,
        max_records=request.max_records,
        variants_per_item=request.variants_per_item,
        include_external_judges=request.external_judges or env_flag("BREAKPOINT_EXTERNAL_JUDGES", False),
        max_live_cost_usd=request.max_live_cost_usd,
    )
    return result.model_dump(mode="json")


@app.get("/product/demo")
def product_demo(external_judges: bool = Query(default=False)) -> dict[str, object]:
    compiler = BreakPointCompiler(seed=42, validator=available_validation_suite(include_external=external_judges))
    bundle = compiler.compile_dataset(items_per_category=2, variants_per_item=2)
    project, version, suite, cases = bundle_to_product_layer(bundle)
    run = run_local_suite(suite, cases)
    clusters = build_failure_clusters(project, cases)
    reviews = seed_human_reviews(cases)
    return {
        "project": project.model_dump(mode="json"),
        "dataset_version": version.model_dump(mode="json"),
        "suite": suite.model_dump(mode="json"),
        "case_count": len(cases),
        "run": run.model_dump(mode="json"),
        "clusters": [cluster.model_dump(mode="json") for cluster in clusters],
        "reviews": [review.model_dump(mode="json") for review in reviews],
    }


@app.get("/metrics")
def metrics() -> dict[str, object]:
    metrics_path = Path("artifacts/demo/metrics.json")
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    compiler = BreakPointCompiler()
    return compiler.compile_dataset(items_per_category=1, variants_per_item=1).metrics
