from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from breakpoint_eval.categories import FAILURE_CATEGORIES
from breakpoint_eval.compiler import BreakPointCompiler


app = FastAPI(
    title="BreakPoint API",
    version="0.1.0",
    description="Compile high-quality LLM eval datasets from real failure modes.",
)


class CompileRequest(BaseModel):
    categories: list[str] | None = None
    items_per_category: int = Field(default=3, ge=1, le=50)
    variants_per_item: int = Field(default=2, ge=0, le=5)
    seed: int = 7


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "BreakPoint", "mode": "deterministic-local"}


@app.get("/categories")
def categories() -> list[dict[str, object]]:
    return [spec.model_dump() for spec in FAILURE_CATEGORIES.values()]


@app.post("/compile")
def compile_dataset(request: CompileRequest) -> dict[str, object]:
    compiler = BreakPointCompiler(seed=request.seed)
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


@app.get("/metrics")
def metrics() -> dict[str, object]:
    metrics_path = Path("artifacts/demo/metrics.json")
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    compiler = BreakPointCompiler()
    return compiler.compile_dataset(items_per_category=1, variants_per_item=1).metrics
