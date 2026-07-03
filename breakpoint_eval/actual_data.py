from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from breakpoint_eval.ci import build_ci_report, default_regression_gate, write_ci_report
from breakpoint_eval.compiler import summarize_dataset
from breakpoint_eval.env import env_flag, load_env
from breakpoint_eval.exporters import export_cases_jsonl, export_lm_eval_task, export_openai_evals
from breakpoint_eval.judges import available_validation_suite
from breakpoint_eval.metrics import compiler_native_metrics
from breakpoint_eval.models import DatasetBundle, Project, ValidationReport
from breakpoint_eval.platform import (
    build_failure_clusters,
    bundle_to_product_layer,
    evaluate_gate,
    run_local_suite,
    seed_human_reviews,
    stable_id,
)
from breakpoint_eval.storage import DuckDBStore
from breakpoint_eval.traces import RawFailureTrace, compile_traces, load_traces


DEFAULT_ACTUAL_TRACES_PATH = Path("data/actual/failure_traces.json")
DEFAULT_ACTUAL_OUTPUT_DIR = Path("artifacts/actual")


MODEL_RATE_USD_PER_MTOK = {
    "openai": {"input": 0.75, "output": 4.50},
    "anthropic": {"input": 2.00, "output": 10.00},
    "gemini": {"input": 0.30, "output": 2.50},
}


class ActualDatasetBuild(BaseModel):
    output_dir: str
    trace_count: int
    base_items: int
    total_cases: int
    validation_passed: int
    external_judges: bool
    estimated_live_cost_usd: float
    judge_names: list[str] = Field(default_factory=list)
    manifest_path: str
    dataset_id: str
    source_urls: list[str] = Field(default_factory=list)


def load_actual_failure_traces(path: str | Path = DEFAULT_ACTUAL_TRACES_PATH) -> list[RawFailureTrace]:
    return load_traces(path)


def build_actual_dataset_artifacts(
    *,
    source_path: str | Path = DEFAULT_ACTUAL_TRACES_PATH,
    output_dir: str | Path = DEFAULT_ACTUAL_OUTPUT_DIR,
    max_records: int | None = None,
    variants_per_item: int = 3,
    include_external_judges: bool | None = None,
    max_live_cost_usd: float = 2.0,
) -> ActualDatasetBuild:
    load_env()
    include_external = env_flag("BREAKPOINT_EXTERNAL_JUDGES", False) if include_external_judges is None else include_external_judges
    traces = load_actual_failure_traces(source_path)
    if max_records:
        traces = traces[:max_records]

    estimated_cost = estimate_live_judge_cost(len(traces), include_external=include_external)
    if include_external and estimated_cost > max_live_cost_usd:
        raise ValueError(
            f"Estimated live judge cost ${estimated_cost:.4f} exceeds max_live_cost_usd=${max_live_cost_usd:.4f}. "
            "Raise --max-live-cost-usd to continue."
        )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    validator = available_validation_suite(include_external=include_external)
    results = compile_traces(
        traces,
        suite_id="suite-actual-public-failures",
        validator=validator,
        variants_per_item=variants_per_item,
    )
    items = [result.eval_item for result in results]
    reports = [result.eval_case.validation_report for result in results if result.eval_case.validation_report is not None]
    attempts_by_category = dict(Counter(item.category for item in items))
    metrics = summarize_dataset(items, reports, [], attempts_by_category)
    metrics.update(
        {
            "dataset_kind": "actual_public_failure_corpus",
            "trace_count": len(traces),
            "source_count": len(source_urls_from_traces(traces)),
            "external_judges": include_external,
            "judge_names": [judge.name for judge in validator.judges],
            "estimated_live_cost_usd": estimated_cost,
            "compiler_runtime_mode": "actual-trace-derived-live" if include_external else "actual-trace-derived-local",
        }
    )
    bundle = DatasetBundle(
        dataset_id=stable_id("actual-dataset", [trace.id for trace in traces], variants_per_item, include_external),
        items=items,
        validation_reports=[report for report in reports if isinstance(report, ValidationReport)],
        metrics=metrics,
    )

    project = Project(
        id="project-breakpoint-actual",
        name="BreakPoint Actual Failure Corpus",
        description="Trace-derived regression suite compiled from public, sourced LLM failure incidents.",
        metadata={"source_path": str(source_path)},
    )
    project, version, suite, cases = bundle_to_product_layer(
        bundle,
        project=project,
        suite_name="Actual Failure Regression Suite",
        include_variants=True,
    )
    run = run_local_suite(suite, cases, model_name="local:oracle-replay")
    reviews = seed_human_reviews(cases, limit=min(16, len(cases)))
    clusters = build_failure_clusters(project, cases)
    native_metrics = compiler_native_metrics(cases, run, reviews=reviews)
    run.metrics.update(native_metrics)
    gate = default_regression_gate(suite)
    gate_result = evaluate_gate(gate, run)
    ci_report = build_ci_report(suite, cases, run, reviews=reviews, gate=gate)

    _write_json(target / "source_traces.json", {"traces": [trace.model_dump(mode="json") for trace in traces]})
    _write_json(target / "trace2eval_results.json", [result.model_dump(mode="json") for result in results])
    _write_json(target / "metrics.json", metrics)
    _write_json(
        target / "product.json",
        {
            "project": project.model_dump(mode="json"),
            "dataset_version": version.model_dump(mode="json"),
            "suite": suite.model_dump(mode="json"),
            "cases": [case.model_dump(mode="json") for case in cases],
            "run": run.model_dump(mode="json"),
            "reviews": [review.model_dump(mode="json") for review in reviews],
            "clusters": [cluster.model_dump(mode="json") for cluster in clusters],
            "gate": gate.model_dump(mode="json"),
            "gate_result": gate_result,
            "native_metrics": native_metrics,
        },
    )
    export_cases_jsonl(cases, target / "cases.jsonl")
    export_openai_evals(suite, cases, target / "openai_evals.yaml")
    export_lm_eval_task(suite, cases, target / "lm_eval_task.yaml")
    write_ci_report(ci_report, target / "ci_report.json")
    store = DuckDBStore(target / "actual.duckdb")
    store.save_bundle(bundle)
    store.save_product_layer(project, version, suite, cases, run=run, reviews=reviews, clusters=clusters, gates=[gate])
    store.close()

    manifest = build_manifest(
        traces=traces,
        bundle=bundle,
        cases_count=len(cases),
        reports=reports,
        include_external=include_external,
        judge_names=[judge.name for judge in validator.judges],
        estimated_cost=estimated_cost,
        variants_per_item=variants_per_item,
    )
    _write_json(target / "manifest.json", manifest)

    return ActualDatasetBuild(
        output_dir=str(target),
        trace_count=len(traces),
        base_items=len(items),
        total_cases=len(cases),
        validation_passed=sum(1 for report in reports if report and report.passed),
        external_judges=include_external,
        estimated_live_cost_usd=estimated_cost,
        judge_names=[judge.name for judge in validator.judges],
        manifest_path=str(target / "manifest.json"),
        dataset_id=bundle.dataset_id,
        source_urls=source_urls_from_traces(traces),
    )


def build_manifest(
    *,
    traces: list[RawFailureTrace],
    bundle: DatasetBundle,
    cases_count: int,
    reports: list[ValidationReport | None],
    include_external: bool,
    judge_names: list[str],
    estimated_cost: float,
    variants_per_item: int,
) -> dict[str, Any]:
    return {
        "corpus": "breakpoint-public-failure-corpus-v1",
        "dataset_id": bundle.dataset_id,
        "dataset_kind": "actual_public_failure_corpus",
        "trace_count": len(traces),
        "base_items": len(bundle.items),
        "total_cases": cases_count,
        "variants_per_item": variants_per_item,
        "validation_passed": sum(1 for report in reports if report and report.passed),
        "external_judges": include_external,
        "judge_names": judge_names,
        "estimated_live_cost_usd": estimated_cost,
        "source_urls": source_urls_from_traces(traces),
        "sources": [
            {
                "trace_id": trace.id,
                "summary": trace.failure_summary,
                "domain": trace.metadata.get("domain", trace.source),
                "severity": trace.incident_severity,
                "source_urls": trace.metadata.get("source_urls", []),
            }
            for trace in traces
        ],
        "metrics": bundle.metrics,
    }


def source_urls_from_traces(traces: list[RawFailureTrace]) -> list[str]:
    urls: set[str] = set()
    for trace in traces:
        for url in trace.metadata.get("source_urls", []):
            urls.add(str(url))
        for doc in trace.retrieved_docs:
            url = doc.metadata.get("source_url")
            if url:
                urls.add(str(url))
    return sorted(urls)


def estimate_live_judge_cost(
    trace_count: int,
    *,
    include_external: bool,
    avg_input_tokens: int = 1800,
    avg_output_tokens: int = 180,
) -> float:
    if not include_external or trace_count <= 0:
        return 0.0
    total = 0.0
    for rates in MODEL_RATE_USD_PER_MTOK.values():
        total += trace_count * ((avg_input_tokens / 1_000_000) * rates["input"])
        total += trace_count * ((avg_output_tokens / 1_000_000) * rates["output"])
    return round(total, 4)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
