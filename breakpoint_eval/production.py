from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from breakpoint_eval.actual_data import estimate_live_judge_cost, source_urls_from_traces
from breakpoint_eval.ci import build_ci_report, default_regression_gate, select_regression_packs, write_ci_report
from breakpoint_eval.compiler import summarize_dataset
from breakpoint_eval.exporters import export_cases_jsonl, export_lm_eval_task, export_openai_evals
from breakpoint_eval.ingestion import IngestionKind, load_production_traces, traces_to_json
from breakpoint_eval.judges import available_validation_suite
from breakpoint_eval.metrics import compiler_native_metrics
from breakpoint_eval.models import DatasetBundle, Project, ValidationReport
from breakpoint_eval.platform import bundle_to_product_layer, build_failure_clusters, run_local_suite, seed_human_reviews, stable_id
from breakpoint_eval.storage import DuckDBStore
from breakpoint_eval.traces import RawFailureTrace, compile_traces


class ProductionPackBuild(BaseModel):
    output_dir: str
    source: str
    trace_count: int
    base_items: int
    total_cases: int
    validation_passed: int
    external_judges: bool
    estimated_live_cost_usd: float
    regression_pack_count: int
    dataset_id: str
    artifact_paths: list[str] = Field(default_factory=list)


def build_production_regression_pack(
    *,
    source_path: str | Path,
    source: IngestionKind | None = None,
    output_dir: str | Path = "artifacts/production_pack",
    max_records: int = 10,
    variants_per_item: int = 3,
    include_external_judges: bool = False,
    max_live_cost_usd: float = 2.0,
    changed_files: list[str] | None = None,
    risk: str = "medium",
) -> ProductionPackBuild:
    ingestion = load_production_traces(source_path, source=source)
    traces = ingestion.traces[:max_records]
    return build_regression_pack_from_traces(
        traces=traces,
        output_dir=output_dir,
        source_name=str(ingestion.source),
        variants_per_item=variants_per_item,
        include_external_judges=include_external_judges,
        max_live_cost_usd=max_live_cost_usd,
        changed_files=changed_files or [],
        risk=risk,
    )


def build_regression_pack_from_traces(
    *,
    traces: list[RawFailureTrace],
    output_dir: str | Path,
    source_name: str = "production",
    variants_per_item: int = 3,
    include_external_judges: bool = False,
    max_live_cost_usd: float = 2.0,
    changed_files: list[str] | None = None,
    risk: str = "medium",
) -> ProductionPackBuild:
    estimated_cost = estimate_live_judge_cost(len(traces), include_external=include_external_judges)
    if include_external_judges and estimated_cost > max_live_cost_usd:
        raise ValueError(
            f"Estimated live judge cost ${estimated_cost:.4f} exceeds max_live_cost_usd=${max_live_cost_usd:.4f}."
        )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    validator = available_validation_suite(include_external=include_external_judges)
    results = compile_traces(
        traces,
        suite_id=stable_id("suite-production", source_name, [trace.id for trace in traces]),
        validator=validator,
        variants_per_item=variants_per_item,
    )
    items = [result.eval_item for result in results]
    reports = [result.eval_case.validation_report for result in results if result.eval_case.validation_report is not None]
    metrics = summarize_dataset(items, reports, [], dict(Counter(item.category for item in items)))
    metrics.update(
        {
            "dataset_kind": "production_failure_pack",
            "source": source_name,
            "trace_count": len(traces),
            "external_judges": include_external_judges,
            "judge_names": [judge.name for judge in validator.judges],
            "estimated_live_cost_usd": estimated_cost,
        }
    )
    bundle = DatasetBundle(
        dataset_id=stable_id("production-dataset", source_name, [trace.id for trace in traces], variants_per_item, include_external_judges),
        items=items,
        validation_reports=[report for report in reports if isinstance(report, ValidationReport)],
        metrics=metrics,
    )
    project = Project(
        id=stable_id("project", source_name),
        name="BreakPoint Production Failure Pack",
        description="Regression pack compiled from production/private failure traces.",
        metadata={"source": source_name},
    )
    project, version, suite, cases = bundle_to_product_layer(
        bundle,
        project=project,
        suite_name="Production Failure Regression Pack",
        include_variants=True,
    )
    run = run_local_suite(suite, cases, model_name="local:oracle-replay")
    reviews = seed_human_reviews(cases, limit=min(20, len(cases)))
    clusters = build_failure_clusters(project, cases)
    native_metrics = compiler_native_metrics(cases, run, reviews=reviews)
    run.metrics.update(native_metrics)
    packs = select_regression_packs(suite, cases, changed_files=changed_files or [], risk=risk)
    ci_report = build_ci_report(suite, cases, run, reviews=reviews, gate=default_regression_gate(suite), packs=packs)

    traces_to_json(traces, target / "source_traces.json")
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
            "packs": [pack.model_dump(mode="json") for pack in packs],
        },
    )
    export_cases_jsonl(cases, target / "cases.jsonl")
    export_openai_evals(suite, cases, target / "openai_evals.yaml")
    export_lm_eval_task(suite, cases, target / "lm_eval_task.yaml")
    write_ci_report(ci_report, target / "ci_report.json")
    store = DuckDBStore(target / "production.duckdb")
    store.save_bundle(bundle)
    store.save_product_layer(project, version, suite, cases, run=run, reviews=reviews, clusters=clusters, gates=[default_regression_gate(suite)])
    store.close()

    manifest = {
        "dataset_id": bundle.dataset_id,
        "source": source_name,
        "trace_count": len(traces),
        "base_items": len(items),
        "total_cases": len(cases),
        "variants_per_item": variants_per_item,
        "validation_passed": sum(1 for report in reports if report and report.passed),
        "external_judges": include_external_judges,
        "estimated_live_cost_usd": estimated_cost,
        "source_urls": source_urls_from_traces(traces),
        "packs": [pack.model_dump(mode="json") for pack in packs],
        "metrics": metrics,
    }
    _write_json(target / "manifest.json", manifest)
    artifact_paths = [
        str(target / name)
        for name in [
            "source_traces.json",
            "trace2eval_results.json",
            "metrics.json",
            "manifest.json",
            "product.json",
            "cases.jsonl",
            "openai_evals.yaml",
            "lm_eval_task.yaml",
            "ci_report.json",
            "ci_report.md",
        ]
    ]
    return ProductionPackBuild(
        output_dir=str(target),
        source=source_name,
        trace_count=len(traces),
        base_items=len(items),
        total_cases=len(cases),
        validation_passed=sum(1 for report in reports if report and report.passed),
        external_judges=include_external_judges,
        estimated_live_cost_usd=estimated_cost,
        regression_pack_count=len(packs),
        dataset_id=bundle.dataset_id,
        artifact_paths=artifact_paths,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
