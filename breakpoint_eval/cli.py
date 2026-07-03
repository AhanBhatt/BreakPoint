from __future__ import annotations

import argparse
import json
from pathlib import Path

from breakpoint_eval.actual_data import DEFAULT_ACTUAL_TRACES_PATH, build_actual_dataset_artifacts
from breakpoint_eval.benchmark import build_failuregym_manifest
from breakpoint_eval.calibration import calibrate_from_trace2eval_results
from breakpoint_eval.categories import FAILURE_CATEGORIES
from breakpoint_eval.compiler import BreakPointCompiler, to_huggingface_dataset, write_jsonl, write_metrics
from breakpoint_eval.ci import build_ci_report, default_regression_gate, select_regression_packs, write_ci_report
from breakpoint_eval.env import env_flag, load_env
from breakpoint_eval.exporters import export_cases_jsonl, export_lm_eval_task, export_openai_evals
from breakpoint_eval.ingestion import load_production_traces, traces_to_json
from breakpoint_eval.judges import available_validation_suite
from breakpoint_eval.model_runs import run_model_comparison_from_product
from breakpoint_eval.models import EvalCase, EvalSuite
from breakpoint_eval.platform import (
    build_failure_clusters,
    bundle_to_product_layer,
    run_local_suite,
    seed_human_reviews,
)
from breakpoint_eval.production import build_production_regression_pack
from breakpoint_eval.reports import write_live_judge_report
from breakpoint_eval.specs import compile_spec, example_rag_freshness_spec, load_spec, write_spec
from breakpoint_eval.storage import DuckDBStore
from breakpoint_eval.traces import compile_traces, load_traces, sample_failure_traces


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(prog="breakpoint", description="BreakPoint eval data compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="Generate and validate an eval dataset")
    compile_parser.add_argument("--items-per-category", type=int, default=8)
    compile_parser.add_argument("--variants-per-item", type=int, default=3)
    compile_parser.add_argument("--seed", type=int, default=7)
    compile_parser.add_argument("--categories", default="", help="Comma-separated category ids")
    compile_parser.add_argument("--out", default="artifacts/demo/breakpoint_eval.jsonl")
    compile_parser.add_argument("--metrics", default="artifacts/demo/metrics.json")
    compile_parser.add_argument("--duckdb", default="artifacts/demo/breakpoint.duckdb")
    compile_parser.add_argument("--hf-preview", action="store_true", help="Print a Hugging Face Dataset preview")
    compile_parser.add_argument("--external-judges", action="store_true", help="Use configured OpenAI/Anthropic/Gemini/vLLM judge adapters")

    subparsers.add_parser("categories", help="List failure categories")

    spec_parser = subparsers.add_parser("compile-spec", help="Compile a BreakPointSpec YAML file")
    spec_parser.add_argument("--spec", default="", help="Path to BreakPointSpec YAML. If omitted, writes/uses an example.")
    spec_parser.add_argument("--out", default="artifacts/specs/compiled_spec_item.json")
    spec_parser.add_argument("--write-example", default="artifacts/specs/rag_freshness_contradiction.yaml")
    spec_parser.add_argument("--external-judges", action="store_true", help="Validate the compiled spec with configured live judges")

    trace_parser = subparsers.add_parser("trace2eval", help="Compile failure traces into eval seeds and cases")
    trace_parser.add_argument("--traces", default="", help="JSON trace file. If omitted, uses sample traces.")
    trace_parser.add_argument("--out", default="artifacts/trace2eval/results.json")
    trace_parser.add_argument("--external-judges", action="store_true", help="Use configured live judges during Trace2Eval validation")

    product_parser = subparsers.add_parser("product-demo", help="Build product-layer suite/run/review artifacts")
    product_parser.add_argument("--out-dir", default="artifacts/product")
    product_parser.add_argument("--external-judges", action="store_true", help="Use configured live judges while compiling demo cases")

    ci_parser = subparsers.add_parser("ci-report", help="Build BreakPoint CI regression report")
    ci_parser.add_argument("--out", default="artifacts/ci/breakpoint_report.json")
    ci_parser.add_argument("--external-judges", action="store_true", help="Use configured live judges while compiling CI cases")

    failuregym_parser = subparsers.add_parser("failuregym", help="Build BreakPoint-FailureGym benchmark package")
    failuregym_parser.add_argument("--out-dir", default="artifacts/failuregym")
    failuregym_parser.add_argument("--external-judges", action="store_true", help="Use configured live judges while compiling FailureGym")

    actual_parser = subparsers.add_parser("actual-data", help="Compile sourced public failure traces into actual eval data")
    actual_parser.add_argument("--source", default=str(DEFAULT_ACTUAL_TRACES_PATH), help="Path to normalized actual failure traces JSON")
    actual_parser.add_argument("--out-dir", default="artifacts/actual")
    actual_parser.add_argument("--max-records", type=int, default=0, help="Limit source traces. 0 means all records.")
    actual_parser.add_argument("--variants-per-item", type=int, default=3)
    actual_parser.add_argument("--external-judges", action="store_true", help="Use configured OpenAI/Anthropic/Gemini/vLLM judges")
    actual_parser.add_argument("--max-live-cost-usd", type=float, default=2.0, help="Estimated live judge budget guard")

    report_parser = subparsers.add_parser("judge-report", help="Build live judge evaluation charts from Trace2Eval results")
    report_parser.add_argument("--results", default="artifacts/actual/trace2eval_results.json")
    report_parser.add_argument("--out-dir", default="artifacts/reports")

    ingest_parser = subparsers.add_parser("ingest-traces", help="Normalize production logs into BreakPoint RawFailureTrace JSON")
    ingest_parser.add_argument("--input", required=True)
    ingest_parser.add_argument("--source", default="", help="Adapter: opentelemetry, langsmith, openinference, litellm, retrieval_log, support_ticket, user_feedback, red_team, incident_report")
    ingest_parser.add_argument("--out", default="artifacts/ingestion/traces.json")
    ingest_parser.add_argument("--report", default="artifacts/ingestion/report.json")

    production_parser = subparsers.add_parser("production-pack", help="Build a regression pack from private/production traces")
    production_parser.add_argument("--input", required=True)
    production_parser.add_argument("--source", default="")
    production_parser.add_argument("--out-dir", default="artifacts/production_pack")
    production_parser.add_argument("--max-records", type=int, default=10)
    production_parser.add_argument("--variants-per-item", type=int, default=3)
    production_parser.add_argument("--external-judges", action="store_true")
    production_parser.add_argument("--max-live-cost-usd", type=float, default=2.0)
    production_parser.add_argument("--changed-files", default="", help="Comma-separated changed files for risk-focused pack selection")
    production_parser.add_argument("--risk", default="medium", choices=["low", "medium", "high"])

    calibration_parser = subparsers.add_parser("calibrate-judges", help="Calibrate live judges against human/gold labels")
    calibration_parser.add_argument("--results", default="artifacts/actual/trace2eval_results.json")
    calibration_parser.add_argument("--labels", default="")
    calibration_parser.add_argument("--out-dir", default="artifacts/calibration")
    calibration_parser.add_argument("--min-accuracy", type=float, default=0.78)

    model_runs_parser = subparsers.add_parser("model-runs", help="Run model-under-test profiles against compiled cases")
    model_runs_parser.add_argument("--product", default="artifacts/actual/product.json")
    model_runs_parser.add_argument("--out-dir", default="artifacts/model_runs")

    ci_packs_parser = subparsers.add_parser("ci-packs", help="Select regression packs from product artifacts and changed-file risk")
    ci_packs_parser.add_argument("--product", default="artifacts/actual/product.json")
    ci_packs_parser.add_argument("--out", default="artifacts/ci/packs.json")
    ci_packs_parser.add_argument("--changed-files", default="")
    ci_packs_parser.add_argument("--risk", default="medium", choices=["low", "medium", "high"])

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI development server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    vllm_parser = subparsers.add_parser("vllm-judge-server", help="Run the local OpenAI-compatible judge endpoint")
    vllm_parser.add_argument("--host", default="127.0.0.1")
    vllm_parser.add_argument("--port", type=int, default=8001)

    args = parser.parse_args()
    if args.command == "compile":
        selected = [part.strip() for part in args.categories.split(",") if part.strip()] or None
        compiler = BreakPointCompiler(seed=args.seed, validator=_validator_from_args(args))
        bundle = compiler.compile_dataset(
            categories=selected,
            items_per_category=args.items_per_category,
            variants_per_item=args.variants_per_item,
        )
        write_jsonl(bundle.items, args.out)
        write_metrics(bundle.metrics, args.metrics)
        store = DuckDBStore(args.duckdb)
        store.save_bundle(bundle)
        store.close()
        print(json.dumps({"dataset_id": bundle.dataset_id, **bundle.metrics}, indent=2))
        if args.hf_preview:
            print(to_huggingface_dataset(bundle.items).select(range(min(3, len(bundle.items)))))
    elif args.command == "categories":
        print(json.dumps([spec.model_dump() for spec in FAILURE_CATEGORIES.values()], indent=2))
    elif args.command == "compile-spec":
        spec_path = Path(args.spec) if args.spec else Path(args.write_example)
        if not spec_path.exists():
            write_spec(example_rag_freshness_spec(), spec_path)
        item = compile_spec(load_spec(spec_path))
        report = _validator_from_args(args).validate(item)
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = item.to_dataset_row()
        payload["validation_report"] = report.model_dump(mode="json")
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"spec": str(spec_path), "out": str(target), "variants": len(item.adversarial_variants), "passed": report.passed},
                indent=2,
            )
        )
    elif args.command == "trace2eval":
        traces = load_traces(args.traces) if args.traces else sample_failure_traces()
        results = compile_traces(traces, validator=_validator_from_args(args))
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([result.model_dump(mode="json") for result in results], indent=2), encoding="utf-8")
        print(json.dumps({"out": str(target), "seeds": len(results)}, indent=2))
    elif args.command == "product-demo":
        bundle = BreakPointCompiler(seed=42, validator=_validator_from_args(args)).compile_dataset(items_per_category=3, variants_per_item=2)
        project, version, suite, cases = bundle_to_product_layer(bundle)
        run = run_local_suite(suite, cases)
        reviews = seed_human_reviews(cases)
        clusters = build_failure_clusters(project, cases)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "product.json").write_text(
            json.dumps(
                {
                    "project": project.model_dump(mode="json"),
                    "dataset_version": version.model_dump(mode="json"),
                    "suite": suite.model_dump(mode="json"),
                    "cases": [case.model_dump(mode="json") for case in cases],
                    "run": run.model_dump(mode="json"),
                    "reviews": [review.model_dump(mode="json") for review in reviews],
                    "clusters": [cluster.model_dump(mode="json") for cluster in clusters],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        export_cases_jsonl(cases, out_dir / "cases.jsonl")
        export_openai_evals(suite, cases, out_dir / "openai_evals.yaml")
        export_lm_eval_task(suite, cases, out_dir / "lm_eval_task.yaml")
        print(json.dumps({"out_dir": str(out_dir), "cases": len(cases)}, indent=2))
    elif args.command == "ci-report":
        bundle = BreakPointCompiler(seed=42, validator=_validator_from_args(args)).compile_dataset(items_per_category=2, variants_per_item=2)
        _, _, suite, cases = bundle_to_product_layer(bundle)
        run = run_local_suite(suite, cases)
        report = build_ci_report(suite, cases, run, reviews=seed_human_reviews(cases), gate=default_regression_gate(suite))
        write_ci_report(report, args.out)
        print(json.dumps({"out": args.out, "passed": report["passed"]}, indent=2))
    elif args.command == "failuregym":
        bundle = BreakPointCompiler(seed=42, validator=_validator_from_args(args)).compile_dataset(items_per_category=3, variants_per_item=2)
        _, _, suite, cases = bundle_to_product_layer(bundle)
        manifest = build_failuregym_manifest(suite, cases, args.out_dir)
        print(json.dumps(manifest, indent=2))
    elif args.command == "actual-data":
        result = build_actual_dataset_artifacts(
            source_path=args.source,
            output_dir=args.out_dir,
            max_records=args.max_records or None,
            variants_per_item=args.variants_per_item,
            include_external_judges=bool(args.external_judges) or env_flag("BREAKPOINT_EXTERNAL_JUDGES", False),
            max_live_cost_usd=args.max_live_cost_usd,
        )
        print(result.model_dump_json(indent=2))
    elif args.command == "judge-report":
        report = write_live_judge_report(args.results, args.out_dir)
        print(
            json.dumps(
                {
                    "out_dir": args.out_dir,
                    "trace_count": report["trace_count"],
                    "judge_count": report["judge_count"],
                    "top_judge": report["judge_summaries"][0]["judge"] if report["judge_summaries"] else None,
                    "indices": report["breakpoint_indices"],
                },
                indent=2,
            )
        )
    elif args.command == "ingest-traces":
        source = args.source or None
        report = load_production_traces(args.input, source=source)
        traces_to_json(report.traces, args.out)
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(json.dumps({"out": args.out, "report": args.report, "source": report.source, "trace_count": report.trace_count}, indent=2))
    elif args.command == "production-pack":
        result = build_production_regression_pack(
            source_path=args.input,
            source=args.source or None,
            output_dir=args.out_dir,
            max_records=args.max_records,
            variants_per_item=args.variants_per_item,
            include_external_judges=bool(args.external_judges) or env_flag("BREAKPOINT_EXTERNAL_JUDGES", False),
            max_live_cost_usd=args.max_live_cost_usd,
            changed_files=[part.strip() for part in args.changed_files.split(",") if part.strip()],
            risk=args.risk,
        )
        print(result.model_dump_json(indent=2))
    elif args.command == "calibrate-judges":
        report = calibrate_from_trace2eval_results(
            args.results,
            labels_path=args.labels or None,
            out_dir=args.out_dir,
            min_accuracy=args.min_accuracy,
        )
        print(report.model_dump_json(indent=2))
    elif args.command == "model-runs":
        comparison = run_model_comparison_from_product(args.product, out_dir=args.out_dir)
        print(json.dumps({"out_dir": args.out_dir, "case_count": comparison.case_count, "summary": comparison.summary}, indent=2))
    elif args.command == "ci-packs":
        data = json.loads(Path(args.product).read_text(encoding="utf-8"))
        suite = EvalSuite.model_validate(data["suite"])
        cases = [EvalCase.model_validate(case) for case in data["cases"]]
        packs = select_regression_packs(
            suite,
            cases,
            changed_files=[part.strip() for part in args.changed_files.split(",") if part.strip()],
            risk=args.risk,
        )
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([pack.model_dump(mode="json") for pack in packs], indent=2), encoding="utf-8")
        print(json.dumps({"out": args.out, "packs": len(packs)}, indent=2))
    elif args.command == "vllm-judge-server":
        import uvicorn

        uvicorn.run("breakpoint_eval.vllm_server:app", host=args.host, port=args.port, reload=False)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("breakpoint_eval.api:app", host=args.host, port=args.port, reload=True)


def _validator_from_args(args: argparse.Namespace):
    include_external = bool(getattr(args, "external_judges", False)) or env_flag("BREAKPOINT_EXTERNAL_JUDGES", False)
    return available_validation_suite(include_external=include_external)


if __name__ == "__main__":
    main()
