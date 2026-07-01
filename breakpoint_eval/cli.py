from __future__ import annotations

import argparse
import json
from pathlib import Path

from breakpoint_eval.benchmark import build_failuregym_manifest
from breakpoint_eval.categories import FAILURE_CATEGORIES
from breakpoint_eval.compiler import BreakPointCompiler, to_huggingface_dataset, write_jsonl, write_metrics
from breakpoint_eval.ci import build_ci_report, default_regression_gate, write_ci_report
from breakpoint_eval.exporters import export_cases_jsonl, export_lm_eval_task, export_openai_evals
from breakpoint_eval.platform import (
    build_failure_clusters,
    bundle_to_product_layer,
    run_local_suite,
    seed_human_reviews,
)
from breakpoint_eval.specs import compile_spec, example_rag_freshness_spec, load_spec, write_spec
from breakpoint_eval.storage import DuckDBStore
from breakpoint_eval.traces import compile_traces, load_traces, sample_failure_traces


def main() -> None:
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

    subparsers.add_parser("categories", help="List failure categories")

    spec_parser = subparsers.add_parser("compile-spec", help="Compile a BreakPointSpec YAML file")
    spec_parser.add_argument("--spec", default="", help="Path to BreakPointSpec YAML. If omitted, writes/uses an example.")
    spec_parser.add_argument("--out", default="artifacts/specs/compiled_spec_item.json")
    spec_parser.add_argument("--write-example", default="artifacts/specs/rag_freshness_contradiction.yaml")

    trace_parser = subparsers.add_parser("trace2eval", help="Compile failure traces into eval seeds and cases")
    trace_parser.add_argument("--traces", default="", help="JSON trace file. If omitted, uses sample traces.")
    trace_parser.add_argument("--out", default="artifacts/trace2eval/results.json")

    product_parser = subparsers.add_parser("product-demo", help="Build product-layer suite/run/review artifacts")
    product_parser.add_argument("--out-dir", default="artifacts/product")

    ci_parser = subparsers.add_parser("ci-report", help="Build BreakPoint CI regression report")
    ci_parser.add_argument("--out", default="artifacts/ci/breakpoint_report.json")

    failuregym_parser = subparsers.add_parser("failuregym", help="Build BreakPoint-FailureGym benchmark package")
    failuregym_parser.add_argument("--out-dir", default="artifacts/failuregym")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI development server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "compile":
        selected = [part.strip() for part in args.categories.split(",") if part.strip()] or None
        compiler = BreakPointCompiler(seed=args.seed)
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
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(item.to_dataset_row(), indent=2), encoding="utf-8")
        print(json.dumps({"spec": str(spec_path), "out": str(target), "variants": len(item.adversarial_variants)}, indent=2))
    elif args.command == "trace2eval":
        traces = load_traces(args.traces) if args.traces else sample_failure_traces()
        results = compile_traces(traces)
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([result.model_dump(mode="json") for result in results], indent=2), encoding="utf-8")
        print(json.dumps({"out": str(target), "seeds": len(results)}, indent=2))
    elif args.command == "product-demo":
        bundle = BreakPointCompiler(seed=42).compile_dataset(items_per_category=3, variants_per_item=2)
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
        bundle = BreakPointCompiler(seed=42).compile_dataset(items_per_category=2, variants_per_item=2)
        _, _, suite, cases = bundle_to_product_layer(bundle)
        run = run_local_suite(suite, cases)
        report = build_ci_report(suite, cases, run, reviews=seed_human_reviews(cases), gate=default_regression_gate(suite))
        write_ci_report(report, args.out)
        print(json.dumps({"out": args.out, "passed": report["passed"]}, indent=2))
    elif args.command == "failuregym":
        bundle = BreakPointCompiler(seed=42).compile_dataset(items_per_category=3, variants_per_item=2)
        _, _, suite, cases = bundle_to_product_layer(bundle)
        manifest = build_failuregym_manifest(suite, cases, args.out_dir)
        print(json.dumps(manifest, indent=2))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("breakpoint_eval.api:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
