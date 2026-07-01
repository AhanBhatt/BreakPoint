from __future__ import annotations

import argparse
import json
from pathlib import Path

from breakpoint_eval.categories import FAILURE_CATEGORIES
from breakpoint_eval.compiler import BreakPointCompiler, to_huggingface_dataset, write_jsonl, write_metrics
from breakpoint_eval.storage import DuckDBStore


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
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("breakpoint_eval.api:app", host=args.host, port=args.port, reload=True)


if __name__ == "__main__":
    main()
