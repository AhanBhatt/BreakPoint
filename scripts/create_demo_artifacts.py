from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from breakpoint_eval.benchmark import build_failuregym_manifest
from breakpoint_eval.ci import build_ci_report, build_regression_packs, default_regression_gate, write_ci_report
from breakpoint_eval.compiler import BreakPointCompiler, write_jsonl, write_metrics
from breakpoint_eval.env import env_flag, load_env
from breakpoint_eval.exporters import export_cases_jsonl, export_lm_eval_task, export_openai_evals
from breakpoint_eval.judges import CalibrationExample, calibrate_judges, default_judge_adapters, available_validation_suite
from breakpoint_eval.metrics import compiler_native_metrics
from breakpoint_eval.platform import (
    build_failure_clusters,
    bundle_to_product_layer,
    evaluate_gate,
    run_local_suite,
    seed_human_reviews,
)
from breakpoint_eval.simulators import generate_simulated_environments
from breakpoint_eval.specs import compile_spec, example_rag_freshness_spec, write_spec
from breakpoint_eval.storage import DuckDBStore
from breakpoint_eval.traces import compile_traces, sample_failure_traces


ARTIFACTS = ROOT / "artifacts"
DEMO = ARTIFACTS / "demo"
IMAGES = ARTIFACTS / "images"


def main() -> None:
    load_env()
    DEMO.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    include_external_judges = env_flag("BREAKPOINT_EXTERNAL_JUDGES", False)
    validator = available_validation_suite(include_external=include_external_judges)
    compiler = BreakPointCompiler(seed=42, validator=validator)
    bundle = compiler.compile_dataset(items_per_category=8, variants_per_item=4)
    write_jsonl(bundle.items, DEMO / "breakpoint_eval.jsonl")
    write_metrics(bundle.metrics, DEMO / "metrics.json")
    (DEMO / "preview.json").write_text(
        json.dumps([item.to_dataset_row() for item in bundle.items[:5]], indent=2),
        encoding="utf-8",
    )
    store = DuckDBStore(DEMO / "breakpoint.duckdb")
    store.save_bundle(bundle)

    api_output = {
        "GET /health": {"status": "ok", "service": "BreakPoint", "mode": "deterministic-local"},
        "POST /compile": {
            "dataset_id": bundle.dataset_id,
            "accepted_items": bundle.metrics["accepted_items"],
            "total_eval_cases": bundle.metrics["total_eval_cases"],
            "preview_category": bundle.items[0].category,
        },
    }
    (DEMO / "api_output.json").write_text(json.dumps(api_output, indent=2), encoding="utf-8")
    (DEMO / "run_output.txt").write_text(
        "\n".join(
            [
                f"dataset_id={bundle.dataset_id}",
                f"accepted_items={bundle.metrics['accepted_items']}",
                f"adversarial_variants={bundle.metrics['adversarial_variants']}",
                f"total_eval_cases={bundle.metrics['total_eval_cases']}",
                f"rejected_candidates={bundle.metrics['rejected_candidates']}",
                f"acceptance_rate={bundle.metrics['acceptance_rate']}",
            ]
        ),
        encoding="utf-8",
    )

    project, version, suite, cases = bundle_to_product_layer(bundle)
    run = run_local_suite(suite, cases)
    reviews = seed_human_reviews(cases, limit=12)
    clusters = build_failure_clusters(project, cases)
    gate = default_regression_gate(suite)
    native_metrics = compiler_native_metrics(cases, run, reviews=reviews)
    run.metrics.update(native_metrics)
    gate_result = evaluate_gate(gate, run)
    ci_report = build_ci_report(suite, cases, run, reviews=reviews, gate=gate)
    packs = build_regression_packs(suite, cases)
    spec = example_rag_freshness_spec()
    spec_item = compile_spec(spec)
    trace_results = compile_traces(sample_failure_traces(), validator=validator)
    simulated = generate_simulated_environments()
    calibration_examples = [
        CalibrationExample(item=case.eval_item, human_passed=True, family=case.failure_family)
        for case in cases[:12]
    ]
    calibration = calibrate_judges(default_judge_adapters(include_external=include_external_judges), calibration_examples)

    store.save_product_layer(
        project,
        version,
        suite,
        cases,
        run=run,
        reviews=reviews,
        clusters=clusters,
        gates=[gate],
    )
    store.close()

    product_payload = {
        "project": project.model_dump(mode="json"),
        "dataset_version": version.model_dump(mode="json"),
        "suite": suite.model_dump(mode="json"),
        "cases": [case.model_dump(mode="json") for case in cases[:36]],
        "run": run.model_dump(mode="json"),
        "reviews": [review.model_dump(mode="json") for review in reviews],
        "clusters": [cluster.model_dump(mode="json") for cluster in clusters],
        "gate": gate.model_dump(mode="json"),
        "gate_result": gate_result,
        "native_metrics": native_metrics,
        "packs": [pack.model_dump(mode="json") for pack in packs],
    }
    (DEMO / "product.json").write_text(json.dumps(product_payload, indent=2), encoding="utf-8")
    write_ci_report(ci_report, DEMO / "ci_report.json")
    export_cases_jsonl(cases, DEMO / "cases.jsonl")
    export_openai_evals(suite, cases, DEMO / "openai_evals.yaml")
    export_lm_eval_task(suite, cases, DEMO / "lm_eval_task.yaml")
    failuregym_manifest = build_failuregym_manifest(suite, cases, ARTIFACTS / "failuregym")

    specs_dir = ARTIFACTS / "specs"
    write_spec(spec, specs_dir / "rag_freshness_contradiction.yaml")
    (specs_dir / "compiled_spec_item.json").write_text(json.dumps(spec_item.to_dataset_row(), indent=2), encoding="utf-8")
    (ARTIFACTS / "trace2eval").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "trace2eval" / "results.json").write_text(
        json.dumps([result.model_dump(mode="json") for result in trace_results], indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS / "trace2eval" / "sample_traces.json").write_text(
        json.dumps([trace.model_dump(mode="json") for trace in sample_failure_traces()], indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS / "simulators.json").write_text(
        json.dumps([env.model_dump(mode="json") for env in simulated], indent=2),
        encoding="utf-8",
    )
    (ARTIFACTS / "judge_calibration.json").write_text(
        json.dumps([report.model_dump(mode="json") for report in calibration], indent=2),
        encoding="utf-8",
    )

    draw_architecture(IMAGES / "architecture.png")


def draw_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("01", "Failure\nTraces", "RAG logs\nsupport tickets\ntool calls"),
        ("02", "BreakPoint\nSpec", "contract\noracle\nhidden traps"),
        ("03", "Case\nCompiler", "task\ncontext\nexpected answer"),
        ("04", "Mutation\nEngine", "paraphrase\nreorder\ninject conflict"),
        ("05", "Judge\nValidation", "live judges\nlocal checks\ncalibration"),
        ("06", "Regression\nPacks", "JSONL / HF\nOpenAI evals\nCI gates"),
    ]
    colors = ["#2563eb", "#0f766e", "#7c3aed", "#be123c", "#b45309", "#0369a1"]
    centers = [0.08, 0.248, 0.416, 0.584, 0.752, 0.92]
    y = 0.52
    box_w = 0.115
    box_h = 0.28
    for idx, ((number, title, subtitle), color, x) in enumerate(zip(stages, colors, centers, strict=True)):
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            linewidth=1.6,
            edgecolor=color,
            facecolor="#ffffff",
        )
        ax.add_patch(box)
        ax.text(x - box_w / 2 + 0.012, y + box_h / 2 - 0.033, number, ha="left", va="center", color=color, fontsize=10.2, weight="bold")
        ax.text(x, y + 0.045, title, ha="center", va="center", color="#0f172a", fontsize=13.2, weight="bold", linespacing=1.08)
        ax.text(x, y - 0.068, subtitle, ha="center", va="center", color="#475569", fontsize=9.7, linespacing=1.2)
        if idx < len(stages) - 1:
            next_x = centers[idx + 1]
            start = (x + box_w / 2 + 0.012, y)
            end = (next_x - box_w / 2 - 0.012, y)
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.0",
                    mutation_scale=18,
                    linewidth=1.8,
                    color="#334155",
                    shrinkA=0,
                    shrinkB=0,
                )
            )

    ax.text(0.5, 0.905, "BreakPoint Eval Data Compiler", ha="center", fontsize=29, weight="bold", color="#111827")
    ax.text(
        0.5,
        0.848,
        "Turn one real LLM failure into a versioned, adversarial regression suite.",
        ha="center",
        fontsize=14,
        color="#475569",
    )

    chips = [
        ("seed lineage", "#dbeafe", "#1d4ed8"),
        ("expected answer", "#dcfce7", "#166534"),
        ("hidden traps", "#ede9fe", "#6d28d9"),
        ("rubric", "#ffe4e6", "#be123c"),
        ("judge report", "#ffedd5", "#c2410c"),
        ("review state", "#e0f2fe", "#0369a1"),
        ("CI metadata", "#ecfccb", "#4d7c0f"),
    ]
    chip_y = 0.225
    chip_w = 0.115
    chip_h = 0.052
    chip_gap = 0.012
    total_w = len(chips) * chip_w + (len(chips) - 1) * chip_gap
    start_x = 0.5 - total_w / 2 + chip_w / 2
    for idx, (label, fill, stroke) in enumerate(chips):
        chip_x = start_x + idx * (chip_w + chip_gap)
        chip = FancyBboxPatch(
            (chip_x - chip_w / 2, chip_y - chip_h / 2),
            chip_w,
            chip_h,
            boxstyle="round,pad=0.01,rounding_size=0.018",
            linewidth=1.0,
            edgecolor=stroke,
            facecolor=fill,
        )
        ax.add_patch(chip)
        ax.text(chip_x, chip_y, label, ha="center", va="center", fontsize=9.8, color="#0f172a", weight="bold")

    ax.text(
        0.5,
        0.145,
        "Every accepted case carries the data needed to reproduce, mutate, validate, export, and gate the failure neighborhood.",
        ha="center",
        fontsize=12.5,
        color="#334155",
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.32, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
