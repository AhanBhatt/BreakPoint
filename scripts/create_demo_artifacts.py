from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from breakpoint_eval.benchmark import build_failuregym_manifest
from breakpoint_eval.ci import build_ci_report, build_regression_packs, default_regression_gate, write_ci_report
from breakpoint_eval.compiler import BreakPointCompiler, write_jsonl, write_metrics
from breakpoint_eval.exporters import export_cases_jsonl, export_lm_eval_task, export_openai_evals
from breakpoint_eval.judges import CalibrationExample, LocalHeuristicJudge, calibrate_judges, default_judge_adapters
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
DASH_PUBLIC = ROOT / "dashboard" / "public"


def main() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    (DASH_PUBLIC / "data").mkdir(parents=True, exist_ok=True)
    (DASH_PUBLIC / "images").mkdir(parents=True, exist_ok=True)

    compiler = BreakPointCompiler(seed=42)
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

    (DASH_PUBLIC / "data" / "metrics.json").write_text(json.dumps(bundle.metrics, indent=2), encoding="utf-8")
    (DASH_PUBLIC / "data" / "sample_items.json").write_text(
        json.dumps([item.to_dataset_row() for item in bundle.items[:8]], indent=2),
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
    trace_results = compile_traces(sample_failure_traces())
    simulated = generate_simulated_environments()
    calibration_examples = [
        CalibrationExample(item=case.eval_item, human_passed=True, family=case.failure_family)
        for case in cases[:12]
    ]
    calibration = calibrate_judges(default_judge_adapters(include_external=False), calibration_examples)

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

    failure_inbox = [
        {
            "id": result.seed.trace_id,
            "family": result.seed.family,
            "summary": result.seed.summary,
            "confidence": result.seed.confidence,
            "signals": result.seed.signals,
            "review_status": result.review_status,
        }
        for result in trace_results
    ]
    mutation_graph = [
        {
            "id": case.id,
            "parent": case.mutation_lineage.parent_case_id,
            "seed": case.mutation_lineage.seed_id,
            "mutator": case.mutation_lineage.mutator,
            "family": case.failure_family,
            "title": case.eval_item.title,
        }
        for case in cases[:80]
    ]
    dashboard_data = {
        "product_summary.json": {
            "project": project.model_dump(mode="json"),
            "suite": suite.model_dump(mode="json"),
            "case_count": len(cases),
            "run_metrics": run.metrics,
            "native_metrics": native_metrics,
            "clusters": [cluster.model_dump(mode="json") for cluster in clusters],
        },
        "failure_inbox.json": failure_inbox,
        "spec_review.json": {
            "spec_yaml": spec.to_yaml(),
            "compiled_item": spec_item.to_dataset_row(),
            "trace_specs": [result.seed.spec.model_dump(mode="json") for result in trace_results],
        },
        "case_review.json": {
            "cases": [case.model_dump(mode="json") for case in cases[:24]],
            "reviews": [review.model_dump(mode="json") for review in reviews],
        },
        "mutation_graph.json": mutation_graph,
        "run_comparison.json": {
            "current": run.model_dump(mode="json"),
            "previous": {"pass_rate": 0.91, "mutation_survival_rate": 0.82, "evidence_freshness_score": 0.86},
            "delta": {
                "pass_rate": round(float(run.metrics["pass_rate"]) - 0.91, 3),
                "mutation_survival_rate": round(float(run.metrics["mutation_survival_rate"]) - 0.82, 3),
                "evidence_freshness_score": round(float(run.metrics["evidence_freshness_score"]) - 0.86, 3),
            },
        },
        "regression_gate.json": {"gate": gate.model_dump(mode="json"), "result": gate_result, "ci_report": ci_report},
        "judge_reliability.json": [report.model_dump(mode="json") for report in calibration],
        "simulators.json": [env.model_dump(mode="json") for env in simulated],
        "failuregym.json": failuregym_manifest,
    }
    for filename, payload in dashboard_data.items():
        (DASH_PUBLIC / "data" / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    draw_architecture(IMAGES / "architecture.png")
    draw_quality_gates(bundle.metrics, IMAGES / "quality_gates.png")
    draw_category_mix(bundle.metrics, IMAGES / "category_mix.png")
    draw_comparison(bundle.metrics, IMAGES / "comparison_results.png")

    for image in IMAGES.glob("*.png"):
        shutil.copy2(image, DASH_PUBLIC / "images" / image.name)


def draw_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    stages = [
        ("Failure\nSources", "traces, tickets,\nfeedback, incidents"),
        ("BreakPoint\nSpec", "contracts, traps,\noracles, rubrics"),
        ("Generators", "tasks, answers,\ncontext, tools"),
        ("Mutators", "paraphrase, reorder,\nconflict, inject"),
        ("Validators", "judges, calibration,\nhuman routing"),
        ("Product\nLayer", "suites, checks,\nruns, gates"),
        ("Exports", "JSONL, HF, CI,\nFailureGym, API"),
    ]
    colors = ["#1d4ed8", "#0f766e", "#7c3aed", "#be123c", "#b45309", "#0369a1", "#4d7c0f"]
    positions = [(0.08 + idx * 0.14, 0.54) for idx in range(len(stages))]
    box_w = 0.105
    box_h = 0.26
    for idx, ((title, subtitle), color, (x, y)) in enumerate(zip(stages, colors, positions, strict=True)):
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.4,
            edgecolor=color,
            facecolor="#ffffff",
        )
        ax.add_patch(box)
        ax.text(x, y + 0.035, title, ha="center", va="center", color="#0f172a", fontsize=12.5, weight="bold")
        ax.text(x, y - 0.062, subtitle, ha="center", va="center", color="#475569", fontsize=9.5)
        if idx < len(stages) - 1:
            next_x, next_y = positions[idx + 1]
            start = (x + box_w / 2 + 0.008, y)
            end = (next_x - box_w / 2 - 0.008, next_y)
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.0",
                    mutation_scale=16,
                    linewidth=1.5,
                    color="#334155",
                )
            )
    ax.text(0.5, 0.88, "BreakPoint Eval Data Compiler", ha="center", fontsize=23, weight="bold", color="#111827")
    ax.text(
        0.5,
        0.12,
        "Failure-to-eval loop: every accepted case keeps seed lineage, expected answer, hidden traps, rubric, variants, judge reports, review state, and CI gate metadata.",
        ha="center",
        fontsize=11.5,
        color="#334155",
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def draw_quality_gates(metrics: dict[str, object], path: Path) -> None:
    scores = metrics["quality_scores"]
    labels = ["Answerable", "Trap coverage", "Format", "Judge agreement", "Low ambiguity"]
    values = [
        scores["answerability_score"],
        scores["trap_coverage_score"],
        scores["format_score"],
        scores["judge_agreement"],
        1 - scores["ambiguity_score"],
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    bars = ax.bar(labels, values, color=["#2563eb", "#0f766e", "#7c3aed", "#be123c", "#b45309"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Validation Gate Scores", fontsize=16, weight="bold", pad=14)
    ax.grid(axis="y", color="#e2e8f0")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=10)
    fig.autofmt_xdate(rotation=15, ha="right")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def draw_category_mix(metrics: dict[str, object], path: Path) -> None:
    counts = metrics["category_counts"]
    labels = [label.replace("_", " ").title() for label in counts]
    values = list(counts.values())
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    colors = ["#2563eb", "#0f766e", "#7c3aed", "#be123c", "#b45309", "#475569", "#0891b2", "#4d7c0f", "#c2410c"]
    ax.barh(labels, values, color=colors[: len(labels)])
    ax.set_xlabel("Accepted base items")
    ax.set_title("Demo Dataset Coverage by Failure Category", fontsize=16, weight="bold", pad=14)
    ax.grid(axis="x", color="#e2e8f0")
    ax.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(values):
        ax.text(value + 0.05, idx, str(value), va="center", fontsize=10)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def draw_comparison(metrics: dict[str, object], path: Path) -> None:
    cases = metrics["total_eval_cases"]
    manual_hours = metrics["estimated_manual_prompt_hours"]
    compiler_hours = round(max(0.4, cases * 0.18 / 60), 1)
    labels = ["Manual authoring", "BreakPoint compiler"]
    values = [manual_hours, compiler_hours]
    fig, ax = plt.subplots(figsize=(9.8, 5.8), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    bars = ax.barh(labels, values, color=["#64748b", "#2563eb"])
    ax.set_xlabel("Estimated hours")
    ax.set_title("Authoring Effort Comparison", fontsize=16, weight="bold", pad=14)
    ax.grid(axis="x", color="#e2e8f0")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, f"{value:.1f}h", va="center", fontsize=11)
    ax.set_xlim(0, max(values) * 1.25)
    ax.invert_yaxis()
    annotation = textwrap.fill(
        f"Demo run produced {cases} total eval cases, including adversarial variants and validation reports.",
        width=56,
    )
    ax.text(
        0.98,
        0.16,
        annotation,
        transform=ax.transAxes,
        ha="right",
        fontsize=10.5,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#d8e1ec"},
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


if __name__ == "__main__":
    main()
