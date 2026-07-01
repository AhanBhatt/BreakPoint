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

from breakpoint_eval.compiler import BreakPointCompiler, write_jsonl, write_metrics
from breakpoint_eval.storage import DuckDBStore


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
    store.close()

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

    draw_architecture(IMAGES / "architecture.png")
    draw_quality_gates(bundle.metrics, IMAGES / "quality_gates.png")
    draw_category_mix(bundle.metrics, IMAGES / "category_mix.png")
    draw_comparison(bundle.metrics, IMAGES / "comparison_results.png")

    for image in IMAGES.glob("*.png"):
        shutil.copy2(image, DASH_PUBLIC / "images" / image.name)


def draw_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")
    stages = [
        ("Failure\nTaxonomy", "real incidents,\npolicy boundaries"),
        ("Generators", "tasks, answers,\ntraps, rubrics"),
        ("Mutators", "paraphrase, reorder,\nconflict, inject"),
        ("Validators", "multi-judge\nquality gates"),
        ("Dataset\nExports", "JSONL, HF,\nDuckDB, API"),
    ]
    colors = ["#1d4ed8", "#0f766e", "#7c3aed", "#be123c", "#b45309"]
    x_positions = [0.08, 0.29, 0.50, 0.71, 0.90]
    y = 0.56
    for idx, ((title, subtitle), color, x) in enumerate(zip(stages, colors, x_positions, strict=True)):
        box = FancyBboxPatch(
            (x - 0.075, y - 0.16),
            0.15,
            0.28,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.4,
            edgecolor=color,
            facecolor="#ffffff",
        )
        ax.add_patch(box)
        ax.text(x, y + 0.035, title, ha="center", va="center", color="#0f172a", fontsize=14, weight="bold")
        ax.text(x, y - 0.075, subtitle, ha="center", va="center", color="#475569", fontsize=10)
        if idx < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.083, y),
                    (x_positions[idx + 1] - 0.083, y),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=1.5,
                    color="#334155",
                )
            )
    ax.text(0.5, 0.88, "BreakPoint Eval Data Compiler", ha="center", fontsize=23, weight="bold", color="#111827")
    ax.text(
        0.5,
        0.18,
        "Fuzzing-style dataset generation: every accepted item has an expected answer, hidden trap, rubric, variants, and validation report.",
        ha="center",
        fontsize=12,
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
