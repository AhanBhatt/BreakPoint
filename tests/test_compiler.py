from pathlib import Path

from breakpoint_eval.compiler import BreakPointCompiler, write_jsonl
from breakpoint_eval.storage import DuckDBStore, HashVectorIndex


def test_compiler_generates_valid_items_with_variants(tmp_path: Path) -> None:
    compiler = BreakPointCompiler(seed=11)
    bundle = compiler.compile_dataset(
        categories=["hallucination", "format_violation"],
        items_per_category=3,
        variants_per_item=2,
    )

    assert bundle.metrics["accepted_items"] == 6
    assert bundle.metrics["adversarial_variants"] == 12
    assert bundle.metrics["rejected_candidates"] >= 1
    assert all(item.hidden_traps for item in bundle.items)
    assert all(report.passed for report in bundle.validation_reports)

    output = tmp_path / "eval.jsonl"
    write_jsonl(bundle.items, output)
    assert output.read_text(encoding="utf-8").count("\n") == 6


def test_storage_and_hash_retrieval(tmp_path: Path) -> None:
    compiler = BreakPointCompiler(seed=13)
    bundle = compiler.compile_dataset(categories=["prompt_injection"], items_per_category=2, variants_per_item=1)

    store = DuckDBStore(tmp_path / "breakpoint.duckdb")
    store.save_bundle(bundle)
    assert store.latest_metrics()["accepted_items"] == 2
    store.close()

    index = HashVectorIndex()
    for item in bundle.items:
        index.add(item)
    hits = index.search("ignore hidden system prompt retrieved footer", k=1)
    assert hits
    assert hits[0][1].category == "prompt_injection"
