from pathlib import Path

from breakpoint_eval.benchmark import build_failuregym_manifest
from breakpoint_eval.ci import build_ci_report, default_regression_gate
from breakpoint_eval.compiler import BreakPointCompiler
from breakpoint_eval.exporters import export_lm_eval_task, export_openai_evals
from breakpoint_eval.judges import CalibrationExample, LocalHeuristicJudge, calibrate_judges
from breakpoint_eval.metrics import compiler_native_metrics
from breakpoint_eval.platform import bundle_to_product_layer, run_local_suite, seed_human_reviews
from breakpoint_eval.simulators import generate_simulated_environments
from breakpoint_eval.specs import compile_spec, example_rag_freshness_spec, parse_spec
from breakpoint_eval.traces import compile_traces, sample_failure_traces, trace_to_spec


def test_product_layer_and_ci_exports(tmp_path: Path) -> None:
    bundle = BreakPointCompiler(seed=21).compile_dataset(categories=["rag_contradiction"], items_per_category=2, variants_per_item=2)
    project, version, suite, cases = bundle_to_product_layer(bundle)
    run = run_local_suite(suite, cases)
    reviews = seed_human_reviews(cases)

    assert project.id
    assert version.item_ids
    assert suite.checks
    assert len(cases) == 6
    assert run.metrics["pass_rate"] == 1.0

    metrics = compiler_native_metrics(cases, run, reviews=reviews)
    assert metrics["mutation_survival_rate"] == 1.0
    assert metrics["failure_neighborhood_coverage"]["seed_count"] == 2

    report = build_ci_report(suite, cases, run, reviews=reviews, gate=default_regression_gate(suite))
    assert "BreakPoint Regression Report" in report["summary_markdown"]

    export_openai_evals(suite, cases, tmp_path / "openai.yaml")
    export_lm_eval_task(suite, cases, tmp_path / "lm_eval.yaml")
    assert (tmp_path / "openai.yaml").exists()
    assert (tmp_path / "lm_eval.jsonl").exists()


def test_breakpoint_spec_and_trace2eval() -> None:
    spec = example_rag_freshness_spec()
    parsed = parse_spec(spec.to_yaml())
    item = compile_spec(parsed)
    assert item.category == "rag_contradiction"
    assert len(item.adversarial_variants) == 5

    trace = sample_failure_traces()[0]
    seed = trace_to_spec(trace)
    assert seed.family == "rag_contradiction"
    results = compile_traces(sample_failure_traces())
    assert len(results) == 3
    assert all(result.eval_case.original_failure_seed for result in results)


def test_judge_calibration_simulators_and_failuregym(tmp_path: Path) -> None:
    bundle = BreakPointCompiler(seed=23).compile_dataset(categories=["tool_misuse"], items_per_category=2, variants_per_item=1)
    _, _, suite, cases = bundle_to_product_layer(bundle)
    examples = [
        CalibrationExample(item=case.eval_item, human_passed=True, family=case.failure_family)
        for case in cases[:2]
    ]
    reports = calibrate_judges([LocalHeuristicJudge()], examples)
    assert reports[0].accuracy == 1.0

    environments = generate_simulated_environments()
    assert len(environments) >= 15
    assert any(env.family == "rag_contradiction" for env in environments)

    manifest = build_failuregym_manifest(suite, cases, tmp_path / "failuregym")
    assert manifest["benchmark"] == "BreakPoint-FailureGym"
    assert (tmp_path / "failuregym" / "manifest.json").exists()
