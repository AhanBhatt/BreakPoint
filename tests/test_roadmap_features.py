import os
import json
from pathlib import Path

from breakpoint_eval.actual_data import build_actual_dataset_artifacts, estimate_live_judge_cost
from breakpoint_eval.benchmark import build_failuregym_manifest
from breakpoint_eval.calibration import calibrate_from_trace2eval_results
from breakpoint_eval.ci import build_ci_report, default_regression_gate, select_regression_packs
from breakpoint_eval.compiler import BreakPointCompiler
from breakpoint_eval.env import load_env
from breakpoint_eval.exporters import export_lm_eval_task, export_openai_evals
from breakpoint_eval.ingestion import ingest_payload
from breakpoint_eval.judges import CalibrationExample, LocalHeuristicJudge, calibrate_judges
from breakpoint_eval.metrics import compiler_native_metrics
from breakpoint_eval.model_runs import run_model_comparison_from_product
from breakpoint_eval.models import EvalCase, EvalSuite
from breakpoint_eval.platform import bundle_to_product_layer, run_local_suite, seed_human_reviews
from breakpoint_eval.production import build_production_regression_pack
from breakpoint_eval.sdk import BreakPoint, TraceBuilder
from breakpoint_eval.simulators import generate_simulated_environments
from breakpoint_eval.specs import compile_spec, example_rag_freshness_spec, parse_spec
from breakpoint_eval.traces import compile_traces, sample_failure_traces, trace_to_spec
from breakpoint_eval.vllm_server import app as vllm_app
from fastapi.testclient import TestClient


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


def test_env_loader_and_vllm_compatible_server(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_JUDGE_MODEL=test-openai\nBREAKPOINT_EXTERNAL_JUDGES=1\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("BREAKPOINT_EXTERNAL_JUDGES", raising=False)
    load_env(env_file)
    assert os.environ["OPENAI_JUDGE_MODEL"] == "test-openai"
    assert os.environ["BREAKPOINT_EXTERNAL_JUDGES"] == "1"

    client = TestClient(vllm_app)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "breakpoint-local-judge",
            "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Task: Determine whether the generated eval item is clear and answerable.\n"
                            "Context: This context is intentionally long enough to judge. It includes explicit evidence, "
                            "a clear expected behavior, and enough surrounding text for the local server to evaluate quality.\n"
                            "Expected: It should pass because the task, context, expected answer, and trap are present.\n"
                            "Hidden traps: [{'kind':'trap'}]"
                        ),
                    }
                ],
            },
    )
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert '"passed":true' in content


def test_actual_failure_corpus_build(tmp_path: Path) -> None:
    result = build_actual_dataset_artifacts(
        output_dir=tmp_path / "actual",
        max_records=2,
        variants_per_item=1,
        include_external_judges=False,
    )

    assert result.trace_count == 2
    assert result.base_items == 2
    assert result.total_cases == 4
    assert result.external_judges is False
    assert result.source_urls
    assert (tmp_path / "actual" / "manifest.json").exists()
    assert (tmp_path / "actual" / "cases.jsonl").exists()
    assert estimate_live_judge_cost(2, include_external=True) > 0


def test_production_ingestion_calibration_model_runs_and_ci_packs(tmp_path: Path) -> None:
    payload = {
        "tickets": [
            {
                "id": "ticket-4721",
                "subject": "Wrong refund policy sent to customer@example.com",
                "body": "The assistant used the archived plan summary instead of the July policy memo.",
                "ai_response": "Refunds are capped at 7 days because the archived plan says so.",
                "correct_resolution": "Use the July 2026 policy memo and state that refunds are capped at 30 days for this plan.",
                "failure_summary": "Support answer cited stale retrieved policy instead of the current memo.",
                "severity": "high",
                "retrieved_docs": [
                    {
                        "id": "archive-plan",
                        "source_type": "cached_summary",
                        "title": "Archived plan summary",
                        "content": "Refunds are capped at 7 days for this plan.",
                        "effective_date": "2025-02-01",
                        "reliability": 0.36,
                        "rank": 0,
                    },
                    {
                        "id": "july-policy",
                        "source_type": "official_policy",
                        "title": "July policy memo",
                        "content": "Refunds are capped at 30 days for this plan.",
                        "effective_date": "2026-07-01",
                        "reliability": 0.94,
                        "rank": 1,
                    },
                ],
            }
        ]
    }
    report = ingest_payload(payload, source="support_ticket")
    assert report.trace_count == 1
    assert "email@example.com" in report.traces[0].user_task

    source_path = tmp_path / "tickets.json"
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    pack = build_production_regression_pack(
        source_path=source_path,
        source="support_ticket",
        output_dir=tmp_path / "pack",
        variants_per_item=1,
        include_external_judges=False,
        changed_files=["app/rag/retriever.py", "agents/refund_tool.py"],
        risk="high",
    )
    assert pack.base_items == 1
    assert pack.total_cases == 2
    assert pack.regression_pack_count >= 5
    assert (tmp_path / "pack" / "product.json").exists()

    calibration = calibrate_from_trace2eval_results(
        tmp_path / "pack" / "trace2eval_results.json",
        out_dir=tmp_path / "calibration",
        min_accuracy=0.5,
    )
    assert calibration.examples == 1
    assert calibration.promoted_pairs
    assert (tmp_path / "calibration" / "calibrated_gate_policy.json").exists()

    comparison = run_model_comparison_from_product(tmp_path / "pack" / "product.json", out_dir=tmp_path / "model_runs")
    assert comparison.case_count == 2
    assert "breakpoint-reference" in comparison.summary["models"]
    assert (tmp_path / "model_runs" / "model_run_summary.md").exists()
    assert (tmp_path / "model_runs" / "model_pass_rates.png").exists()
    assert (tmp_path / "model_runs" / "family_model_matrix.png").exists()

    product = json.loads((tmp_path / "pack" / "product.json").read_text(encoding="utf-8"))
    suite = EvalSuite.model_validate(product["suite"])
    cases = [EvalCase.model_validate(case) for case in product["cases"]]
    packs = select_regression_packs(suite, cases, changed_files=["app/rag/retriever.py"], risk="high")
    assert any(pack.name == "changed-files focus pack" for pack in packs)


def test_sdk_trace_builders_and_pack(tmp_path: Path) -> None:
    trace = TraceBuilder.rag_failure(
        id="sdk-rag-001",
        question="Which refund window applies to the Aster subscription?",
        bad_answer="The archived summary says the refund window is 7 days.",
        expected_behavior="Use the official July memo and answer that the refund window is 30 days.",
        retrieved_docs=[
            {
                "id": "archived-summary",
                "source_type": "cached_summary",
                "content": "Aster subscription refund window is 7 days.",
                "effective_date": "2025-11-01",
                "reliability": 0.35,
                "rank": 0,
            },
            {
                "id": "official-july-memo",
                "source_type": "official_policy",
                "content": "Aster subscription refund window is 30 days.",
                "effective_date": "2026-07-01",
                "reliability": 0.96,
                "rank": 1,
            },
        ],
    )
    client = BreakPoint(variants_per_item=1)
    cases = client.compile_traces([trace])
    assert len(cases) == 1
    assert cases[0].eval_item.metadata["trace_id"] == "sdk-rag-001"

    build = client.build_pack([trace], output_dir=tmp_path / "sdk_pack")
    assert build.total_cases == 2
    assert (tmp_path / "sdk_pack" / "cases.jsonl").exists()
