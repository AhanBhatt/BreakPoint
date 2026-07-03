# BreakPoint

BreakPoint is a Python SDK and CLI for turning real LLM failures into regression eval datasets. Feed it failure traces, support tickets, red-team transcripts, RAG logs, or tool-call traces; it compiles them into versioned eval cases with expected answers, hidden traps, rubrics, adversarial variants, validation reports, and exportable runner formats.

The core idea is failure-to-eval compilation: one observed failure becomes a small failure neighborhood that can be replayed against future prompts, models, RAG pipelines, and agent/tool policies.

![BreakPoint architecture](artifacts/images/architecture.png)

## What Exists

- Typed Python schemas for eval items, expected answers, hidden traps, rubrics, adversarial variants, model votes, validation reports, dataset bundles, suites, runs, reviews, clusters, and gates.
- Failure taxonomy for hallucination, instruction conflict, multi-hop reasoning, tool misuse, long-context retrieval, refusal boundaries, format violations, RAG contradiction, and prompt injection.
- BreakPointSpec YAML for declarative failure-family definitions.
- Trace2Eval ingestion for normalized failure traces, including RAG evidence, tool calls, original model output, expected behavior, provenance, and severity.
- Production ingestion adapters for OpenTelemetry/OpenInference spans, LangSmith traces, LiteLLM logs, retrieval logs, support tickets, thumbs-down feedback, red-team transcripts, and incident reports.
- Mutators for irrelevant context, reordered facts, renamed entities, paraphrased instructions, prompt attacks, and stale/contradictory evidence.
- Live judge adapters for OpenAI, Anthropic, Gemini, and an OpenAI-compatible local vLLM endpoint.
- Judge calibration, model-under-test comparison, risk-focused CI pack selection, and production regression-pack compilation.
- Export paths for JSONL, Hugging Face rows, DuckDB, OpenAI Evals-style YAML, lm-eval YAML/JSONL, CI reports, and FastAPI.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest
```

Compile demo eval data:

```powershell
.\.venv\Scripts\python -m breakpoint_eval.cli compile --items-per-category 4 --variants-per-item 2
```

Compile sourced public failure traces:

```powershell
.\.venv\Scripts\python -m breakpoint_eval.cli actual-data
```

Compile actual traces with live judges:

```powershell
.\.venv\Scripts\python -m breakpoint_eval.cli actual-data --external-judges --max-live-cost-usd 2
```

Generate live judge evaluation charts:

```powershell
.\.venv\Scripts\python -m breakpoint_eval.cli judge-report --results artifacts/actual/trace2eval_results.json --out-dir artifacts/reports
```

Build a production-style regression pack through the ingestion layer:

```powershell
.\.venv\Scripts\python -m breakpoint_eval.cli ingest-traces --input data/actual/failure_traces.json --source incident_report
.\.venv\Scripts\python -m breakpoint_eval.cli production-pack --input artifacts/ingestion/traces.json --source incident_report --max-records 10 --external-judges --changed-files "app/rag/retriever.py,agents/tools/order_tool.py" --risk high
```

## Python SDK

```python
from breakpoint_eval import BreakPoint, TraceBuilder

trace = TraceBuilder.rag_failure(
    id="prod-rag-001",
    question="Which policy is current?",
    bad_answer="The cached summary says the limit is 7 days.",
    expected_behavior="Use the newer official policy and answer 30 days.",
    retrieved_docs=[
        {"id": "cache", "content": "The limit is 7 days.", "effective_date": "2025-02-01", "reliability": 0.35},
        {"id": "policy", "content": "The limit is 30 days.", "effective_date": "2026-07-01", "reliability": 0.95},
    ],
)

bp = BreakPoint(include_external_judges=False, variants_per_item=3)
build = bp.build_pack([trace], output_dir="artifacts/sdk_pack")

print(build.total_cases, build.artifact_paths)
```

## Actual Failure Corpus

The current actual corpus lives at `data/actual/failure_traces.json`. It contains 12 sourced public incidents converted into normalized traces:

- Air Canada bereavement-fare chatbot hallucination
- NYC MyCity legal/compliance misadvice
- Mata v. Avianca fake legal citations
- DPD support chatbot manipulation
- Chevrolet dealership chatbot pricing manipulation
- CNET finance article errors
- NEDA Tessa harmful medical-adjacent advice
- Microsoft Tay social manipulation
- Pak'nSave Savey Mealbot unsafe recipe generation
- Google Bard JWST demo error
- Michael Cohen fake legal citations
- McDonald's AI drive-thru order failures

The latest live run produced:

| Metric | Value |
| --- | ---: |
| Sourced traces | 12 |
| Accepted base items | 12 |
| Adversarial variants | 36 |
| Total eval cases | 48 |
| Live judge set | OpenAI, Anthropic, Gemini, local vLLM |
| Estimated live judge cost | $0.1026 |
| Base validation passed | 12/12 |

## Live Judge Results

These charts are generated from `artifacts/actual/trace2eval_results.json`, not from the synthetic category-quota demo. The data includes per-case judge pass/fail decisions, confidence values, provider disagreement, estimated cost per judgment, source reliability spread, severity, and failure-family labels.

![Live judge reliability](artifacts/reports/judge_scoreboard.png)

![Judge confidence matrix](artifacts/reports/judge_confidence_matrix.png)

![Failure index by trace](artifacts/reports/failure_index_by_trace.png)

![Cost reliability Pareto](artifacts/reports/cost_reliability_pareto.png)

BreakPoint uses its own failure-oriented indices rather than broad competitor benchmark indexes:

- Failure Neighborhood Index
- Evidence Conflict Index
- Boundary Precision Index
- Instruction Attack Index
- Judge Consensus Index
- Source Tension Index

Latest index values:

| BreakPoint index | Score |
| --- | ---: |
| Failure Neighborhood Index | 82.9 |
| Evidence Conflict Index | 85.4 |
| Boundary Precision Index | 78.5 |
| Instruction Attack Index | 89.1 |
| Judge Consensus Index | 93.8 |
| Source Tension Index | 66.3 |

## Calibration and Model Runs

`calibrate-judges` produced `artifacts/calibration/judge_calibration_report.json`, `gold_labels.json`, and `calibrated_gate_policy.json` from the 12 live-judged actual traces. Current promoted judge/family pairs: 26. OpenAI, the local vLLM judge, and all local deterministic judges were promoted across all observed families; Anthropic was promoted for prompt-injection and RAG contradiction; Gemini was promoted across the observed families.

`model-runs` compares model-under-test profiles against the compiled cases. The current 48-case run produced:

| Profile | Pass rate | Avg score | Cost per reliable pass |
| --- | ---: | ---: | ---: |
| breakpoint-reference | 100.0% | 1.000 | $0.000100 |
| stale-rag-baseline | 41.7% | 0.504 | $0.000192 |
| over-refusal-baseline | 0.0% | 0.350 | $0.002400 |
| injection-prone-agent | 91.7% | 0.933 | $0.000131 |

![Model pass rates](artifacts/model_runs/model_pass_rates.png)

![Family model matrix](artifacts/model_runs/family_model_matrix.png)

![Cost latency tradeoff](artifacts/model_runs/cost_latency_tradeoff.png)

## Production Pack Validation

The ingestion/production-pack route was validated by normalizing 12 incident-report traces and compiling the first 10 into a live-judged regression pack:

| Artifact | Value |
| --- | ---: |
| Ingested traces | 12 |
| Production pack traces | 10 |
| Accepted base items | 10 |
| Total eval cases | 40 |
| Regression packs | 5 |
| Estimated live judge cost | $0.0855 |

## CLI Reference

```powershell
.\.venv\Scripts\python -m breakpoint_eval.cli categories
.\.venv\Scripts\python -m breakpoint_eval.cli compile-spec
.\.venv\Scripts\python -m breakpoint_eval.cli trace2eval --traces path\to\traces.json
.\.venv\Scripts\python -m breakpoint_eval.cli actual-data --external-judges
.\.venv\Scripts\python -m breakpoint_eval.cli judge-report
.\.venv\Scripts\python -m breakpoint_eval.cli ingest-traces --input path\to\logs.json --source support_ticket
.\.venv\Scripts\python -m breakpoint_eval.cli production-pack --input artifacts\ingestion\traces.json
.\.venv\Scripts\python -m breakpoint_eval.cli calibrate-judges
.\.venv\Scripts\python -m breakpoint_eval.cli model-runs
.\.venv\Scripts\python -m breakpoint_eval.cli ci-packs --changed-files "app/rag/retriever.py"
.\.venv\Scripts\python -m breakpoint_eval.cli failuregym
.\.venv\Scripts\python -m breakpoint_eval.cli vllm-judge-server --port 8001
```

## Outputs

`actual-data` writes:

- `artifacts/actual/source_traces.json`
- `artifacts/actual/trace2eval_results.json`
- `artifacts/actual/metrics.json`
- `artifacts/actual/manifest.json`
- `artifacts/actual/product.json`
- `artifacts/actual/cases.jsonl`
- `artifacts/actual/openai_evals.yaml`
- `artifacts/actual/lm_eval_task.yaml`
- `artifacts/actual/lm_eval_task.jsonl`
- `artifacts/actual/ci_report.json`
- `artifacts/actual/ci_report.md`

`judge-report` writes:

- `artifacts/reports/live_judge_report.json`
- `artifacts/reports/judge_scoreboard.png`
- `artifacts/reports/judge_confidence_matrix.png`
- `artifacts/reports/failure_index_by_trace.png`
- `artifacts/reports/cost_reliability_pareto.png`

`calibrate-judges`, `model-runs`, `ingest-traces`, `production-pack`, and `ci-packs` write:

- `artifacts/calibration/judge_calibration_report.json`
- `artifacts/calibration/calibrated_gate_policy.json`
- `artifacts/model_runs/model_run_comparison.json`
- `artifacts/model_runs/model_pass_rates.png`
- `artifacts/model_runs/family_model_matrix.png`
- `artifacts/model_runs/cost_latency_tradeoff.png`
- `artifacts/ingestion/traces.json`
- `artifacts/ingestion/report.json`
- `artifacts/production_pack/manifest.json`
- `artifacts/production_pack/cases.jsonl`
- `artifacts/production_pack/ci_report.json`
- `artifacts/ci/packs.json`

## Integration Shape

BreakPoint is meant to sit inside existing Python evaluation and release workflows:

1. Normalize real failures into `RawFailureTrace` records.
2. Compile traces into BreakPointSpec drafts and eval cases.
3. Generate adversarial variants around each failure.
4. Validate with local and live judges.
5. Export cases to your eval runner.
6. Gate regressions in CI using recent failure packs.

The repeatable local path uses deterministic judges. Live judges are opt-in through `--external-judges`, `external_judges=True`, or `BREAKPOINT_EXTERNAL_JUDGES=1`.

## Repository Layout

```text
breakpoint_eval/          Python SDK, compiler, validators, reports, CLI
data/actual/              Sourced public failure traces
examples/                 SDK examples for RAG, tool, support, and red-team traces
scripts/                  Artifact and report generation helpers
tests/                    Unit and API tests
artifacts/actual/         Generated actual-data outputs
artifacts/reports/        Generated live judge charts and report JSON
artifacts/model_runs/     Generated model-under-test comparisons and charts
```
