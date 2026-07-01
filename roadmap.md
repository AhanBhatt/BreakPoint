# Roadmap

Positioning:

- BreakPoint is not another broad model leaderboard.
- Vals tells you how models perform on benchmarks. BreakPoint turns your
  failures into the benchmarks.
- The product surface should feel familiar: projects, suites, cases, checks,
  runs, dashboards, human review, CI, imports, and exports.
- The core is different: BreakPoint is a failure-to-eval compiler that turns one
  real failure trace into a versioned, validated, adversarial regression suite
  around the whole failure neighborhood.

Implemented:

- BreakPoint Python package with typed Pydantic schemas for eval items, expected
  answers, hidden traps, rubrics, adversarial variants, model votes, validation
  reports, and dataset bundles.
- Failure taxonomy covering hallucination, instruction conflict, multi-hop
  reasoning, tool misuse, long-context retrieval, refusal boundaries, format
  violations, RAG contradiction, and prompt injection.
- Deterministic category generators that create tasks, context, expected
  answers, hidden traps, grading rubrics, and metadata.
- Adversarial mutators for irrelevant context, reordered facts, paraphrased
  instructions, renamed entities, and conflicting evidence.
- Local multi-judge validation suite that scores ambiguity, answerability, trap
  coverage, format compliance, and judge agreement.
- Filtering loop that rejects ambiguous or low-quality candidates and keeps
  generating until each requested category quota is filled.
- Product-layer objects for `Project`, `DatasetVersion`, `EvalSuite`,
  `EvalCase`, `Check`, `Run`, `ModelOutput`, `Judgment`, `HumanReview`,
  `FailureCluster`, and `RegressionGate`.
- Product-layer conversion that preserves compiler metadata on every case:
  original failure seed, prompt, context, files, tools, expected answer, hidden
  traps, rubric, oracle, mutation lineage, validation report, and failure
  family.
- Local suite runner, check scorer, run summarizer, failure-cluster builder,
  seeded review queue, and regression-gate evaluator.
- BreakPointSpec YAML DSL for declarative, versioned, reviewable failure-family
  definitions.
- BreakPointSpec compiler support for category, risk, answer policy, evidence
  rules, source priority, freshness rules, traps, mutators, oracles, required
  claims, forbidden claims, and rubric criteria.
- Trace2Eval ingestion path for RAG logs, tool-call traces, user feedback,
  support tickets, red-team transcripts, and incident-report-style records.
- Trace2Eval pipeline for redacting secrets/PII, extracting task/context/model
  output/tool calls/retrieved documents, classifying failure family, drafting a
  BreakPointSpec, compiling a base case, mutating around the failure, validating
  the result, and creating a product-layer eval case.
- Judge adapters for local heuristic judging, OpenAI, Anthropic, Gemini, and
  vLLM-compatible endpoints.
- Judge calibration reports for human agreement, inter-judge agreement,
  pass-rate deltas, disagreement counts, unreliable family detection, and
  uncertainty routing to human review.
- Tool simulator for stale results, malformed responses, partial results,
  permission denial, contradictions, timeouts, wrong schemas, and cached/live
  conflicts.
- Retrieval simulator for stale-first ranking, unreliable-source ranking,
  prompt injection in retrieved documents, split evidence, wrong citations, and
  missing-evidence abstention.
- Agent simulator for wrong tool selection, wrong arguments, unnecessary tool
  calls, no retry after tool error, and final-answer-correct/trace-invalid cases.
- Compiler-native metrics for mutation survival, failure-neighborhood coverage,
  oracle confidence, regression fragility, trace causality, evidence freshness,
  injection resistance, and abstention precision/recall.
- JSONL export for generated eval datasets.
- Hugging Face row export helper.
- OpenAI Evals-style YAML export.
- lm-eval task YAML plus JSONL export.
- Reproducibility manifest export.
- DuckDB persistence for eval items, validation reports, dataset metrics,
  projects, dataset versions, suites, cases, runs, reviews, clusters, and gates.
- Local hash vector index used as a lightweight retrieval fallback when FAISS or
  Qdrant are not configured.
- LangGraph-compatible orchestration surface with a deterministic built-in graph
  fallback.
- FastAPI service with `/health`, `/categories`, `/compile`, `/metrics`,
  `/specs/compile`, `/traces/compile`, and `/product/demo`.
- CLI commands for `compile`, `categories`, `serve`, `compile-spec`,
  `trace2eval`, `product-demo`, `ci-report`, and `failuregym`.
- Next.js dashboard upgraded from inspection-only views into a v0.2 review and
  triage workflow: Failure Inbox, Spec Review, Case Review, Mutation Graph, Run
  Comparison, Regression Gate, Judge Reliability, Simulators, and FailureGym.
- GitHub Actions workflow for Python tests, artifact generation, dashboard
  install/audit/build, and LaTeX PDF generation.
- CI regression-pack reporting for smoke, recent failure, mutation, and safety
  packs, including pass rate, mutation survival, evidence freshness,
  prompt-injection resistance, human-review queue size, and gate failures.
- BreakPoint-FailureGym v1 package organized around failure families rather than
  industries, with tracks for RAG Freshness and Citation, Tool-Use Faults,
  Prompt-Injection in Retrieved Context, Format and Structured Output
  Robustness, and Refusal Boundary.
- Generated demo artifacts under `artifacts/demo`, including JSONL, product
  JSON, CI report JSON/Markdown, OpenAI Evals YAML, lm-eval YAML/JSONL, preview
  JSON, metrics JSON, API output JSON, run output text, and local DuckDB store.
- Generated BreakPointSpec artifacts under `artifacts/specs`.
- Generated Trace2Eval artifacts under `artifacts/trace2eval`.
- Generated FailureGym package under `artifacts/failuregym`.
- Generated simulator and judge-calibration reports under `artifacts`.
- Generated dashboard data under `dashboard/public/data`.
- Generated diagrams and charts under `artifacts/images`, including
  architecture, validation gates, category coverage, authoring comparison, and
  dashboard screenshot.
- LaTeX whitepaper source under `docs/latex`.
- Generated system design PDF at `output/pdf/breakpoint_system_design.pdf`.
- `.gitignore` covering virtualenvs, Node/Next build outputs, local stores,
  temporary PDF renders, LaTeX build products, and generated PDFs.

Validated:

- Demo compiler run generated 72 accepted base items, 288 adversarial variants,
  360 total eval cases, and 18 rejected candidates.
- Demo acceptance rate was 0.8 with balanced coverage across all nine failure
  categories.
- Product-layer demo generated project, dataset version, suite, cases, run,
  judgments, reviews, failure clusters, and regression gate artifacts.
- BreakPointSpec example compiled into a concrete eval item with adversarial
  variants and validation metadata.
- Trace2Eval sample traces compiled into failure seeds, generated specs, eval
  items, validation reports, and product-layer cases.
- Judge calibration ran against local gold examples and produced agreement,
  disagreement, confidence, and human-review routing reports.
- Simulator report generated controlled tool, retrieval, and agent environments.
- BreakPoint-FailureGym manifest generated all five intended tracks with
  reproducibility manifests and export files.
- CI report generated smoke, recent-failure, mutation, and safety packs plus an
  explanatory regression-gate Markdown report.
- Python tests passed: 6 passed, 1 FastAPI/Starlette test-client deprecation
  warning.
- FastAPI compile endpoint was exercised through tests.
- FastAPI product, spec, and trace surfaces are covered by local smoke paths and
  generated demo artifacts.
- DuckDB save/read path was exercised through tests.
- Hash retrieval fallback was exercised through tests.
- Next.js dashboard production build completed successfully.
- `npm audit --audit-level=moderate` reported 0 vulnerabilities after the
  patched dependency override.
- Dashboard was served locally and returned HTTP 200 at
  `http://127.0.0.1:3000`.
- Dashboard screenshot was captured from the real rendered page with headless
  Chrome and visually inspected.
- Generated architecture, validation, category, and comparison images were
  visually inspected.
- LaTeX PDF was built with `pdflatex`.
- Final PDF was rendered with Poppler and visually inspected page by page.

Environment-limited coverage:

- External judge adapters are implemented, but live OpenAI, Anthropic, Gemini,
  and vLLM calls only run when credentials or endpoints are configured. The
  repeatable local validation path uses deterministic heuristic judges.
- Trace2Eval supports normalized trace records and local sample traces. Direct
  ingestion from OpenTelemetry, LangSmith, OpenInference, LiteLLM, production RAG
  stores, support systems, and incident-management tools still needs
  environment-specific adapters and credentials.
- DSPy, LangChain, LangGraph, FAISS, Qdrant, MinIO/S3, and PostgreSQL are mapped
  or adapter-ready, but the default runnable path remains local-first.
- Human review workflow artifacts and dashboard queues are implemented locally.
  Multi-user reviewer auth, assignment, audit logs, and durable review-state
  transitions remain production hardening work.
- GitHub Actions workflow and CI report generation are implemented in the repo.
  Posting PR comments and enforcing branch protection requires repository-level
  GitHub configuration.
- FailureGym is generated from compiler families and sample traces. A public
  benchmark release still requires larger trace-derived seed corpora, external
  model runs, and human-reviewed calibration sets.

Next:

1. Wire real production sources into Trace2Eval.

- Add direct adapters for OpenTelemetry spans, LangSmith traces,
  OpenInference traces, LiteLLM logs, retrieval logs, support tickets,
  thumbs-down feedback, red-team transcripts, and incident reports.
- Add connector-specific redaction policies and per-source provenance metadata.
- Build the polished demo path: upload 10 real failed RAG/tool traces and show
  one failure becoming dozens of versioned regression cases.

2. Run live multi-model judging and calibration.

- Configure OpenAI, Anthropic, Gemini, local Hugging Face, and vLLM judges in a
  controlled environment.
- Build a human-labeled gold seed set for each failure family.
- Measure human agreement, judge agreement, answer-order bias, verbosity bias,
  rubric sensitivity, and unreliable judge/category pairs.
- Promote only calibrated judge/category combinations into CI gates.

3. Make review workflow durable and collaborative.

- Persist review edits, approvals, rejections, ambiguity labels, and hidden-trap
  decisions as first-class state transitions.
- Add reviewer assignment, reviewer agreement, audit history, and exportable
  human-review reports.
- Add dashboard actions for approving BreakPointSpec drafts, editing rubrics,
  and promoting accepted cases into dataset versions.

4. Harden CI/CD around failure regression packs.

- Add a PR comment bot that publishes BreakPoint regression reports.
- Support pack selection by PR risk, changed files, recent incidents, and
  failure family.
- Track deltas for pass rate, mutation survival, RAG freshness,
  prompt-injection resistance, human-review backlog, latency, cost, and cost per
  reliable pass.
- Make branch protection consume BreakPoint gates directly.

5. Expand FailureGym from generated demo to public benchmark.

- Mine seed failures from real incident corpora or approved public traces.
- Add model runs across GPT, Claude, Gemini, and local open models.
- Publish pass rate, mutation survival, judge confidence, human-review
  agreement, failure-family breakdowns, lineage manifests, and reproducibility
  manifests.
- Keep the benchmark failure-family-first instead of broad-domain-first.

6. Add shared deployment storage.

- Add PostgreSQL migrations for product-layer state.
- Add MinIO/S3 artifact storage for traces, datasets, reports, screenshots, and
  manifests.
- Add FAISS/Qdrant-backed similarity search over failures, specs, prompts,
  variants, and rejected candidates.

7. Add richer generators after trace-derived loops are real.

- Prioritize new generators only when they exercise Trace2Eval, BreakPointSpec,
  simulator, judge calibration, or regression-gate behavior.
- Avoid adding more template-only domains until the failure-to-eval loop is
  producing real regression packs.

Deprioritize:

- A prettier generic dashboard without review actions.
- A broad public leaderboard before the failure-to-eval loop is proven.
- Shallow template-only domains that do not improve trace-derived regression
  coverage.
