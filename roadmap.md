# Roadmap

Exists:

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
- JSONL export for generated eval datasets.
- Hugging Face `Dataset` export helper.
- DuckDB persistence for eval items, validation reports, and dataset metrics.
- Local hash vector index used as a lightweight retrieval fallback when FAISS or
  Qdrant are not configured.
- LangGraph-compatible orchestration surface with a deterministic built-in graph
  fallback.
- FastAPI service with `/health`, `/categories`, `/compile`, and `/metrics`.
- CLI with `compile`, `categories`, and `serve` commands.
- Next.js dashboard showing coverage, validation gates, sample evals, generated
  result artifacts, and key metrics.
- Generated demo artifacts under `artifacts/demo`, including JSONL, preview JSON,
  metrics JSON, API output JSON, run output text, and local DuckDB store.
- Generated diagrams and charts under `artifacts/images`, including architecture,
  validation gates, category coverage, authoring comparison, and dashboard
  screenshot.
- LaTeX whitepaper source under `docs/latex`.
- Generated system design PDF at `output/pdf/breakpoint_system_design.pdf`.
- `.gitignore` covering virtualenvs, Node/Next build outputs, local stores,
  temporary PDF renders, and generated PDFs.

Validated:

- Demo compiler run generated 72 accepted base items, 288 adversarial variants,
  360 total eval cases, and 18 rejected candidates.
- Demo acceptance rate was 0.8 with balanced coverage across all nine failure
  categories.
- Python tests passed: 3 passed, 1 FastAPI/Starlette test-client deprecation
  warning.
- FastAPI compile endpoint was exercised through tests.
- DuckDB save/read path was exercised through tests.
- Hash retrieval fallback was exercised through tests.
- Next.js dashboard production build completed successfully.
- `npm audit --audit-level=moderate` reported 0 vulnerabilities after pinning a
  patched PostCSS override.
- Dashboard was served locally and returned HTTP 200 at `http://127.0.0.1:3000`.
- Dashboard screenshot was captured from the real rendered page with headless
  Chrome and visually inspected.
- Generated architecture, validation, category, and comparison images were
  visually inspected.
- LaTeX PDF was built with `pdflatex`.
- Final PDF was rendered with Poppler and visually inspected page by page.

Environment-limited coverage:

- Validation currently uses deterministic local surrogate judges. The interfaces
  are ready for real multi-model judging, but no external model APIs were called.
- DSPy, LangChain, LangGraph, FAISS, Qdrant, MinIO/S3, and PostgreSQL are mapped
  or adapter-ready, but the current runnable path is intentionally local-first.
- The dashboard is an inspection UI, not yet a full review/approval workflow.
- The current generators are template-driven. They prove the compiler shape, but
  they are not yet powered by mined production failures or incident corpora.
- The project directory was not a Git repository during this pass, so Git status
  and commit validation were not available.

Next:

- Add a failure-spec DSL so teams can define categories, task families,
  allowed/refused boundaries, hidden traps, evidence rules, and rubric criteria
  without editing Python templates.
- Add real model judge adapters through DSPy/LangChain for OpenAI, Anthropic,
  Gemini, local Hugging Face models, and configurable ensemble voting.
- Add judge calibration runs with gold seed sets, inter-judge agreement reports,
  and automatic threshold tuning.
- Mine real failure modes from incident reports, eval logs, user feedback,
  production traces, RAG retrieval failures, tool-call traces, and red-team
  transcripts.
- Add richer task compilers for long-context retrieval, RAG contradiction,
  prompt injection, multi-tool arbitration, refusal boundaries, structured
  outputs, and temporal reasoning.
- Add tool simulators that can return stale, partial, contradictory, malformed,
  or permission-blocked results.
- Add retrieval simulators backed by FAISS or Qdrant, including document
  freshness, source reliability, prompt-injection payloads, and citation
  constraints.
- Add PostgreSQL for shared project state and MinIO/S3 for generated artifact
  storage.
- Add dataset versioning, lineage, reproducibility manifests, and diff views
  between compiler runs.
- Add human review workflows in the dashboard: accept/reject, edit rubrics,
  inspect hidden traps, compare variants, and export approved subsets.
- Add eval-runner integrations for OpenAI Evals-style runners, lm-eval-harness,
  LangSmith, custom CI gates, and Hugging Face dataset publishing.
- Add quality analytics for category coverage, mutation survival rate,
  ambiguity rejection rate, model disagreement hotspots, and regression trends.
- Add CI that runs unit tests, API smoke tests, dashboard build, npm audit, demo
  artifact generation, LaTeX build, and PDF render checks.
- Add deployment profiles for local developer mode, Docker Compose, and cloud
  services with FastAPI, dashboard, PostgreSQL, Qdrant, and object storage.
- Add security controls for secrets, model API keys, uploaded corpora, generated
  unsafe-content boundaries, and prompt-injection test payload isolation.
- Add a public benchmark package once the compiler has enough validated,
  versioned, real-failure-derived eval families.
