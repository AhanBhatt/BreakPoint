import fs from "node:fs";
import path from "node:path";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Code2,
  Database,
  FileCheck2,
  FlaskConical,
  GitBranch,
  GitCompare,
  Inbox,
  Network,
  ShieldCheck,
  Sparkles
} from "lucide-react";

type Metrics = {
  accepted_items: number;
  adversarial_variants: number;
  total_eval_cases: number;
  rejected_candidates: number;
  acceptance_rate: number;
  estimated_manual_prompt_hours: number;
  category_counts: Record<string, number>;
  quality_scores: Record<string, number>;
};

type SampleItem = {
  title: string;
  category: string;
  task: string;
  expected: string;
  variant_count: number;
};

type ProductSummary = {
  case_count: number;
  native_metrics: Record<string, number | Record<string, unknown>>;
  clusters: Array<{ id: string; family: string; title: string; case_ids: string[]; severity: string }>;
};

type FailureInboxItem = {
  id: string;
  family: string;
  summary: string;
  confidence: number;
  signals: string[];
  review_status: string;
};

type SpecReview = {
  spec_yaml: string;
  compiled_item: { title: string; category: string; variant_count: number };
  trace_specs: Array<{ id: string; name: string; risk: string }>;
};

type CaseReview = {
  cases: Array<{
    id: string;
    failure_family: string;
    review_status: string;
    eval_item: { title: string; task: string };
    mutation_lineage: { mutator: string; generation: number };
  }>;
  reviews: Array<{ case_id: string; reviewer: string; status: string; notes: string; ambiguity_flag: boolean }>;
};

type MutationNode = {
  id: string;
  parent: string | null;
  seed: string;
  mutator: string;
  family: string;
  title: string;
};

type RunComparison = {
  previous: Record<string, number>;
  delta: Record<string, number>;
};

type RegressionGate = {
  result: { passed: boolean; failures: string[]; pass_rate: number; mutation_survival_rate: number; needs_review: number };
  ci_report: { summary_markdown: string };
};

type JudgeReliability = Array<{
  judge_name: string;
  accuracy: number;
  false_positive_rate: number;
  false_negative_rate: number;
  average_confidence: number;
  notes: string[];
}>;

type SimulatorItem = { id: string; family: string; expected_path: string[]; scoring_notes: string[] };

type FailureGym = {
  benchmark: string;
  version: string;
  tracks: Array<{ id: string; name: string; case_count: number; families: string[] }>;
};

const fallbackMetrics: Metrics = {
  accepted_items: 0,
  adversarial_variants: 0,
  total_eval_cases: 0,
  rejected_candidates: 0,
  acceptance_rate: 0,
  estimated_manual_prompt_hours: 0,
  category_counts: {},
  quality_scores: {}
};

function readJson<T>(relativePath: string, fallback: T): T {
  const file = path.join(process.cwd(), "public", relativePath);
  if (!fs.existsSync(file)) {
    return fallback;
  }
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

function prettyCategory(value: string) {
  return value.replaceAll("_", " ");
}

function prettyMutator(value: string) {
  const labels: Record<string, string> = {
    base: "base",
    conflicting_evidence: "conflict",
    irrelevant_context: "irrelevant",
    paraphrased_instruction: "paraphrase",
    renamed_entities: "renamed",
    reordered_facts: "reordered"
  };
  return labels[value] ?? value.replaceAll("_", " ");
}

export default function Home() {
  const metrics = readJson<Metrics>("data/metrics.json", fallbackMetrics);
  const samples = readJson<SampleItem[]>("data/sample_items.json", []);
  const product = readJson<ProductSummary>("data/product_summary.json", { case_count: 0, native_metrics: {}, clusters: [] });
  const inbox = readJson<FailureInboxItem[]>("data/failure_inbox.json", []);
  const specReview = readJson<SpecReview>("data/spec_review.json", { spec_yaml: "", compiled_item: { title: "", category: "", variant_count: 0 }, trace_specs: [] });
  const caseReview = readJson<CaseReview>("data/case_review.json", { cases: [], reviews: [] });
  const mutationGraph = readJson<MutationNode[]>("data/mutation_graph.json", []);
  const runComparison = readJson<RunComparison>("data/run_comparison.json", { previous: {}, delta: {} });
  const regressionGate = readJson<RegressionGate>("data/regression_gate.json", { result: { passed: false, failures: [], pass_rate: 0, mutation_survival_rate: 0, needs_review: 0 }, ci_report: { summary_markdown: "" } });
  const judgeReliability = readJson<JudgeReliability>("data/judge_reliability.json", []);
  const simulators = readJson<SimulatorItem[]>("data/simulators.json", []);
  const failureGym = readJson<FailureGym>("data/failuregym.json", { benchmark: "BreakPoint-FailureGym", version: "v1", tracks: [] });
  const maxCategory = Math.max(...Object.values(metrics.category_counts), 1);
  const quality = metrics.quality_scores ?? {};
  const native = product.native_metrics ?? {};

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark"><FlaskConical size={20} /></span>
          BreakPoint
        </div>
        <nav className="nav" aria-label="Dashboard">
          <span className="nav-item active"><Activity size={16} /> Overview</span>
          <span className="nav-item"><Inbox size={16} /> Inbox</span>
          <span className="nav-item"><Code2 size={16} /> Specs</span>
          <span className="nav-item"><GitBranch size={16} /> Mutations</span>
          <span className="nav-item"><ShieldCheck size={16} /> Gates</span>
          <span className="nav-item"><Database size={16} /> Exports</span>
        </nav>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <h1>Eval Data Compiler</h1>
            <p className="lede">
              Turns real LLM failures into versioned regression suites with specs, mutation lineage,
              judge calibration, human review, CI gates, and exportable failure packs.
            </p>
          </div>
          <span className="pill">v0.2 failure-to-eval loop</span>
        </header>

        <section className="kpi-grid" aria-label="Key metrics">
          <Kpi icon={<FileCheck2 size={16} />} label="Accepted base items" value={metrics.accepted_items} />
          <Kpi icon={<Sparkles size={16} />} label="Adversarial variants" value={metrics.adversarial_variants} />
          <Kpi icon={<Database size={16} />} label="Total eval cases" value={metrics.total_eval_cases} />
          <Kpi icon={<Network size={16} />} label="Suite cases" value={product.case_count} />
          <Kpi icon={<ShieldCheck size={16} />} label="Mutation survival" value={`${toPct(native.mutation_survival_rate)}%`} />
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title"><BarChart3 size={18} /> Failure category coverage</div>
            <div className="bars">
              {Object.entries(metrics.category_counts).map(([category, count], index) => (
                <div className="bar-row" key={category}>
                  <span>{prettyCategory(category)}</span>
                  <span className="bar-track">
                    <span className="bar-fill" style={{ width: `${(count / maxCategory) * 100}%`, background: palette[index % palette.length] }} />
                  </span>
                  <strong>{count}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="panel-title"><ShieldCheck size={18} /> Validation gates</div>
            <div className="quality-grid">
              <Quality label="Answerability" value={quality.answerability_score} />
              <Quality label="Trap coverage" value={quality.trap_coverage_score} />
              <Quality label="Format score" value={quality.format_score} />
              <Quality label="Judge agreement" value={quality.judge_agreement} />
            </div>
          </div>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title"><Inbox size={18} /> Failure inbox</div>
            <div className="sample-list">
              {inbox.map((item) => (
                <article className="sample" key={item.id}>
                  <div className="sample-head">
                    <span className="sample-title">{item.summary}</span>
                    <span className="tag">{prettyCategory(item.family)} · {toPct(item.confidence)}%</span>
                  </div>
                  <p>{item.signals.join(", ")} · {item.review_status}</p>
                </article>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="panel-title"><Code2 size={18} /> Spec review</div>
            <div className="code-box">{specReview.spec_yaml.split("\n").slice(0, 15).join("\n")}</div>
            <div className="mini-grid">
              {specReview.trace_specs.slice(0, 3).map((spec) => (
                <div className="quality" key={spec.id}>
                  <strong>{spec.risk}</strong>
                  <span>{spec.name}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title"><FileCheck2 size={18} /> Case review queue</div>
            <div className="sample-list">
              {caseReview.cases.slice(0, 5).map((item) => (
                <article className="sample" key={item.id}>
                  <div className="sample-head">
                    <span className="sample-title">{item.eval_item.title}</span>
                    <span className="tag">{item.review_status} · {prettyMutator(item.mutation_lineage.mutator)}</span>
                  </div>
                  <p>{item.eval_item.task}</p>
                </article>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="panel-title"><Network size={18} /> Mutation graph</div>
            <div className="node-list">
              {mutationGraph.slice(0, 12).map((node) => (
                <div className={`node gen-${Math.min(node.parent ? 1 : 0, 1)}`} key={node.id}>
                  <strong>{prettyMutator(node.mutator)}</strong>
                  <span>{prettyCategory(node.family)}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title"><GitCompare size={18} /> Run comparison</div>
            <div className="quality-grid">
              <Quality label="Pass rate delta" value={runComparison.delta.pass_rate ?? 0} />
              <Quality label="Mutation delta" value={runComparison.delta.mutation_survival_rate ?? 0} />
              <Quality label="Freshness delta" value={runComparison.delta.evidence_freshness_score ?? 0} />
              <Quality label="Oracle confidence" value={Number(native.oracle_confidence ?? 0)} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-title">{regressionGate.result.passed ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />} Regression gate</div>
            <div className="gate-status">{regressionGate.result.passed ? "PASS" : "FAIL"}</div>
            <p className="panel-copy">{regressionGate.result.failures.length ? regressionGate.result.failures.join(" · ") : "All configured gate thresholds passed."}</p>
            <div className="quality-grid">
              <Quality label="Pass rate" value={regressionGate.result.pass_rate} />
              <Quality label="Mutation survival" value={regressionGate.result.mutation_survival_rate} />
            </div>
          </div>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title"><ShieldCheck size={18} /> Judge reliability</div>
            <div className="bars">
              {judgeReliability.map((judge) => (
                <div className="bar-row" key={judge.judge_name}>
                  <span>{judge.judge_name}</span>
                  <span className="bar-track"><span className="bar-fill" style={{ width: `${judge.accuracy * 100}%` }} /></span>
                  <strong>{toPct(judge.accuracy)}%</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="panel-title"><FlaskConical size={18} /> Simulators</div>
            <div className="sample-list">
              {simulators.slice(0, 5).map((sim) => (
                <article className="sample" key={sim.id}>
                  <div className="sample-head">
                    <span className="sample-title">{sim.id}</span>
                    <span className="tag">{prettyCategory(sim.family)}</span>
                  </div>
                  <p>{sim.scoring_notes.join(" ")}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="grid">
          <div className="panel">
            <div className="panel-title"><FileCheck2 size={18} /> Sample generated evals</div>
            <div className="sample-list">
              {samples.slice(0, 4).map((sample) => (
                <article className="sample" key={`${sample.category}-${sample.title}`}>
                  <div className="sample-head">
                    <span className="sample-title">{sample.title}</span>
                    <span className="tag">{prettyCategory(sample.category)} · {sample.variant_count} variants</span>
                  </div>
                  <p>{sample.task}</p>
                </article>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="panel-title"><Activity size={18} /> Generated result artifacts</div>
            <div className="sample-list">
              <img className="artifact" src="/images/quality_gates.png" alt="Validation gate chart" />
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-title"><Database size={18} /> {failureGym.benchmark} {failureGym.version}</div>
          <div className="artifact-row">
            {failureGym.tracks.map((track) => (
              <div className="quality" key={track.id}>
                <strong>{track.case_count}</strong>
                <span>{track.name}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="artifact-row">
          <img className="artifact" src="/images/architecture.png" alt="BreakPoint architecture" />
          <img className="artifact" src="/images/category_mix.png" alt="Category coverage" />
          <img className="artifact" src="/images/comparison_results.png" alt="Authoring effort comparison" />
        </section>
      </section>
    </main>
  );
}

function Kpi({ icon, label, value }: { icon: React.ReactNode; label: string; value: number | string }) {
  return (
    <div className="card">
      <span className="kpi-label">{icon}{label}</span>
      <div className="kpi-value">{value}</div>
    </div>
  );
}

function Quality({ label, value = 0 }: { label: string; value?: number }) {
  return (
    <div className="quality">
      <strong>{Number(value).toFixed(2)}</strong>
      <span>{label}</span>
    </div>
  );
}

function toPct(value: unknown) {
  return Math.round(Number(value ?? 0) * 100);
}

const palette = ["#2563eb", "#0f766e", "#7c3aed", "#be123c", "#b45309", "#475569", "#0891b2", "#4d7c0f", "#c2410c"];
