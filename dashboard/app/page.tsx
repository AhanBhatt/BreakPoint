import fs from "node:fs";
import path from "node:path";
import { Activity, BarChart3, Database, FileCheck2, FlaskConical, GitBranch, ShieldCheck, Sparkles } from "lucide-react";

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

export default function Home() {
  const metrics = readJson<Metrics>("data/metrics.json", fallbackMetrics);
  const samples = readJson<SampleItem[]>("data/sample_items.json", []);
  const maxCategory = Math.max(...Object.values(metrics.category_counts), 1);
  const quality = metrics.quality_scores ?? {};

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark"><FlaskConical size={20} /></span>
          BreakPoint
        </div>
        <nav className="nav" aria-label="Dashboard">
          <span className="nav-item active"><Activity size={16} /> Overview</span>
          <span className="nav-item"><GitBranch size={16} /> Mutations</span>
          <span className="nav-item"><ShieldCheck size={16} /> Validators</span>
          <span className="nav-item"><Database size={16} /> Exports</span>
        </nav>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <h1>Eval Data Compiler</h1>
            <p className="lede">
              Generates targeted eval cases from failure categories, mutates them into adversarial variants,
              validates quality with multiple judges, and exports review-ready datasets.
            </p>
          </div>
          <span className="pill">mode: deterministic-local</span>
        </header>

        <section className="kpi-grid" aria-label="Key metrics">
          <Kpi icon={<FileCheck2 size={16} />} label="Accepted base items" value={metrics.accepted_items} />
          <Kpi icon={<Sparkles size={16} />} label="Adversarial variants" value={metrics.adversarial_variants} />
          <Kpi icon={<Database size={16} />} label="Total eval cases" value={metrics.total_eval_cases} />
          <Kpi icon={<ShieldCheck size={16} />} label="Rejected candidates" value={metrics.rejected_candidates} />
          <Kpi icon={<BarChart3 size={16} />} label="Manual hours avoided" value={`${metrics.estimated_manual_prompt_hours}h`} />
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

const palette = ["#2563eb", "#0f766e", "#7c3aed", "#be123c", "#b45309", "#475569", "#0891b2", "#4d7c0f", "#c2410c"];
