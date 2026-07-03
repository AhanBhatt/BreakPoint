from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


LIVE_JUDGE_PREFIXES = ("openai:", "anthropic:", "gemini:", "vllm:")

JUDGE_LABELS = {
    "openai:gpt-5.4-mini": "OpenAI\nGPT-5.4 mini",
    "anthropic:claude-sonnet-5": "Anthropic\nClaude Sonnet 5",
    "gemini:gemini-2.5-flash": "Gemini\n2.5 Flash",
    "vllm:breakpoint-local-judge": "Local\nvLLM judge",
}

TRACE_LABELS = {
    "actual-air-canada-bereavement-2024": "Air Canada",
    "actual-nyc-mycity-illegal-advice-2024": "NYC MyCity",
    "actual-mata-avianca-fake-cases-2023": "Mata v Avianca",
    "actual-dpd-support-chatbot-2024": "DPD Support",
    "actual-chevrolet-chatbot-dollar-tahoe-2023": "Chevy Dealer",
    "actual-cnet-ai-finance-errors-2023": "CNET Finance",
    "actual-neda-tessa-harmful-advice-2023": "NEDA Tessa",
    "actual-microsoft-tay-toxic-learning-2016": "Microsoft Tay",
    "actual-paknsave-savey-mealbot-chlorine-2023": "Pak'nSave Mealbot",
    "actual-google-bard-jwst-error-2023": "Google Bard JWST",
    "actual-michael-cohen-fake-citations-2023": "Cohen Citations",
    "actual-mcdonalds-ai-drive-thru-errors-2024": "McDonald's Drive-Thru",
}

FAMILY_COLORS = {
    "rag_contradiction": "#14b8a6",
    "refusal_boundary": "#f59e0b",
    "prompt_injection": "#ef4444",
    "tool_misuse": "#60a5fa",
}

JUDGE_COLORS = {
    "openai:gpt-5.4-mini": "#22c55e",
    "anthropic:claude-sonnet-5": "#fb923c",
    "gemini:gemini-2.5-flash": "#60a5fa",
    "vllm:breakpoint-local-judge": "#a78bfa",
}

JUDGE_COST_PER_TEST_USD = {
    "openai:gpt-5.4-mini": 0.00216,
    "anthropic:claude-sonnet-5": 0.00540,
    "gemini:gemini-2.5-flash": 0.00099,
    "vllm:breakpoint-local-judge": 0.00005,
}


def write_live_judge_report(
    results_path: str | Path = "artifacts/actual/trace2eval_results.json",
    out_dir: str | Path = "artifacts/reports",
) -> dict[str, Any]:
    results = _load_results(results_path)
    report = build_live_judge_report(results)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "live_judge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    draw_live_judge_charts(report, target)
    return report


def build_live_judge_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_trace_row(result) for result in results]
    judge_names = sorted({vote["model_name"] for row in rows for vote in row["live_votes"]})
    judge_summaries = [_judge_summary(judge, rows) for judge in judge_names]
    judge_summaries.sort(key=lambda item: item["breakpoint_reliability"], reverse=True)
    indices = _breakpoint_indices(rows)
    return {
        "report": "breakpoint-live-judge-evaluation",
        "trace_count": len(rows),
        "judge_count": len(judge_names),
        "custom_indices": [
            "Failure Neighborhood Index",
            "Evidence Conflict Index",
            "Boundary Precision Index",
            "Instruction Attack Index",
            "Judge Consensus Index",
            "Source Tension Index",
        ],
        "trace_rows": rows,
        "judge_summaries": judge_summaries,
        "breakpoint_indices": indices,
    }


def draw_live_judge_charts(report: dict[str, Any], out_dir: str | Path) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#dbeafe",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
        }
    )

    paths = []
    paths.append(str(_draw_scoreboard(report, target / "judge_scoreboard.png", plt)))
    paths.append(str(_draw_confidence_matrix(report, target / "judge_confidence_matrix.png", plt, LinearSegmentedColormap)))
    paths.append(str(_draw_trace_indices(report, target / "failure_index_by_trace.png", plt, Patch)))
    paths.append(str(_draw_pareto(report, target / "cost_reliability_pareto.png", plt)))
    return paths


def _load_results(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Trace2Eval results must be a JSON list")
    return data


def _trace_row(result: dict[str, Any]) -> dict[str, Any]:
    seed = result["seed"]
    trace = seed["redacted_trace"]
    item = result["eval_item"]
    report = result["eval_case"]["validation_report"]
    docs = trace.get("retrieved_docs", [])
    reliabilities = [float(doc.get("reliability", 0)) for doc in docs]
    reliability_spread = round((max(reliabilities) - min(reliabilities)) if reliabilities else 0.0, 3)
    live_votes = [vote for vote in report["votes"] if str(vote["model_name"]).startswith(LIVE_JUDGE_PREFIXES)]
    external_agreement = _safe_mean([1.0 if vote["passed"] else 0.0 for vote in live_votes])
    confidence_mean = _safe_mean([float(vote["confidence"]) for vote in live_votes])
    severity_weight = {"low": 0.35, "medium": 0.55, "high": 0.78, "critical": 0.95}.get(trace.get("incident_severity"), 0.55)
    context_words = len(str(item.get("context", "")).split())
    variant_count = len(item.get("adversarial_variants", []))
    breakpoint_index = round(
        100
        * (
            0.42 * external_agreement
            + 0.30 * confidence_mean
            + 0.12 * reliability_spread
            + 0.10 * severity_weight
            + 0.06 * min(1.0, variant_count / 5)
        ),
        1,
    )
    return {
        "trace_id": seed["trace_id"],
        "label": TRACE_LABELS.get(seed["trace_id"], seed["trace_id"].replace("actual-", "").replace("-", " ").title()),
        "family": seed["family"],
        "severity": trace.get("incident_severity", "medium"),
        "context_words": context_words,
        "variant_count": variant_count,
        "source_count": len(docs),
        "source_reliability_spread": reliability_spread,
        "external_agreement": round(external_agreement, 3),
        "external_confidence": round(confidence_mean, 3),
        "breakpoint_failure_index": breakpoint_index,
        "live_votes": live_votes,
    }


def _judge_summary(judge: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    votes = []
    majority = []
    for row in rows:
        row_votes = row["live_votes"]
        vote = next((item for item in row_votes if item["model_name"] == judge), None)
        if vote is None:
            continue
        votes.append(vote)
        passed_votes = sum(1 for item in row_votes if item["passed"])
        majority.append(passed_votes >= (len(row_votes) / 2))
    pass_rate = _safe_mean([1.0 if vote["passed"] else 0.0 for vote in votes])
    avg_confidence = _safe_mean([float(vote["confidence"]) for vote in votes])
    disagreement_rate = _safe_mean([1.0 if bool(vote["passed"]) != bool(maj) else 0.0 for vote, maj in zip(votes, majority, strict=False)])
    parse_success = _safe_mean([0.0 if str(vote.get("rationale", "")).startswith(("non-json", "adapter error")) else 1.0 for vote in votes])
    reliability = round(100 * (0.40 * pass_rate + 0.35 * avg_confidence + 0.15 * (1 - disagreement_rate) + 0.10 * parse_success), 2)
    return {
        "judge": judge,
        "label": JUDGE_LABELS.get(judge, judge),
        "pass_rate": round(pass_rate, 3),
        "average_confidence": round(avg_confidence, 3),
        "disagreement_rate": round(disagreement_rate, 3),
        "parse_success_rate": round(parse_success, 3),
        "cost_per_test_usd": JUDGE_COST_PER_TEST_USD.get(judge, 0.0),
        "breakpoint_reliability": reliability,
    }


def _breakpoint_indices(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    all_scores = [row["breakpoint_failure_index"] for row in rows]
    return {
        "Failure Neighborhood Index": round(_safe_mean(all_scores), 1),
        "Evidence Conflict Index": round(_safe_mean([row["breakpoint_failure_index"] for row in by_family["rag_contradiction"]]), 1),
        "Boundary Precision Index": round(_safe_mean([row["breakpoint_failure_index"] for row in by_family["refusal_boundary"]]), 1),
        "Instruction Attack Index": round(_safe_mean([row["breakpoint_failure_index"] for row in by_family["prompt_injection"]]), 1),
        "Judge Consensus Index": round(100 * _safe_mean([row["external_agreement"] for row in rows]), 1),
        "Source Tension Index": round(100 * _safe_mean([row["source_reliability_spread"] for row in rows]), 1),
    }


def _draw_scoreboard(report: dict[str, Any], path: Path, plt: Any) -> Path:
    judges = report["judge_summaries"]
    labels = [judge["label"] for judge in judges]
    values = [judge["breakpoint_reliability"] for judge in judges]
    colors = [JUDGE_COLORS.get(judge["judge"], "#94a3b8") for judge in judges]
    fig, ax = plt.subplots(figsize=(13.5, 7), dpi=170)
    _dark_canvas(fig, ax)
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="#e5e7eb", linewidth=0.8)
    for index, (bar, judge) in enumerate(zip(bars, judges, strict=True)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.1, f"{values[index]:.1f}", ha="center", fontsize=11, weight="bold")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            5,
            f"pass {judge['pass_rate']:.0%}\nconf {judge['average_confidence']:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0f172a",
            weight="bold",
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("BreakPoint reliability score")
    ax.set_title("Live Judge Reliability on Actual Failure-Derived Cases", fontsize=20, pad=28)
    trace_count = int(report.get("trace_count", len(report.get("trace_rows", []))))
    trace_word = "trace" if trace_count == 1 else "traces"
    fig.text(
        0.5,
        0.89,
        f"OpenAI, Anthropic, Gemini, and local vLLM judged the same {trace_count} public failure {trace_word}.",
        ha="center",
        color="#cbd5e1",
        fontsize=11,
    )
    ax.grid(axis="y", color="#334155", alpha=0.75)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _draw_confidence_matrix(report: dict[str, Any], path: Path, plt: Any, cmap_cls: Any) -> Path:
    rows = report["trace_rows"]
    judges = [summary["judge"] for summary in report["judge_summaries"]]
    values = []
    passed = []
    for row in rows:
        vote_map = {vote["model_name"]: vote for vote in row["live_votes"]}
        values.append([float(vote_map[judge]["confidence"]) * 100 for judge in judges])
        passed.append([bool(vote_map[judge]["passed"]) for judge in judges])
    cmap = cmap_cls.from_list("breakpoint_conf", ["#312e81", "#0e7490", "#22c55e", "#facc15"])
    fig, ax = plt.subplots(figsize=(12.5, 8), dpi=170)
    _dark_canvas(fig, ax)
    image = ax.imshow(values, cmap=cmap, vmin=65, vmax=100, aspect="auto")
    ax.set_xticks(range(len(judges)), [JUDGE_LABELS.get(judge, judge).replace("\n", " ") for judge in judges], rotation=18, ha="right")
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            marker = "" if passed[y][x] else " x"
            ax.text(x, y, f"{value:.0f}{marker}", ha="center", va="center", fontsize=9, weight="bold", color="#ffffff")
    ax.set_title("Judge Confidence by Failure Trace", fontsize=20, pad=18)
    ax.text(0.0, 1.02, "Numbers are live judge confidence percentages; x marks a failed/rejected judgment.", transform=ax.transAxes, color="#cbd5e1")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025)
    colorbar.set_label("confidence")
    colorbar.ax.yaxis.set_tick_params(color="#cbd5e1")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _draw_trace_indices(report: dict[str, Any], path: Path, plt: Any, patch_cls: Any) -> Path:
    rows = sorted(report["trace_rows"], key=lambda row: row["breakpoint_failure_index"])
    fig, ax = plt.subplots(figsize=(13.5, 8), dpi=170)
    _dark_canvas(fig, ax)
    colors = [FAMILY_COLORS.get(row["family"], "#94a3b8") for row in rows]
    values = [row["breakpoint_failure_index"] for row in rows]
    labels = [row["label"] for row in rows]
    bars = ax.barh(labels, values, color=colors, edgecolor="#e5e7eb", linewidth=0.7)
    for bar, row in zip(bars, rows, strict=True):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"{row['breakpoint_failure_index']:.1f}", va="center", fontsize=10, weight="bold")
        ax.text(2, bar.get_y() + bar.get_height() / 2, row["severity"], va="center", fontsize=9, color="#111827", weight="bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Failure Neighborhood Index")
    ax.set_title("BreakPoint Failure Indices by Public Incident", fontsize=20, pad=18)
    ax.grid(axis="x", color="#334155", alpha=0.75)
    legend = [
        patch_cls(facecolor=color, label=family.replace("_", " "))
        for family, color in FAMILY_COLORS.items()
        if any(row["family"] == family for row in rows)
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, labelcolor="#e2e8f0")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _draw_pareto(report: dict[str, Any], path: Path, plt: Any) -> Path:
    judges = report["judge_summaries"]
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=170)
    _dark_canvas(fig, ax)
    for judge in judges:
        x = judge["cost_per_test_usd"] * 100
        y = judge["breakpoint_reliability"]
        color = JUDGE_COLORS.get(judge["judge"], "#94a3b8")
        ax.scatter(x, y, s=360 * max(0.25, judge["pass_rate"]), color=color, edgecolor="#f8fafc", linewidth=1.2, alpha=0.92)
        ax.text(x + 0.012, y + 0.4, judge["label"].replace("\n", " "), fontsize=10, color="#e2e8f0")
    ax.set_xlabel("estimated cents per judged case")
    ax.set_ylabel("BreakPoint reliability score")
    ax.set_title("Cost vs Reliable Judgment Pareto", fontsize=20, pad=26)
    fig.text(
        0.5,
        0.89,
        "vLLM is a local endpoint estimate; hosted-provider cost is estimated from configured judge token budgets.",
        ha="center",
        color="#cbd5e1",
        fontsize=11,
    )
    ax.grid(color="#334155", alpha=0.75)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _dark_canvas(fig: Any, ax: Any) -> None:
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#172033")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.tick_params(colors="#cbd5e1")


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0
