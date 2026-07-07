from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from pydantic import BaseModel, Field


class GoldLabel(BaseModel):
    item_id: str
    trace_id: str
    family: str
    human_passed: bool
    case_id: str | None = None
    reviewer: str = "gold-reviewer"
    confidence: float = Field(default=0.9, ge=0, le=1)
    notes: str = ""
    label_source: str = "gold_label"
    candidate_output_passed: str | None = None
    severity: str | None = None
    failure_family_correct: str | None = None
    needs_domain_expert: str | None = None


class JudgeFamilyCalibration(BaseModel):
    judge_name: str
    family: str
    examples: int
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    average_confidence: float
    promoted: bool


class JudgeCalibrationSuiteReport(BaseModel):
    examples: int
    judges: list[str]
    families: list[str]
    overall: dict[str, Any]
    by_judge_family: list[JudgeFamilyCalibration]
    inter_judge_agreement: dict[str, float]
    answer_order_bias: dict[str, float]
    verbosity_bias: dict[str, float]
    rubric_sensitivity: dict[str, float]
    promoted_pairs: list[dict[str, str]]
    needs_human_review: list[dict[str, Any]]


def labels_from_trace2eval_results(results: list[dict[str, Any]]) -> list[GoldLabel]:
    labels = []
    for result in results:
        labels.append(
            GoldLabel(
                item_id=result["eval_item"]["id"],
                trace_id=result["seed"]["trace_id"],
                family=result["seed"]["family"],
                human_passed=bool(result["validation_passed"]),
                confidence=0.92 if result["validation_passed"] else 0.72,
                notes="Seeded from accepted Trace2Eval validation; replace with reviewer labels for production calibration.",
            )
        )
    return labels


def load_gold_labels(path: str | Path, results: list[dict[str, Any]] | None = None) -> list[GoldLabel]:
    label_path = Path(path)
    text = label_path.read_text(encoding="utf-8")
    if label_path.suffix.lower() == ".jsonl":
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("labels", [])
    family_by_item, family_by_case, family_by_trace = _label_family_maps(results or [])
    return [
        _normalize_gold_label(item, family_by_item=family_by_item, family_by_case=family_by_case, family_by_trace=family_by_trace)
        for item in data
    ]


def write_gold_labels(labels: list[GoldLabel], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"labels": [label.model_dump(mode="json") for label in labels]}, indent=2), encoding="utf-8")


def calibrate_from_trace2eval_results(
    results_path: str | Path,
    *,
    labels_path: str | Path | None = None,
    out_dir: str | Path = "artifacts/calibration",
    min_accuracy: float = 0.78,
    min_examples: int = 1,
) -> JudgeCalibrationSuiteReport:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    labels = load_gold_labels(labels_path, results=results) if labels_path else labels_from_trace2eval_results(results)
    report = calibrate_votes(results, labels, min_accuracy=min_accuracy, min_examples=min_examples)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "judge_calibration_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    write_gold_labels(labels, target / "gold_labels.json")
    (target / "calibrated_gate_policy.json").write_text(
        json.dumps({"promoted_pairs": report.promoted_pairs}, indent=2),
        encoding="utf-8",
    )
    return report


def _label_family_maps(
    results: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    family_by_item = {}
    family_by_case = {}
    family_by_trace = {}
    for result in results:
        family = result["seed"]["family"]
        item_id = result["eval_item"]["id"]
        case_id = result["eval_case"]["id"]
        trace_id = result["seed"]["trace_id"]
        family_by_item[item_id] = family
        family_by_case[case_id] = family
        family_by_trace[trace_id] = family
    return family_by_item, family_by_case, family_by_trace


def _normalize_gold_label(
    item: dict[str, Any],
    *,
    family_by_item: dict[str, str],
    family_by_case: dict[str, str],
    family_by_trace: dict[str, str],
) -> GoldLabel:
    if "family" in item and "human_passed" in item:
        return _apply_compiler_acceptance(GoldLabel.model_validate(item))

    item_id = str(item.get("item_id", ""))
    case_id = str(item.get("case_id", "")) or None
    trace_id = str(item.get("trace_id", ""))
    family = (
        item.get("family")
        or item.get("failure_family")
        or family_by_item.get(item_id)
        or (family_by_case.get(case_id) if case_id else None)
        or family_by_trace.get(trace_id)
        or "unknown"
    )
    reviewer_confidence = item.get("reviewer_confidence", item.get("confidence", 4))
    confidence = _reviewer_confidence_to_unit(reviewer_confidence)
    human_passed = item.get("human_passed")
    if human_passed is None:
        human_passed = _derive_human_passed(item)

    notes = str(item.get("notes", ""))
    if item.get("failure_family_correct") and item.get("failure_family_correct") != "yes":
        notes = (notes + " " if notes else "") + f"[failure_family_correct={item.get('failure_family_correct')}]"

    return _apply_compiler_acceptance(GoldLabel(
        item_id=item_id,
        case_id=case_id,
        trace_id=trace_id,
        family=str(family),
        human_passed=bool(human_passed),
        reviewer=str(item.get("reviewer", item.get("reviewer_id", "human-reviewer"))),
        confidence=confidence,
        notes=notes,
        label_source=str(item.get("label_source", "human_review_jsonl")),
        candidate_output_passed=item.get("candidate_output_passed"),
        severity=item.get("severity"),
        failure_family_correct=item.get("failure_family_correct"),
        needs_domain_expert=item.get("needs_domain_expert"),
    ))


def _apply_compiler_acceptance(label: GoldLabel) -> GoldLabel:
    if not label.human_passed:
        return label
    reasons = []
    if label.failure_family_correct == "no":
        reasons.append("failure_family_correct=no")
    if label.needs_domain_expert == "yes":
        reasons.append("needs_domain_expert=yes")
    if not reasons:
        return label
    note = label.notes
    marker = f"[compiler_acceptance=false: {', '.join(reasons)}]"
    if marker not in note:
        note = (note + " " if note else "") + marker
    return label.model_copy(update={"human_passed": False, "notes": note})


def _derive_human_passed(item: dict[str, Any]) -> bool:
    return (
        item.get("case_valid") == "yes"
        and item.get("expected_answer_correct") == "yes"
        and item.get("hidden_trap_valid") == "yes"
        and item.get("rubric_clear") == "yes"
        and item.get("ambiguous") == "no"
    )


def _reviewer_confidence_to_unit(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.8
    if numeric > 1:
        return round(max(0.2, min(1.0, numeric / 5)), 3)
    return round(max(0.0, min(1.0, numeric)), 3)


def calibrate_votes(
    results: list[dict[str, Any]],
    labels: list[GoldLabel],
    *,
    min_accuracy: float = 0.78,
    min_examples: int = 1,
) -> JudgeCalibrationSuiteReport:
    label_map = {label.item_id: label for label in labels}
    rows = []
    for result in results:
        item_id = result["eval_item"]["id"]
        label = label_map.get(item_id)
        if not label:
            continue
        votes = result["eval_case"]["validation_report"]["votes"]
        rows.append({"result": result, "label": label, "votes": votes})

    judges = sorted({vote["model_name"] for row in rows for vote in row["votes"]})
    families = sorted({row["label"].family for row in rows})
    by_judge_family = []
    promoted_pairs = []
    for judge in judges:
        for family in families:
            selected = [(row, _vote_for(row["votes"], judge)) for row in rows if row["label"].family == family]
            selected = [(row, vote) for row, vote in selected if vote is not None]
            if not selected:
                continue
            examples = len(selected)
            correct = sum(1 for row, vote in selected if bool(vote["passed"]) == row["label"].human_passed)
            false_positive = sum(1 for row, vote in selected if bool(vote["passed"]) and not row["label"].human_passed)
            false_negative = sum(1 for row, vote in selected if not bool(vote["passed"]) and row["label"].human_passed)
            accuracy = correct / examples
            promoted = examples >= min_examples and accuracy >= min_accuracy
            calibration = JudgeFamilyCalibration(
                judge_name=judge,
                family=family,
                examples=examples,
                accuracy=round(accuracy, 3),
                false_positive_rate=round(false_positive / examples, 3),
                false_negative_rate=round(false_negative / examples, 3),
                average_confidence=round(mean(float(vote["confidence"]) for _, vote in selected), 3),
                promoted=promoted,
            )
            by_judge_family.append(calibration)
            if promoted:
                promoted_pairs.append({"judge": judge, "family": family})

    return JudgeCalibrationSuiteReport(
        examples=len(rows),
        judges=judges,
        families=families,
        overall=_overall(rows, judges),
        by_judge_family=by_judge_family,
        inter_judge_agreement=_inter_judge_agreement(rows, judges),
        answer_order_bias=_answer_order_bias(rows, judges),
        verbosity_bias=_verbosity_bias(rows, judges),
        rubric_sensitivity=_rubric_sensitivity(rows, judges),
        promoted_pairs=promoted_pairs,
        needs_human_review=_needs_human_review(rows, judges),
    )


def _vote_for(votes: list[dict[str, Any]], judge: str) -> dict[str, Any] | None:
    return next((vote for vote in votes if vote["model_name"] == judge), None)


def _overall(rows: list[dict[str, Any]], judges: list[str]) -> dict[str, Any]:
    metrics = {}
    for judge in judges:
        selected = [(row, _vote_for(row["votes"], judge)) for row in rows]
        selected = [(row, vote) for row, vote in selected if vote is not None]
        if not selected:
            continue
        metrics[judge] = {
            "accuracy": round(mean(1.0 if bool(vote["passed"]) == row["label"].human_passed else 0.0 for row, vote in selected), 3),
            "average_confidence": round(mean(float(vote["confidence"]) for _, vote in selected), 3),
        }
    return metrics


def _inter_judge_agreement(rows: list[dict[str, Any]], judges: list[str]) -> dict[str, float]:
    output = {}
    for judge in judges:
        agreements = []
        for row in rows:
            vote = _vote_for(row["votes"], judge)
            if vote is None:
                continue
            other = [other for other in row["votes"] if other["model_name"] != judge]
            if not other:
                continue
            majority = sum(1 for other_vote in other if other_vote["passed"]) >= len(other) / 2
            agreements.append(1.0 if bool(vote["passed"]) == majority else 0.0)
        output[judge] = round(mean(agreements), 3) if agreements else 0.0
    return output


def _answer_order_bias(rows: list[dict[str, Any]], judges: list[str]) -> dict[str, float]:
    output = {}
    for judge in judges:
        diffs = []
        for row in rows:
            vote = _vote_for(row["votes"], judge)
            if vote is None:
                continue
            context = row["result"]["eval_item"]["context"]
            failed_pos = context.find("Failed model output")
            retrieved_pos = context.find("Retrieved doc")
            if failed_pos >= 0 and retrieved_pos >= 0:
                sign = 1 if failed_pos < retrieved_pos else -1
                diffs.append(sign * (float(vote["confidence"]) - 0.5))
        output[judge] = round(mean(diffs), 3) if diffs else 0.0
    return output


def _verbosity_bias(rows: list[dict[str, Any]], judges: list[str]) -> dict[str, float]:
    output = {}
    for judge in judges:
        samples = []
        for row in rows:
            vote = _vote_for(row["votes"], judge)
            if vote is not None:
                samples.append((len(str(vote.get("rationale", "")).split()), float(vote["confidence"])))
        output[judge] = round(_simple_correlation(samples), 3)
    return output


def _rubric_sensitivity(rows: list[dict[str, Any]], judges: list[str]) -> dict[str, float]:
    output = {}
    for judge in judges:
        by_family: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            vote = _vote_for(row["votes"], judge)
            if vote is not None:
                by_family[row["label"].family].append(float(vote["confidence"]))
        family_means = [mean(values) for values in by_family.values() if values]
        output[judge] = round(pstdev(family_means), 3) if len(family_means) > 1 else 0.0
    return output


def _needs_human_review(rows: list[dict[str, Any]], judges: list[str]) -> list[dict[str, Any]]:
    flagged = []
    for row in rows:
        votes = row["votes"]
        if not votes:
            continue
        pass_rate = mean(1.0 if vote["passed"] else 0.0 for vote in votes)
        avg_conf = mean(float(vote["confidence"]) for vote in votes)
        if pass_rate < 0.75 or avg_conf < 0.72:
            flagged.append(
                {
                    "trace_id": row["label"].trace_id,
                    "item_id": row["label"].item_id,
                    "family": row["label"].family,
                    "pass_rate": round(pass_rate, 3),
                    "average_confidence": round(avg_conf, 3),
                }
            )
    return flagged


def _simple_correlation(samples: list[tuple[float, float]]) -> float:
    if len(samples) < 2:
        return 0.0
    xs = [sample[0] for sample in samples]
    ys = [sample[1] for sample in samples]
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in samples)
    x_den = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    y_den = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    if not x_den or not y_den:
        return 0.0
    return numerator / (x_den * y_den)
