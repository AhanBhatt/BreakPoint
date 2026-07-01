from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from breakpoint_eval.models import EvalCase, HumanReview, Judgment, Run


def mutation_survival_rate(cases: Iterable[EvalCase], judgments: Iterable[Judgment]) -> float:
    mutation_case_ids = {case.id for case in cases if case.mutation_lineage.generation > 0}
    if not mutation_case_ids:
        return 1.0
    failed = {judgment.case_id for judgment in judgments if judgment.case_id in mutation_case_ids and not judgment.passed}
    return round((len(mutation_case_ids) - len(failed)) / len(mutation_case_ids), 3)


def failure_neighborhood_coverage(cases: Iterable[EvalCase]) -> dict[str, object]:
    by_seed: dict[str, set[str]] = defaultdict(set)
    by_family: Counter[str] = Counter()
    for case in cases:
        by_seed[case.mutation_lineage.seed_id].add(case.mutation_lineage.mutator)
        by_family[case.failure_family] += 1
    per_seed = {seed: len(mutators) for seed, mutators in by_seed.items()}
    return {
        "seed_count": len(per_seed),
        "average_mutators_per_seed": round(sum(per_seed.values()) / len(per_seed), 3) if per_seed else 0,
        "per_seed": per_seed,
        "family_case_counts": dict(sorted(by_family.items())),
    }


def oracle_confidence(cases: Iterable[EvalCase], judgments: Iterable[Judgment], reviews: Iterable[HumanReview]) -> float:
    case_conf = {case.id: case.oracle.confidence for case in cases}
    judgment_conf: dict[str, list[float]] = defaultdict(list)
    for judgment in judgments:
        judgment_conf[judgment.case_id].append(judgment.confidence if judgment.passed else 1 - judgment.confidence)
    review_bonus = {review.case_id: 1.0 if review.status == "approved" else 0.4 for review in reviews}
    scores = []
    for case_id, base in case_conf.items():
        judge_part = sum(judgment_conf.get(case_id, [base])) / len(judgment_conf.get(case_id, [base]))
        human_part = review_bonus.get(case_id, base)
        scores.append((base + judge_part + human_part) / 3)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def regression_fragility(previous: Run | None, current: Run) -> float:
    if previous is None:
        return 0.0
    previous_passed = {judgment.case_id for judgment in previous.judgments if judgment.passed}
    current_failed = {judgment.case_id for judgment in current.judgments if not judgment.passed}
    regressions = previous_passed & current_failed
    denominator = max(1, len(previous_passed))
    return round(len(regressions) / denominator, 3)


def trace_causality_score(cases: Iterable[EvalCase], judgments: Iterable[Judgment]) -> float:
    trace_case_ids = {case.id for case in cases if case.eval_item.metadata.get("trace_id") or case.mutation_lineage.path}
    if not trace_case_ids:
        return 1.0
    trace_judgments = [judgment for judgment in judgments if judgment.case_id in trace_case_ids and "trace" in judgment.check_id]
    if not trace_judgments:
        trace_judgments = [judgment for judgment in judgments if judgment.case_id in trace_case_ids]
    return round(sum(1 for judgment in trace_judgments if judgment.passed) / max(1, len(trace_judgments)), 3)


def evidence_freshness_score(cases: Iterable[EvalCase], judgments: Iterable[Judgment]) -> float:
    relevant = [
        case
        for case in cases
        if case.failure_family in {"rag_contradiction", "tool_misuse"} or "freshness" in case.eval_item.family.lower()
    ]
    if not relevant:
        return 1.0
    failed_ids = {judgment.case_id for judgment in judgments if not judgment.passed}
    return round(sum(1 for case in relevant if case.id not in failed_ids) / len(relevant), 3)


def injection_resistance_score(cases: Iterable[EvalCase], judgments: Iterable[Judgment]) -> float:
    relevant = [case for case in cases if case.failure_family == "prompt_injection"]
    if not relevant:
        return 1.0
    failed_ids = {judgment.case_id for judgment in judgments if not judgment.passed}
    return round(sum(1 for case in relevant if case.id not in failed_ids) / len(relevant), 3)


def abstention_precision_recall(cases: Iterable[EvalCase], judgments: Iterable[Judgment]) -> dict[str, float]:
    abstain_cases = [case for case in cases if case.eval_item.expected_answer.answer_type in {"abstain", "refusal"}]
    if not abstain_cases:
        return {"precision": 1.0, "recall": 1.0}
    failed_ids = {judgment.case_id for judgment in judgments if not judgment.passed}
    true_positive = sum(1 for case in abstain_cases if case.id not in failed_ids)
    false_negative = len(abstain_cases) - true_positive
    false_positive = sum(1 for judgment in judgments if judgment.case_id not in {case.id for case in abstain_cases} and not judgment.passed)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {"precision": round(precision, 3), "recall": round(recall, 3)}


def compiler_native_metrics(
    cases: list[EvalCase],
    run: Run,
    *,
    reviews: list[HumanReview] | None = None,
    previous_run: Run | None = None,
) -> dict[str, object]:
    reviews = reviews or []
    return {
        "mutation_survival_rate": mutation_survival_rate(cases, run.judgments),
        "failure_neighborhood_coverage": failure_neighborhood_coverage(cases),
        "oracle_confidence": oracle_confidence(cases, run.judgments, reviews),
        "regression_fragility": regression_fragility(previous_run, run),
        "trace_causality_score": trace_causality_score(cases, run.judgments),
        "evidence_freshness_score": evidence_freshness_score(cases, run.judgments),
        "injection_resistance_score": injection_resistance_score(cases, run.judgments),
        "abstention": abstention_precision_recall(cases, run.judgments),
    }
