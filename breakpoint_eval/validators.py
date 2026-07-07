from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from breakpoint_eval.models import EvalItem, ModelVote, ValidationReport


@dataclass(frozen=True)
class LocalJudge:
    """A deterministic stand-in for model judges used in offline demos and tests."""

    name: str
    strictness: float

    def vote(self, item: EvalItem) -> ModelVote:
        reasons: list[str] = []
        passed = True

        if len(item.task.strip()) < 24 or len(item.context.strip()) < 80:
            passed = False
            reasons.append("task/context too short")

        if not item.expected_answer.value.strip() or item.expected_answer.value.lower() in {"unknown", "maybe"}:
            passed = False
            reasons.append("weak expected answer")

        if not item.hidden_traps:
            passed = False
            reasons.append("missing hidden trap")

        if item.expected_answer.answer_type == "json":
            try:
                parsed = json.loads(item.expected_answer.value)
                if not isinstance(parsed, dict):
                    passed = False
                    reasons.append("expected JSON is not an object")
            except json.JSONDecodeError:
                passed = False
                reasons.append("expected JSON is invalid")

        if item.metadata.get("quality_probe") == "ambiguous":
            passed = False
            reasons.append("seeded ambiguous candidate rejected")

        confidence = 0.91 if passed else 0.16 + (0.08 * self.strictness)
        rationale = "passes local judge checks" if passed else "; ".join(reasons)
        return ModelVote(model_name=self.name, passed=passed, confidence=round(confidence, 3), rationale=rationale)


class ValidationSuite:
    def __init__(self, judges: list[LocalJudge] | None = None) -> None:
        self.judges = judges or [
            LocalJudge("surrogate:gpt-5-strict", 0.95),
            LocalJudge("surrogate:claude-opus-skeptic", 0.9),
            LocalJudge("surrogate:gemini-pro-format", 0.85),
        ]

    def validate(self, item: EvalItem) -> ValidationReport:
        votes = self._vote_all(item)
        ambiguity_score = self._ambiguity_score(item)
        answerability_score = self._answerability_score(item)
        trap_coverage_score = self._trap_coverage(item)
        format_score = self._format_score(item)

        passed = (
            sum(1 for vote in votes if vote.passed) >= 2
            and ambiguity_score <= 0.32
            and answerability_score >= 0.72
            and trap_coverage_score >= 0.65
            and format_score >= 0.8
        )
        reasons: list[str] = []
        if not passed:
            reasons.extend(vote.rationale for vote in votes if not vote.passed)
            if ambiguity_score > 0.32:
                reasons.append("ambiguity score above threshold")
            if answerability_score < 0.72:
                reasons.append("answerability score below threshold")
            if trap_coverage_score < 0.65:
                reasons.append("trap coverage below threshold")
            if format_score < 0.8:
                reasons.append("format score below threshold")

        return ValidationReport(
            item_id=item.id,
            votes=votes,
            ambiguity_score=round(ambiguity_score, 3),
            answerability_score=round(answerability_score, 3),
            trap_coverage_score=round(trap_coverage_score, 3),
            format_score=round(format_score, 3),
            passed=passed,
            reasons=sorted(set(reasons)),
        )

    def _vote_all(self, item: EvalItem) -> list[ModelVote]:
        if len(self.judges) <= 1:
            return [self.judges[0].vote(item)]
        max_workers = max(1, min(len(self.judges), int(os.environ.get("BREAKPOINT_JUDGE_PARALLELISM", "10"))))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda judge: judge.vote(item), self.judges))

    @staticmethod
    def _ambiguity_score(item: EvalItem) -> float:
        score = 0.08
        vague_markers = ["maybe", "several", "various", "probably", "roughly"]
        text = f"{item.task} {item.expected_answer.value}".lower()
        score += sum(0.08 for marker in vague_markers if marker in text)
        if item.metadata.get("quality_probe") == "ambiguous":
            score += 0.5
        numeric_candidates = re.findall(r"\b\d{2,4}\b", item.context)
        if len(set(numeric_candidates)) > 8 and item.expected_answer.answer_type != "json":
            score += 0.05
        return min(score, 1.0)

    @staticmethod
    def _answerability_score(item: EvalItem) -> float:
        answer = item.expected_answer.value.lower()
        context = item.context.lower()
        if item.expected_answer.answer_type in {"refusal", "abstain", "json", "tool_decision", "reasoned"}:
            return 0.9
        includes = item.expected_answer.must_include or [answer[:32]]
        coverage = sum(1 for token in includes if token.lower() in context or token.lower() in answer)
        return min(1.0, 0.55 + (coverage / max(1, len(includes))) * 0.42)

    @staticmethod
    def _trap_coverage(item: EvalItem) -> float:
        if not item.hidden_traps:
            return 0.0
        prompt = item.prompt.lower()
        covered = 0
        for trap in item.hidden_traps:
            trigger = trap.trigger.lower()
            if trigger and (trigger in prompt or any(piece in prompt for piece in trigger.split()[:3])):
                covered += 1
        return min(1.0, covered / len(item.hidden_traps))

    @staticmethod
    def _format_score(item: EvalItem) -> float:
        if item.expected_answer.answer_type != "json":
            return 1.0
        try:
            parsed = json.loads(item.expected_answer.value)
        except json.JSONDecodeError:
            return 0.0
        required_keys = {"answer", "confidence", "citations"}
        return 1.0 if required_keys.issubset(parsed) else 0.55
