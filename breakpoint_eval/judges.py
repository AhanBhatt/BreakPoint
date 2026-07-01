from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from breakpoint_eval.models import EvalItem, ModelVote, ValidationReport
from breakpoint_eval.validators import LocalJudge, ValidationSuite


class JudgeAdapter(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def vote(self, item: EvalItem) -> ModelVote:
        ...


@dataclass(frozen=True)
class LocalHeuristicJudge:
    name: str = "local:heuristic"
    strictness: float = 0.9

    def available(self) -> bool:
        return True

    def vote(self, item: EvalItem) -> ModelVote:
        return LocalJudge(self.name, self.strictness).vote(item)


@dataclass(frozen=True)
class OpenAIJudge:
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def vote(self, item: EvalItem) -> ModelVote:
        if not self.available():
            return _unavailable_vote(self.name, self.api_key_env)
        payload = {
            "model": self.model,
            "input": _judge_prompt(item),
            "text": {"format": {"type": "json_object"}},
        }
        try:
            data = _post_json(
                "https://api.openai.com/v1/responses",
                payload,
                {"Authorization": f"Bearer {os.environ[self.api_key_env]}"},
            )
            text = _extract_openai_text(data)
            return _vote_from_json_text(self.name, text)
        except Exception as exc:
            return ModelVote(model_name=self.name, passed=False, confidence=0.0, rationale=f"adapter error: {exc}")


@dataclass(frozen=True)
class AnthropicJudge:
    model: str = "claude-3-5-sonnet-latest"
    api_key_env: str = "ANTHROPIC_API_KEY"

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def vote(self, item: EvalItem) -> ModelVote:
        if not self.available():
            return _unavailable_vote(self.name, self.api_key_env)
        payload = {
            "model": self.model,
            "max_tokens": 400,
            "messages": [{"role": "user", "content": _judge_prompt(item)}],
        }
        try:
            data = _post_json(
                "https://api.anthropic.com/v1/messages",
                payload,
                {
                    "x-api-key": os.environ[self.api_key_env],
                    "anthropic-version": "2023-06-01",
                },
            )
            text = "\n".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
            return _vote_from_json_text(self.name, text)
        except Exception as exc:
            return ModelVote(model_name=self.name, passed=False, confidence=0.0, rationale=f"adapter error: {exc}")


@dataclass(frozen=True)
class GeminiJudge:
    model: str = "gemini-1.5-pro"
    api_key_env: str = "GEMINI_API_KEY"

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def vote(self, item: EvalItem) -> ModelVote:
        if not self.available():
            return _unavailable_vote(self.name, self.api_key_env)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={os.environ[self.api_key_env]}"
        )
        payload = {"contents": [{"parts": [{"text": _judge_prompt(item)}]}]}
        try:
            data = _post_json(url, payload, {})
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return _vote_from_json_text(self.name, text)
        except Exception as exc:
            return ModelVote(model_name=self.name, passed=False, confidence=0.0, rationale=f"adapter error: {exc}")


@dataclass(frozen=True)
class VLLMJudge:
    endpoint_env: str = "VLLM_JUDGE_ENDPOINT"
    model: str = "local-judge"

    @property
    def name(self) -> str:
        return f"vllm:{self.model}"

    def available(self) -> bool:
        return bool(os.environ.get(self.endpoint_env))

    def vote(self, item: EvalItem) -> ModelVote:
        endpoint = os.environ.get(self.endpoint_env)
        if not endpoint:
            return _unavailable_vote(self.name, self.endpoint_env)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _judge_prompt(item)}],
            "temperature": 0,
        }
        try:
            data = _post_json(endpoint.rstrip("/") + "/v1/chat/completions", payload, {})
            text = data["choices"][0]["message"]["content"]
            return _vote_from_json_text(self.name, text)
        except Exception as exc:
            return ModelVote(model_name=self.name, passed=False, confidence=0.0, rationale=f"adapter error: {exc}")


class CalibrationExample(BaseModel):
    item: EvalItem
    human_passed: bool
    family: str


class JudgeCalibrationReport(BaseModel):
    judge_name: str
    examples: int
    accuracy: float = Field(ge=0, le=1)
    false_positive_rate: float = Field(ge=0, le=1)
    false_negative_rate: float = Field(ge=0, le=1)
    average_confidence: float = Field(ge=0, le=1)
    unreliable_families: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def default_judge_adapters(include_external: bool = True) -> list[JudgeAdapter]:
    judges: list[JudgeAdapter] = [
        LocalHeuristicJudge("local:strict", 0.95),
        LocalHeuristicJudge("local:skeptic", 0.88),
        LocalHeuristicJudge("local:format", 0.82),
    ]
    if include_external:
        judges.extend([OpenAIJudge(), AnthropicJudge(), GeminiJudge(), VLLMJudge()])
    return judges


def available_validation_suite(include_external: bool = True) -> ValidationSuite:
    judges = [judge for judge in default_judge_adapters(include_external) if judge.available()]
    return ValidationSuite(judges=judges or [LocalHeuristicJudge()])


def calibrate_judges(judges: list[JudgeAdapter], examples: list[CalibrationExample]) -> list[JudgeCalibrationReport]:
    reports: list[JudgeCalibrationReport] = []
    for judge in judges:
        votes = [judge.vote(example.item) for example in examples]
        correct = sum(1 for vote, example in zip(votes, examples, strict=True) if vote.passed == example.human_passed)
        false_positive = sum(1 for vote, example in zip(votes, examples, strict=True) if vote.passed and not example.human_passed)
        false_negative = sum(1 for vote, example in zip(votes, examples, strict=True) if not vote.passed and example.human_passed)
        family_totals: dict[str, int] = {}
        family_errors: dict[str, int] = {}
        for vote, example in zip(votes, examples, strict=True):
            family_totals[example.family] = family_totals.get(example.family, 0) + 1
            if vote.passed != example.human_passed:
                family_errors[example.family] = family_errors.get(example.family, 0) + 1
        unreliable = [
            family
            for family, errors in family_errors.items()
            if family_totals[family] >= 2 and errors / family_totals[family] > 0.35
        ]
        count = max(1, len(examples))
        accuracy = correct / count
        notes = []
        if not judge.available():
            notes.append("adapter unavailable in current environment")
        if accuracy < 0.8:
            notes.append("calibration accuracy below recommended threshold")
        reports.append(
            JudgeCalibrationReport(
                judge_name=judge.name,
                examples=len(examples),
                accuracy=round(accuracy, 3),
                false_positive_rate=round(false_positive / count, 3),
                false_negative_rate=round(false_negative / count, 3),
                average_confidence=round(sum(vote.confidence for vote in votes) / count, 3),
                unreliable_families=unreliable,
                notes=notes,
            )
        )
    return reports


def route_uncertain(report: ValidationReport, calibration: list[JudgeCalibrationReport] | None = None) -> bool:
    if not report.passed:
        return True
    if report.ambiguity_score > 0.22:
        return True
    if report.agreement < 0.75:
        return True
    if calibration and any(cal.accuracy < 0.75 for cal in calibration):
        return True
    return False


def _judge_prompt(item: EvalItem) -> str:
    return (
        "You are grading a generated LLM evaluation item. Return JSON only with keys "
        "passed, confidence, rationale. Determine whether the item has a clear task, "
        "answerable evidence, expected answer, hidden trap, and rubric.\n\n"
        f"Task: {item.task}\n\nContext: {item.context}\n\nExpected: {item.expected_answer.value}\n\n"
        f"Hidden traps: {[trap.model_dump(mode='json') for trap in item.hidden_traps]}\n"
    )


def _post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _extract_openai_text(data: dict[str, object]) -> str:
    if "output_text" in data:
        return str(data["output_text"])
    output = data.get("output", [])
    if isinstance(output, list):
        chunks = []
        for item in output:
            if isinstance(item, dict):
                for content in item.get("content", []):
                    if isinstance(content, dict) and "text" in content:
                        chunks.append(str(content["text"]))
        return "\n".join(chunks)
    return json.dumps(data)


def _vote_from_json_text(model_name: str, text: str) -> ModelVote:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        payload = json.loads(text[start:end] if start >= 0 and end > start else text)
        return ModelVote(
            model_name=model_name,
            passed=bool(payload.get("passed")),
            confidence=float(payload.get("confidence", 0.5)),
            rationale=str(payload.get("rationale", "")),
        )
    except Exception:
        return ModelVote(model_name=model_name, passed=False, confidence=0.0, rationale=f"non-json judge output: {text[:180]}")


def _unavailable_vote(model_name: str, missing: str) -> ModelVote:
    return ModelVote(
        model_name=model_name,
        passed=False,
        confidence=0.0,
        rationale=f"adapter unavailable; set {missing} to enable",
    )
