from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from breakpoint_eval.env import load_env
from breakpoint_eval.models import EvalItem, ModelVote, ValidationReport
from breakpoint_eval.validators import LocalJudge, ValidationSuite


load_env()

DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.4-mini"
DEFAULT_ANTHROPIC_JUDGE_MODEL = "claude-sonnet-5"
DEFAULT_GEMINI_JUDGE_MODEL = "gemini-2.5-flash"
DEFAULT_VLLM_JUDGE_MODEL = "breakpoint-local-judge"


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
    model: str = DEFAULT_OPENAI_JUDGE_MODEL
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
    model: str = DEFAULT_ANTHROPIC_JUDGE_MODEL
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
            "max_tokens": 1000,
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
    model: str = DEFAULT_GEMINI_JUDGE_MODEL
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
        payload = {
            "contents": [{"parts": [{"text": _judge_prompt(item)}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 500, "responseMimeType": "application/json"},
        }
        try:
            data = _post_json(url, payload, {})
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return _vote_from_json_text(self.name, text)
        except Exception as exc:
            return ModelVote(model_name=self.name, passed=False, confidence=0.0, rationale=f"adapter error: {exc}")


@dataclass(frozen=True)
class VLLMJudge:
    endpoint_env: str = "VLLM_JUDGE_ENDPOINT"
    model: str = DEFAULT_VLLM_JUDGE_MODEL
    api_key_env: str = "VLLM_JUDGE_API_KEY"

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
            headers = _bearer_header(os.environ.get(self.api_key_env, ""))
            data = _post_json(endpoint.rstrip("/") + "/v1/chat/completions", payload, headers)
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
    load_env()
    judges: list[JudgeAdapter] = [
        LocalHeuristicJudge("local:strict", 0.95),
        LocalHeuristicJudge("local:skeptic", 0.88),
        LocalHeuristicJudge("local:format", 0.82),
    ]
    if include_external:
        judges.extend(
            [
                OpenAIJudge(model=os.environ.get("OPENAI_JUDGE_MODEL", DEFAULT_OPENAI_JUDGE_MODEL)),
                AnthropicJudge(model=os.environ.get("ANTHROPIC_JUDGE_MODEL", DEFAULT_ANTHROPIC_JUDGE_MODEL)),
                GeminiJudge(model=os.environ.get("GEMINI_JUDGE_MODEL", DEFAULT_GEMINI_JUDGE_MODEL)),
                VLLMJudge(model=os.environ.get("VLLM_JUDGE_MODEL", DEFAULT_VLLM_JUDGE_MODEL)),
            ]
        )
    return judges


def available_validation_suite(include_external: bool = True) -> ValidationSuite:
    load_env()
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
        "You are grading a generated LLM evaluation item. Return compact valid JSON only, "
        "with exactly these keys: passed, confidence, rationale. confidence must be a number "
        "from 0 to 1. Keep rationale under 35 words. Determine whether the item has a clear "
        "task, answerable evidence, expected answer, hidden trap, and rubric.\n\n"
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


def _bearer_header(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


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
            confidence=_coerce_confidence(payload.get("confidence", 0.5)),
            rationale=str(payload.get("rationale", "")),
        )
    except Exception:
        loose_vote = _vote_from_loose_text(model_name, text)
        if loose_vote is not None:
            return loose_vote
        return ModelVote(model_name=model_name, passed=False, confidence=0.0, rationale=f"non-json judge output: {text[:180]}")


def _vote_from_loose_text(model_name: str, text: str) -> ModelVote | None:
    import re

    passed_match = re.search(r'"?passed"?\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    if not passed_match:
        return None
    confidence_match = re.search(r'"?confidence"?\s*:\s*("?[\w. %]+"?|\d+(?:\.\d+)?)', text, flags=re.IGNORECASE)
    rationale_match = re.search(r'"?rationale"?\s*:\s*"([^"]*)', text, flags=re.IGNORECASE)
    confidence: object = 0.5
    if confidence_match:
        confidence = confidence_match.group(1).strip('"')
    rationale = rationale_match.group(1) if rationale_match else f"parsed from loose judge output: {text[:120]}"
    return ModelVote(
        model_name=model_name,
        passed=passed_match.group(1).lower() == "true",
        confidence=_coerce_confidence(confidence),
        rationale=rationale,
    )


def _coerce_confidence(value: object) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        labels = {"very high": 0.95, "high": 0.85, "medium": 0.6, "moderate": 0.6, "low": 0.35, "very low": 0.15}
        if normalized in labels:
            return labels[normalized]
        if normalized.endswith("%"):
            try:
                return max(0.0, min(1.0, float(normalized.rstrip("%")) / 100))
            except ValueError:
                return 0.5
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _unavailable_vote(model_name: str, missing: str) -> ModelVote:
    return ModelVote(
        model_name=model_name,
        passed=False,
        confidence=0.0,
        rationale=f"adapter unavailable; set {missing} to enable",
    )
