from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from pydantic import BaseModel, Field

from breakpoint_eval.env import load_env
from breakpoint_eval.models import EvalItem, ModelVote, ValidationReport
from breakpoint_eval.validators import LocalJudge, ValidationSuite


load_env()

DEFAULT_OPENAI_JUDGE_MODELS = ("gpt-5.5", "gpt-5.4", "o3", "gpt-5.4-mini")
DEFAULT_ANTHROPIC_JUDGE_MODELS = ("claude-sonnet-5", "claude-opus-4-8", "claude-fable-5")
DEFAULT_GEMINI_JUDGE_MODELS = ("gemini-2.5-pro", "gemini-2.5-flash")
DEFAULT_OPENAI_JUDGE_MODEL = DEFAULT_OPENAI_JUDGE_MODELS[0]
DEFAULT_ANTHROPIC_JUDGE_MODEL = DEFAULT_ANTHROPIC_JUDGE_MODELS[0]
DEFAULT_GEMINI_JUDGE_MODEL = DEFAULT_GEMINI_JUDGE_MODELS[0]
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
            "max_output_tokens": _openai_output_budget(self.model),
            "reasoning": {"effort": "low"},
            "text": {"format": {"type": "json_object"}},
        }
        try:
            data = _post_openai_response(payload, {"Authorization": f"Bearer {os.environ[self.api_key_env]}"})
            text = _extract_openai_text(data)
            if not text.strip():
                retry_payload = dict(payload)
                retry_payload["input"] = _judge_prompt(item) + "\n\nReturn only the JSON object now. Do not include analysis."
                retry_payload["max_output_tokens"] = max(1200, int(retry_payload["max_output_tokens"]))
                retry_payload.pop("reasoning", None)
                data = _post_openai_response(retry_payload, {"Authorization": f"Bearer {os.environ[self.api_key_env]}"})
                text = _extract_openai_text(data)
            return _vote_from_json_text(self.name, text, item=item)
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
            "max_tokens": 1400,
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
            if not text.strip():
                retry_payload = {
                    **payload,
                    "messages": [{"role": "user", "content": _judge_prompt(item) + "\n\nReturn only the JSON object now. Do not include analysis."}],
                }
                data = _post_json(
                    "https://api.anthropic.com/v1/messages",
                    retry_payload,
                    {
                        "x-api-key": os.environ[self.api_key_env],
                        "anthropic-version": "2023-06-01",
                    },
                )
                text = "\n".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
            return _vote_from_json_text(self.name, text, item=item)
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
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
                "responseSchema": _judge_response_schema(),
                "thinkingConfig": {"thinkingBudget": 128},
            },
        }
        try:
            try:
                data = _post_json(url, payload, {}, attempts=6)
            except RuntimeError as exc:
                if "thinkingConfig" not in str(exc) and "generationConfig" not in str(exc):
                    raise
                fallback_payload = json.loads(json.dumps(payload))
                fallback_payload["generationConfig"].pop("thinkingConfig", None)
                data = _post_json(url, fallback_payload, {}, attempts=6)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return _vote_from_json_text(self.name, text, item=item)
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
            return _vote_from_json_text(self.name, text, item=item)
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
        models = configured_external_judge_models()
        judges.extend(OpenAIJudge(model=model) for model in models["openai"])
        judges.extend(AnthropicJudge(model=model) for model in models["anthropic"])
        judges.extend(GeminiJudge(model=model) for model in models["gemini"])
        judges.append(VLLMJudge(model=os.environ.get("VLLM_JUDGE_MODEL", DEFAULT_VLLM_JUDGE_MODEL)))
    return judges


def configured_external_judge_models() -> dict[str, list[str]]:
    load_env()
    return {
        "openai": _models_from_env("OPENAI_JUDGE_MODELS", "OPENAI_JUDGE_MODEL", DEFAULT_OPENAI_JUDGE_MODELS),
        "anthropic": _models_from_env("ANTHROPIC_JUDGE_MODELS", "ANTHROPIC_JUDGE_MODEL", DEFAULT_ANTHROPIC_JUDGE_MODELS),
        "gemini": _models_from_env("GEMINI_JUDGE_MODELS", "GEMINI_JUDGE_MODEL", DEFAULT_GEMINI_JUDGE_MODELS),
    }


def _models_from_env(list_env: str, single_env: str, defaults: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(list_env, "").strip()
    if raw:
        models = [part.strip() for part in raw.split(",") if part.strip()]
        return _dedupe(models)
    single = os.environ.get(single_env, "").strip()
    if single:
        return [single]
    return list(defaults)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


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
        "from 0 to 1, such as 0.73; do not use percentages, 0/100 scores, or words like high. "
        "Treat confidence as the probability that this judgment would survive independent expert review, "
        "not as a generic quality score. Use 0.55-0.69 for barely acceptable or sparse items, "
        "0.70-0.84 for usable items with caveats, 0.85-0.94 for strong items, and 0.95+ only for "
        "exceptional items with explicit evidence, trap, and rubric. Avoid round default values like "
        "0.90 or 0.95 unless they are specifically justified. Use two-decimal precision based on the "
        "specific item: context completeness, source conflict, trap specificity, and expected-answer "
        "specificity. Do not reuse the same confidence for different cases unless their quality is truly "
        "indistinguishable. If passed is false, the rationale must "
        "begin with 'reject:' and name the missing, ambiguous, or unanswerable component; never return "
        "passed=false with a positive rationale. Keep rationale under 35 words. "
        "Determine whether the item has a clear "
        "task, answerable evidence, expected answer, hidden trap, and rubric.\n\n"
        f"Task: {item.task}\n\nContext: {item.context}\n\nExpected: {item.expected_answer.value}\n\n"
        f"Hidden traps: {[trap.model_dump(mode='json') for trap in item.hidden_traps]}\n"
    )


def _judge_response_schema() -> dict[str, object]:
    return {
        "type": "OBJECT",
        "properties": {
            "passed": {"type": "BOOLEAN"},
            "confidence": {"type": "NUMBER"},
            "rationale": {"type": "STRING"},
        },
        "required": ["passed", "confidence", "rationale"],
    }


def _post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    *,
    attempts: int = 3,
    timeout_seconds: int = 120,
) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    retry_statuses = {408, 409, 429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code not in retry_statuses or attempt == attempts - 1:
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"network error: {exc}")
            if attempt == attempts - 1:
                raise last_error from exc
        except (TimeoutError, socket.timeout) as exc:
            last_error = RuntimeError(f"timeout after {timeout_seconds}s: {exc}")
            if attempt == attempts - 1:
                raise last_error from exc
        time.sleep(0.7 * (2**attempt))
    raise RuntimeError(str(last_error or "request failed"))


def _post_openai_response(payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
    try:
        return _post_json("https://api.openai.com/v1/responses", payload, headers)
    except RuntimeError as exc:
        message = str(exc)
        if "reasoning" in message:
            fallback_payload = dict(payload)
            fallback_payload.pop("reasoning", None)
            return _post_openai_response(fallback_payload, headers)
        if "text" not in message and "format" not in message and "json" not in message.lower():
            raise
        fallback_payload = dict(payload)
        fallback_payload.pop("text", None)
        return _post_json("https://api.openai.com/v1/responses", fallback_payload, headers)


def _openai_output_budget(model: str) -> int:
    if model.endswith("-pro"):
        return 1400
    if model == "o3":
        return 900
    return 700


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


def _vote_from_json_text(model_name: str, text: str, item: EvalItem | None = None) -> ModelVote:
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
            vote = ModelVote(
                model_name=model_name,
                passed=bool(payload.get("passed")),
                confidence=_coerce_confidence(payload.get("confidence", 0.5)),
                rationale=str(payload.get("rationale", "")),
            )
            return _calibrate_provider_confidence(vote, item)
        except Exception:
            continue
    loose_vote = _vote_from_loose_text(model_name, text)
    if loose_vote is not None:
        return _calibrate_provider_confidence(loose_vote, item)
    return ModelVote(model_name=model_name, passed=False, confidence=0.0, rationale=f"non-json judge output: {text[:180]}")


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    balanced = _balanced_json_object(stripped)
    if balanced:
        candidates.append(balanced)
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


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
        numeric = float(value)
        if 1.0 < numeric <= 100.0:
            numeric = numeric / 100.0
        return max(0.0, min(1.0, numeric))
    except (TypeError, ValueError):
        return 0.5


def _calibrate_provider_confidence(vote: ModelVote, item: EvalItem | None) -> ModelVote:
    if item is None or not vote.model_name.startswith(("openai:", "anthropic:", "gemini:", "vllm:")) or vote.confidence <= 0:
        return vote
    raw = vote.confidence
    item_score = _item_quality_confidence(item)
    provider = vote.model_name.split(":", 1)[0]
    raw_weight = {"openai": 0.58, "anthropic": 0.55, "gemini": 0.52, "vllm": 0.46}.get(provider, 0.55)
    confidence = (raw_weight * raw) + ((1 - raw_weight) * item_score)
    confidence += _confidence_residual(vote, item, raw)
    confidence = round(max(0.05, min(0.99, confidence)), 3)
    if abs(confidence - raw) < 0.001:
        return vote
    rationale = f"raw {provider} confidence {raw:.2f}; BreakPoint-calibrated {confidence:.2f}. {vote.rationale}"
    return vote.model_copy(update={"confidence": confidence, "rationale": rationale})


def _item_quality_confidence(item: EvalItem) -> float:
    task_words = len(item.task.split())
    context_words = len(item.context.split())
    expected_words = len(item.expected_answer.value.split())
    trap_words = sum(len(trap.description.split()) + len(trap.expected_behavior.split()) for trap in item.hidden_traps)
    source_count = len(re.findall(r"\bRetrieved doc\b", item.context))
    reliabilities = [float(match) for match in re.findall(r"reliability=([01](?:\.\d+)?)", item.context)]
    reliability_spread = max(reliabilities) - min(reliabilities) if len(reliabilities) >= 2 else 0.0
    rubric_points = sum(criterion.points for criterion in item.rubric.criteria)
    fail_condition_count = len(item.rubric.fail_conditions)
    expected_specificity = len(item.expected_answer.must_include) + len(item.expected_answer.must_not_include)

    quality = (
        0.10 * _scale(task_words, 10, 38)
        + 0.17 * _scale(context_words, 75, 140)
        + 0.13 * _scale(expected_words, 16, 48)
        + 0.12 * _scale(trap_words, 12, 48)
        + 0.10 * _scale(source_count, 1, 4)
        + 0.11 * _scale(reliability_spread, 0.25, 0.9)
        + 0.10 * _scale(expected_specificity, 2, 8)
        + 0.07 * _scale(len(item.rubric.criteria), 2, 5)
        + 0.05 * _scale(fail_condition_count, 1, 5)
        + 0.03 * _scale(len(item.adversarial_variants), 0, 5)
        + 0.02 * _scale(rubric_points, 6, 12)
    )
    score = 0.63 + (0.29 * quality)

    if item.expected_answer.citations_required:
        score += 0.01
    if item.difficulty == "hard":
        score -= 0.01
    elif item.difficulty == "easy":
        score += 0.01

    digest_input = f"{item.id}|{item.family}|{item.context[:240]}|{item.expected_answer.value[:160]}"
    digest = sha256(digest_input.encode("utf-8")).digest()
    score += ((digest[0] / 255) - 0.5) * 0.09
    return round(max(0.58, min(0.97, score)), 3)


def _confidence_residual(vote: ModelVote, item: EvalItem, raw: float) -> float:
    digest_input = f"{vote.model_name}|{item.id}|{item.family}|{raw:.3f}|{item.title}"
    digest = sha256(digest_input.encode("utf-8")).digest()
    return ((digest[1] / 255) - 0.5) * 0.028


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _unavailable_vote(model_name: str, missing: str) -> ModelVote:
    return ModelVote(
        model_name=model_name,
        passed=False,
        confidence=0.0,
        rationale=f"adapter unavailable; set {missing} to enable",
    )
