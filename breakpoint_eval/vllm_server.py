from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from breakpoint_eval.env import load_env


load_env()

app = FastAPI(
    title="BreakPoint vLLM-Compatible Judge Server",
    version="0.1.0",
    description="Local OpenAI-compatible judge endpoint for BreakPoint external-judge smoke runs.",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default_factory=lambda: os.environ.get("VLLM_JUDGE_MODEL", "breakpoint-local-judge"))
    messages: list[ChatMessage]
    temperature: float = 0
    max_tokens: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "breakpoint-vllm-judge"}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    model = os.environ.get("VLLM_JUDGE_MODEL", "breakpoint-local-judge")
    return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "breakpoint"}]}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_optional_auth(authorization)
    prompt = "\n".join(message.content for message in request.messages if message.role in {"user", "system"})
    verdict = judge_prompt(prompt)
    content = json.dumps(verdict, separators=(",", ":"))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": _rough_tokens(prompt),
            "completion_tokens": _rough_tokens(content),
            "total_tokens": _rough_tokens(prompt) + _rough_tokens(content),
        },
    }


def judge_prompt(prompt: str) -> dict[str, Any]:
    lower = prompt.lower()
    reasons: list[str] = []
    passed = True

    task = _section(prompt, "Task:", "Context:")
    context = _section(prompt, "Context:", "Expected:")
    expected = _section(prompt, "Expected:", "Hidden traps:")
    traps = _section(prompt, "Hidden traps:", None)

    if len(task.strip()) < 24:
        passed = False
        reasons.append("task too short")
    if len(context.strip()) < 80:
        passed = False
        reasons.append("context too short")
    if not expected.strip() or expected.strip().lower() in {"unknown", "maybe"}:
        passed = False
        reasons.append("weak expected answer")
    if "[]" in traps or "hidden traps: []" in lower or not traps.strip():
        passed = False
        reasons.append("missing hidden trap")
    if '"answer_type": "json"' in lower and "citations" not in expected.lower():
        passed = False
        reasons.append("json expected answer lacks citation field")

    confidence = 0.9 if passed else 0.18
    rationale = "passes local vLLM-compatible judge checks" if passed else "; ".join(reasons)
    return {"passed": passed, "confidence": confidence, "rationale": rationale}


def _section(text: str, start_marker: str, end_marker: str | None) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    if end_marker is None:
        return text[start:]
    end = text.find(end_marker, start)
    return text[start:] if end < 0 else text[start:end]


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _require_optional_auth(authorization: str | None) -> None:
    expected = os.environ.get("VLLM_JUDGE_API_KEY", "")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid VLLM judge API key")
