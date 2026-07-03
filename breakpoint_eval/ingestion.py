from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from breakpoint_eval.platform import stable_id
from breakpoint_eval.traces import (
    RawFailureTrace,
    RetrievedDocumentTrace,
    ToolCallTrace,
    TraceSource,
    normalize_trace,
    redact_trace,
)


IngestionKind = Literal[
    "opentelemetry",
    "langsmith",
    "openinference",
    "litellm",
    "retrieval_log",
    "support_ticket",
    "user_feedback",
    "red_team",
    "incident_report",
]


class RedactionPolicy(BaseModel):
    name: str
    redact_pii: bool = True
    redact_secrets: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionReport(BaseModel):
    source: IngestionKind
    input_path: str | None = None
    traces: list[RawFailureTrace]
    trace_count: int
    redaction_policy: RedactionPolicy
    provenance: dict[str, Any] = Field(default_factory=dict)


def load_production_traces(
    path: str | Path,
    *,
    source: IngestionKind | None = None,
    redaction_policy: RedactionPolicy | None = None,
) -> IngestionReport:
    target = Path(path)
    payload = _read_payload(target)
    inferred_source = source or _infer_ingestion_kind(payload, target)
    report = ingest_payload(payload, source=inferred_source, redaction_policy=redaction_policy)
    report.input_path = str(target)
    return report


def ingest_payload(
    payload: Any,
    *,
    source: IngestionKind,
    redaction_policy: RedactionPolicy | None = None,
) -> IngestionReport:
    policy = redaction_policy or RedactionPolicy(name=f"{source}-default")
    adapter = {
        "opentelemetry": _from_span_payload,
        "openinference": _from_span_payload,
        "langsmith": _from_langsmith_payload,
        "litellm": _from_litellm_payload,
        "retrieval_log": _from_retrieval_payload,
        "support_ticket": _from_support_ticket_payload,
        "user_feedback": _from_user_feedback_payload,
        "red_team": _from_red_team_payload,
        "incident_report": _from_incident_payload,
    }[source]
    traces = [redact_trace(trace) if policy.redact_pii or policy.redact_secrets else trace for trace in adapter(payload, source)]
    return IngestionReport(
        source=source,
        traces=traces,
        trace_count=len(traces),
        redaction_policy=policy,
        provenance={
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "schema": "breakpoint.production_ingestion.v1",
        },
    )


def write_ingestion_report(report: IngestionReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def traces_to_json(traces: list[RawFailureTrace], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"traces": [trace.model_dump(mode="json") for trace in traces]}, indent=2), encoding="utf-8")


def _read_payload(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_ingestion_kind(payload: Any, path: Path) -> IngestionKind:
    text = json.dumps(payload if not isinstance(payload, list) else payload[:2]).lower()
    name = path.name.lower()
    if "langsmith" in text or "run_type" in text:
        return "langsmith"
    if "openinference" in text:
        return "openinference"
    if "trace_id" in text and ("span" in text or "attributes" in text):
        return "opentelemetry"
    if "litellm" in text or "model_response" in text:
        return "litellm"
    if "support" in name or "ticket" in text:
        return "support_ticket"
    if "thumb" in text or "rating" in text or "feedback" in text:
        return "user_feedback"
    if "transcript" in text or "red_team" in text:
        return "red_team"
    if "retrieved_docs" in text or "retrieval" in text:
        return "retrieval_log"
    return "incident_report"


def _records(payload: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    return [payload]


def _from_span_payload(payload: Any, source: IngestionKind) -> list[RawFailureTrace]:
    traces = []
    for span in _records(payload, "spans", "data", "resourceSpans"):
        attrs = span.get("attributes") or span.get("attrs") or span
        user_task = _first(attrs, "gen_ai.prompt", "llm.input", "input.value", "prompt", "user_task") or _messages_to_text(attrs.get("messages"))
        model_output = _first(attrs, "gen_ai.completion", "llm.output", "output.value", "response", "model_output")
        docs = _docs_from_any(_first_obj(attrs, "retrieval.documents", "retrieved_docs", "documents", "retrieval"))
        tools = _tools_from_any(_first_obj(attrs, "tool.calls", "tool_calls", "tools"))
        traces.append(
            _trace(
                source="opentelemetry" if source == "opentelemetry" else "openinference",
                record=span,
                user_task=user_task,
                model_output=model_output,
                expected=_first(attrs, "expected", "expected_behavior", "ground_truth"),
                summary=_first(attrs, "failure.summary", "feedback", "error.message", "summary"),
                docs=docs,
                tools=tools,
            )
        )
    return traces


def _from_langsmith_payload(payload: Any, source: IngestionKind) -> list[RawFailureTrace]:
    traces = []
    for run in _records(payload, "runs", "data"):
        inputs = run.get("inputs") or {}
        outputs = run.get("outputs") or {}
        extras = run.get("extra") or {}
        traces.append(
            _trace(
                source="langsmith",
                record=run,
                user_task=_first(inputs, "input", "question", "prompt") or _first(run, "input"),
                model_output=_first(outputs, "output", "answer", "response") or _first(run, "output"),
                expected=_first(run, "expected", "reference") or _first(extras, "expected"),
                summary=_first(run, "feedback", "error") or _first(extras, "failure_summary"),
                docs=_docs_from_any(extras.get("retrieved_docs") or run.get("retrieved_docs")),
                tools=_tools_from_any(extras.get("tool_calls") or run.get("tool_calls")),
            )
        )
    return traces


def _from_litellm_payload(payload: Any, source: IngestionKind) -> list[RawFailureTrace]:
    traces = []
    for record in _records(payload, "logs", "events", "data"):
        messages = record.get("messages") or record.get("input") or []
        response = record.get("response") or record.get("model_response") or record.get("output")
        traces.append(
            _trace(
                source="litellm",
                record=record,
                user_task=_messages_to_text(messages) or _first(record, "prompt", "user_task"),
                model_output=_response_to_text(response),
                expected=_first(record, "expected", "ground_truth"),
                summary=_first(record, "feedback", "failure_summary", "error"),
                docs=_docs_from_any(record.get("retrieved_docs") or record.get("retrieval")),
                tools=_tools_from_any(record.get("tool_calls") or record.get("tools")),
            )
        )
    return traces


def _from_retrieval_payload(payload: Any, source: IngestionKind) -> list[RawFailureTrace]:
    traces = []
    for record in _records(payload, "queries", "records", "traces"):
        traces.append(
            _trace(
                source="rag_log",
                record=record,
                user_task=_first(record, "query", "question", "user_task", "prompt"),
                model_output=_first(record, "answer", "model_output", "response"),
                expected=_first(record, "expected", "ground_truth", "expected_behavior"),
                summary=_first(record, "failure_summary", "feedback", "summary"),
                docs=_docs_from_any(record.get("retrieved_docs") or record.get("documents") or record.get("chunks")),
            )
        )
    return traces


def _from_support_ticket_payload(payload: Any, source: IngestionKind) -> list[RawFailureTrace]:
    traces = []
    for ticket in _records(payload, "tickets", "records"):
        subject = _first(ticket, "subject", "title") or "Support ticket"
        body = _first(ticket, "body", "description", "customer_message") or ""
        traces.append(
            _trace(
                source="support_ticket",
                record=ticket,
                user_task=f"{subject}\n\n{body}".strip(),
                model_output=_first(ticket, "ai_response", "model_output", "response", "resolution"),
                expected=_first(ticket, "correct_resolution", "expected_behavior", "expected"),
                summary=_first(ticket, "failure_summary", "feedback", "review_notes") or f"Support ticket failure: {subject}",
                docs=_docs_from_any(ticket.get("retrieved_docs") or ticket.get("knowledge_base")),
                tools=_tools_from_any(ticket.get("tool_calls")),
            )
        )
    return traces


def _from_user_feedback_payload(payload: Any, source: IngestionKind) -> list[RawFailureTrace]:
    traces = []
    for feedback in _records(payload, "feedback", "records"):
        traces.append(
            _trace(
                source="user_feedback",
                record=feedback,
                user_task=_first(feedback, "prompt", "input", "user_task", "question"),
                model_output=_first(feedback, "response", "model_output", "answer"),
                expected=_first(feedback, "expected", "expected_behavior", "correct_answer"),
                summary=_first(feedback, "comment", "feedback", "reason") or "Negative user feedback on model response.",
                docs=_docs_from_any(feedback.get("retrieved_docs") or feedback.get("documents")),
                tools=_tools_from_any(feedback.get("tool_calls")),
            )
        )
    return traces


def _from_red_team_payload(payload: Any, source: IngestionKind) -> list[RawFailureTrace]:
    traces = []
    for transcript in _records(payload, "transcripts", "records"):
        turns = transcript.get("turns") or transcript.get("messages") or []
        user_turns = [turn.get("content", "") for turn in turns if isinstance(turn, dict) and turn.get("role") == "user"]
        assistant_turns = [turn.get("content", "") for turn in turns if isinstance(turn, dict) and turn.get("role") == "assistant"]
        traces.append(
            _trace(
                source="red_team",
                record=transcript,
                user_task=user_turns[0] if user_turns else _first(transcript, "prompt", "user_task"),
                model_output=assistant_turns[-1] if assistant_turns else _first(transcript, "model_output", "response"),
                expected=_first(transcript, "expected_behavior", "safe_behavior")
                or "Resist the attack, preserve instruction hierarchy, and avoid unsafe or policy-violating output.",
                summary=_first(transcript, "failure_summary", "finding", "summary") or "Red-team transcript exposed unsafe behavior.",
                docs=_docs_from_any(transcript.get("retrieved_docs")),
                tools=_tools_from_any(transcript.get("tool_calls")),
            )
        )
    return traces


def _from_incident_payload(payload: Any, source: IngestionKind) -> list[RawFailureTrace]:
    return [normalize_trace(record) for record in _records(payload, "incidents", "traces", "records")]


def _trace(
    *,
    source: TraceSource,
    record: dict[str, Any],
    user_task: str | None = None,
    model_output: str | None = None,
    expected: str | None = None,
    summary: str | None = None,
    docs: list[RetrievedDocumentTrace] | None = None,
    tools: list[ToolCallTrace] | None = None,
) -> RawFailureTrace:
    record_id = str(record.get("id") or record.get("run_id") or record.get("trace_id") or stable_id("record", record))
    severity = record.get("incident_severity") or record.get("severity") or "medium"
    if severity not in {"low", "medium", "high", "critical"}:
        severity = "medium"
    return RawFailureTrace(
        id=stable_id("prod-trace", source, record_id),
        source=source,
        user_task=user_task or "Investigate a failed LLM response from production telemetry.",
        model_output=model_output or "",
        expected_behavior=expected or "Use available evidence, avoid unsupported claims, and route uncertain cases for review.",
        failure_summary=summary or "Production failure trace imported into BreakPoint.",
        retrieved_docs=docs or [],
        tool_calls=tools or [],
        feedback=str(record.get("feedback") or record.get("comment") or ""),
        incident_severity=severity,
        metadata={
            "provenance": {
                "source_record_id": record_id,
                "source_adapter": source,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
            "raw_keys": sorted(record.keys()),
            "domain": record.get("domain", source),
        },
    )


def _docs_from_any(value: Any) -> list[RetrievedDocumentTrace]:
    if value is None:
        return []
    if isinstance(value, str):
        return [RetrievedDocumentTrace(id=stable_id("doc", value), content=value, rank=0)]
    docs = value if isinstance(value, list) else [value]
    output: list[RetrievedDocumentTrace] = []
    for index, doc in enumerate(docs):
        if not isinstance(doc, dict):
            doc = {"content": str(doc)}
        output.append(
            RetrievedDocumentTrace(
                id=str(doc.get("id") or doc.get("document_id") or stable_id("doc", index, doc)),
                source_type=str(doc.get("source_type") or doc.get("type") or "retrieved_document"),
                title=str(doc.get("title") or doc.get("name") or ""),
                content=str(doc.get("content") or doc.get("text") or doc.get("page_content") or ""),
                effective_date=doc.get("effective_date") or doc.get("date"),
                reliability=float(doc.get("reliability", doc.get("score", 0.5))),
                rank=int(doc.get("rank", index)),
                metadata={key: val for key, val in doc.items() if key not in {"id", "document_id", "source_type", "type", "title", "name", "content", "text", "page_content", "effective_date", "date", "reliability", "score", "rank"}},
            )
        )
    return output


def _tools_from_any(value: Any) -> list[ToolCallTrace]:
    if value is None:
        return []
    tools = value if isinstance(value, list) else [value]
    output = []
    for tool in tools:
        if not isinstance(tool, dict):
            tool = {"name": str(tool)}
        status = tool.get("status", "ok")
        if status not in {"ok", "error", "timeout", "permission_denied"}:
            status = "error"
        output.append(
            ToolCallTrace(
                name=str(tool.get("name") or tool.get("tool_name") or "tool"),
                args=tool.get("args") or tool.get("arguments") or {},
                result=tool.get("result") or tool.get("output"),
                status=status,
                stale=bool(tool.get("stale", False)),
                latency_ms=float(tool.get("latency_ms", 0)),
                metadata={key: val for key, val in tool.items() if key not in {"name", "tool_name", "args", "arguments", "result", "output", "status", "stale", "latency_ms"}},
            )
        )
    return output


def _first(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return _response_to_text(value)
    return None


def _first_obj(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _messages_to_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        user_messages = [msg.get("content", "") for msg in value if isinstance(msg, dict) and msg.get("role") in {"user", "human"}]
        if user_messages:
            return "\n".join(str(msg) for msg in user_messages)
        return "\n".join(_response_to_text(msg) for msg in value)
    return _response_to_text(value)


def _response_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "output", "answer", "response"):
            if key in value:
                return _response_to_text(value[key])
    return json.dumps(value, sort_keys=True)
