from __future__ import annotations

from pathlib import Path
from typing import Any

from breakpoint_eval.actual_data import build_actual_dataset_artifacts
from breakpoint_eval.ingestion import IngestionKind, load_production_traces
from breakpoint_eval.judges import available_validation_suite
from breakpoint_eval.models import EvalCase
from breakpoint_eval.production import ProductionPackBuild, build_production_regression_pack, build_regression_pack_from_traces
from breakpoint_eval.reports import write_live_judge_report
from breakpoint_eval.traces import RawFailureTrace, RetrievedDocumentTrace, ToolCallTrace, compile_traces


class TraceBuilder:
    """Small typed builders for common production failure sources."""

    @staticmethod
    def rag_failure(
        *,
        id: str,
        question: str,
        bad_answer: str,
        expected_behavior: str,
        retrieved_docs: list[dict[str, Any]],
        summary: str = "",
        severity: str = "high",
        metadata: dict[str, Any] | None = None,
    ) -> RawFailureTrace:
        return RawFailureTrace(
            id=id,
            source="rag_log",
            user_task=question,
            model_output=bad_answer,
            expected_behavior=expected_behavior,
            failure_summary=summary or "RAG answer failed to follow the best available evidence.",
            retrieved_docs=[RetrievedDocumentTrace.model_validate(doc) for doc in retrieved_docs],
            incident_severity=_severity(severity),
            metadata=metadata or {},
        )

    @staticmethod
    def tool_failure(
        *,
        id: str,
        task: str,
        bad_answer: str,
        expected_behavior: str,
        tool_calls: list[dict[str, Any]],
        summary: str = "",
        severity: str = "high",
        metadata: dict[str, Any] | None = None,
    ) -> RawFailureTrace:
        return RawFailureTrace(
            id=id,
            source="tool_trace",
            user_task=task,
            model_output=bad_answer,
            expected_behavior=expected_behavior,
            failure_summary=summary or "Agent used the wrong tool result or tool path.",
            tool_calls=[ToolCallTrace.model_validate(call) for call in tool_calls],
            incident_severity=_severity(severity),
            metadata=metadata or {},
        )

    @staticmethod
    def support_ticket_failure(
        *,
        id: str,
        subject: str,
        customer_message: str,
        bad_response: str,
        correct_resolution: str,
        summary: str = "",
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> RawFailureTrace:
        return RawFailureTrace(
            id=id,
            source="support_ticket",
            user_task=f"{subject}\n\n{customer_message}".strip(),
            model_output=bad_response,
            expected_behavior=correct_resolution,
            failure_summary=summary or f"Support-ticket response failed: {subject}",
            incident_severity=_severity(severity),
            metadata=metadata or {},
        )

    @staticmethod
    def red_team_failure(
        *,
        id: str,
        attack_prompt: str,
        unsafe_response: str,
        expected_behavior: str,
        summary: str = "",
        severity: str = "critical",
        metadata: dict[str, Any] | None = None,
    ) -> RawFailureTrace:
        return RawFailureTrace(
            id=id,
            source="red_team",
            user_task=attack_prompt,
            model_output=unsafe_response,
            expected_behavior=expected_behavior,
            failure_summary=summary or "Red-team prompt exposed unsafe model behavior.",
            incident_severity=_severity(severity),
            metadata=metadata or {},
        )


class BreakPoint:
    """SDK facade for trace ingestion, compilation, and report generation."""

    def __init__(self, *, include_external_judges: bool = False, variants_per_item: int = 3) -> None:
        self.include_external_judges = include_external_judges
        self.variants_per_item = variants_per_item

    def ingest(self, path: str | Path, *, source: IngestionKind | None = None) -> list[RawFailureTrace]:
        return load_production_traces(path, source=source).traces

    def compile_traces(self, traces: list[RawFailureTrace]) -> list[EvalCase]:
        results = compile_traces(
            traces,
            validator=available_validation_suite(include_external=self.include_external_judges),
            variants_per_item=self.variants_per_item,
        )
        return [result.eval_case for result in results]

    def build_pack(
        self,
        traces: list[RawFailureTrace],
        *,
        output_dir: str | Path = "artifacts/sdk_pack",
    ) -> ProductionPackBuild:
        return build_regression_pack_from_traces(
            traces=traces,
            output_dir=output_dir,
            variants_per_item=self.variants_per_item,
            include_external_judges=self.include_external_judges,
        )

    def build_pack_from_file(
        self,
        path: str | Path,
        *,
        source: IngestionKind | None = None,
        output_dir: str | Path = "artifacts/production_pack",
    ) -> ProductionPackBuild:
        return build_production_regression_pack(
            source_path=path,
            source=source,
            output_dir=output_dir,
            variants_per_item=self.variants_per_item,
            include_external_judges=self.include_external_judges,
        )

    def build_actual(self, *, output_dir: str | Path = "artifacts/actual") -> Any:
        return build_actual_dataset_artifacts(
            output_dir=output_dir,
            variants_per_item=self.variants_per_item,
            include_external_judges=self.include_external_judges,
        )

    def write_report(
        self,
        *,
        results_path: str | Path = "artifacts/actual/trace2eval_results.json",
        out_dir: str | Path = "artifacts/reports",
    ) -> dict[str, Any]:
        return write_live_judge_report(results_path, out_dir)


def _severity(value: str) -> str:
    return value if value in {"low", "medium", "high", "critical"} else "medium"
