"""BreakPoint eval data compiler."""

from breakpoint_eval.actual_data import build_actual_dataset_artifacts
from breakpoint_eval.calibration import calibrate_from_trace2eval_results
from breakpoint_eval.compiler import BreakPointCompiler
from breakpoint_eval.ingestion import ingest_payload, load_production_traces
from breakpoint_eval.models import DatasetBundle, EvalItem, ValidationReport
from breakpoint_eval.production import build_production_regression_pack
from breakpoint_eval.reports import write_live_judge_report
from breakpoint_eval.sdk import BreakPoint, TraceBuilder
from breakpoint_eval.specs import BreakPointSpec, compile_spec
from breakpoint_eval.traces import RawFailureTrace, Trace2EvalResult, compile_trace

__all__ = [
    "BreakPoint",
    "BreakPointCompiler",
    "BreakPointSpec",
    "DatasetBundle",
    "EvalItem",
    "RawFailureTrace",
    "TraceBuilder",
    "Trace2EvalResult",
    "ValidationReport",
    "build_actual_dataset_artifacts",
    "build_production_regression_pack",
    "calibrate_from_trace2eval_results",
    "compile_spec",
    "compile_trace",
    "ingest_payload",
    "load_production_traces",
    "write_live_judge_report",
]

__version__ = "0.3.0"
