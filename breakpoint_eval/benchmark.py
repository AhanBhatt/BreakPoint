from __future__ import annotations

import json
from pathlib import Path

from breakpoint_eval.exporters import export_cases_jsonl, export_repro_manifest
from breakpoint_eval.models import EvalCase, EvalSuite


FAILUREGYM_TRACKS = {
    "rag_freshness_citation": {"families": {"rag_contradiction", "long_context_retrieval"}},
    "tool_use_faults": {"families": {"tool_misuse"}},
    "prompt_injection_retrieved_context": {"families": {"prompt_injection", "instruction_conflict"}},
    "format_structured_output": {"families": {"format_violation"}},
    "refusal_boundary": {"families": {"refusal_boundary"}},
}


def build_failuregym_manifest(suite: EvalSuite, cases: list[EvalCase], path: str | Path) -> dict[str, object]:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    tracks = []
    for name, config in FAILUREGYM_TRACKS.items():
        track_cases = [case for case in cases if case.failure_family in config["families"]]
        track_path = root / f"{name}.jsonl"
        export_cases_jsonl(track_cases, track_path)
        tracks.append(
            {
                "id": name,
                "name": name.replace("_", " ").title(),
                "case_count": len(track_cases),
                "families": sorted(config["families"]),
                "path": track_path.name,
            }
        )
    export_repro_manifest(suite, cases, root / "repro_manifest.json")
    manifest = {
        "benchmark": "BreakPoint-FailureGym",
        "version": "v1",
        "suite_id": suite.id,
        "description": "Failure-family benchmark generated from BreakPoint compiler outputs.",
        "tracks": tracks,
        "metrics": [
            "pass_rate",
            "mutation_survival_rate",
            "oracle_confidence",
            "trace_causality_score",
            "evidence_freshness_score",
            "injection_resistance_score",
            "abstention_precision_recall",
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
