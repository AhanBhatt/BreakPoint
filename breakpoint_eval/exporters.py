from __future__ import annotations

import json
from pathlib import Path

import yaml

from breakpoint_eval.models import EvalCase, EvalSuite


def export_cases_jsonl(cases: list[EvalCase], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dataset_row(), ensure_ascii=False) + "\n")


def export_openai_evals(suite: EvalSuite, cases: list[EvalCase], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "eval_name": suite.name,
        "registry_path": "breakpoint/generated",
        "dataset": [
            {
                "input": case.eval_item.prompt,
                "ideal": case.eval_item.expected_answer.value,
                "metadata": {
                    "case_id": case.id,
                    "failure_family": case.failure_family,
                    "mutation_lineage": case.mutation_lineage.model_dump(mode="json"),
                    "hidden_traps": [trap.model_dump(mode="json") for trap in case.eval_item.hidden_traps],
                },
            }
            for case in cases
        ],
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def export_lm_eval_task(suite: EvalSuite, cases: list[EvalCase], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": suite.id,
        "dataset_path": str(target.with_suffix(".jsonl").name),
        "output_type": "generate_until",
        "training_split": "test",
        "test_split": "test",
        "doc_to_text": "{{prompt}}",
        "doc_to_target": "{{expected}}",
        "metric_list": [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}],
        "metadata": {"version": 1, "suite_name": suite.name},
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    export_cases_jsonl(cases, target.with_suffix(".jsonl"))


def export_huggingface_rows(cases: list[EvalCase]) -> list[dict[str, object]]:
    return [case.to_dataset_row() for case in cases]


def export_repro_manifest(suite: EvalSuite, cases: list[EvalCase], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": suite.model_dump(mode="json"),
        "case_count": len(cases),
        "families": sorted({case.failure_family for case in cases}),
        "base_cases": sum(1 for case in cases if case.mutation_lineage.generation == 0),
        "variant_cases": sum(1 for case in cases if case.mutation_lineage.generation > 0),
        "lineage": [
            {
                "case_id": case.id,
                "seed_id": case.mutation_lineage.seed_id,
                "mutator": case.mutation_lineage.mutator,
                "path": case.mutation_lineage.path,
            }
            for case in cases
        ],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
