from __future__ import annotations

import json
import math
from pathlib import Path

from breakpoint_eval.models import DatasetBundle, EvalItem


class DuckDBStore:
    def __init__(self, path: str | Path) -> None:
        import duckdb

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.path))

    def initialize(self) -> None:
        self.connection.execute(
            """
            create table if not exists eval_items (
                dataset_id varchar,
                item_id varchar,
                category varchar,
                family varchar,
                prompt varchar,
                expected varchar,
                payload json
            )
            """
        )
        self.connection.execute(
            """
            create table if not exists validation_reports (
                dataset_id varchar,
                item_id varchar,
                passed boolean,
                payload json
            )
            """
        )
        self.connection.execute(
            """
            create table if not exists dataset_metrics (
                dataset_id varchar primary key,
                payload json
            )
            """
        )

    def save_bundle(self, bundle: DatasetBundle) -> None:
        self.initialize()
        self.connection.execute("delete from eval_items where dataset_id = ?", [bundle.dataset_id])
        self.connection.execute("delete from validation_reports where dataset_id = ?", [bundle.dataset_id])
        self.connection.execute("delete from dataset_metrics where dataset_id = ?", [bundle.dataset_id])
        for item in bundle.items:
            self.connection.execute(
                "insert into eval_items values (?, ?, ?, ?, ?, ?, ?)",
                [
                    bundle.dataset_id,
                    item.id,
                    item.category,
                    item.family,
                    item.prompt,
                    item.expected_answer.value,
                    json.dumps(item.model_dump(mode="json")),
                ],
            )
        for report in bundle.validation_reports + bundle.rejected_reports:
            self.connection.execute(
                "insert into validation_reports values (?, ?, ?, ?)",
                [bundle.dataset_id, report.item_id, report.passed, json.dumps(report.model_dump(mode="json"))],
            )
        self.connection.execute(
            "insert into dataset_metrics values (?, ?)",
            [bundle.dataset_id, json.dumps(bundle.metrics)],
        )

    def latest_metrics(self) -> dict[str, object] | None:
        row = self.connection.execute(
            "select payload from dataset_metrics order by dataset_id desc limit 1"
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self.connection.close()


class HashVectorIndex:
    """Small local retrieval index used when FAISS/Qdrant are not configured."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions
        self.rows: list[tuple[str, list[float], EvalItem]] = []

    def add(self, item: EvalItem) -> None:
        self.rows.append((item.id, self._embed(item.prompt), item))

    def search(self, query: str, k: int = 5) -> list[tuple[float, EvalItem]]:
        query_vec = self._embed(query)
        scored = [(self._cosine(query_vec, vec), item) for _, vec, item in self.rows]
        return sorted(scored, key=lambda pair: pair[0], reverse=True)[:k]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            bucket = hash(token) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=False))
