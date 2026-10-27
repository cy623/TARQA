from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tarqa.answer.normalizer import normalize_sql_result
from tarqa.data.metadata import SchemaCatalog
from tarqa.reasoning.executor import DuckDBReadOnlyExecutor
from tarqa.reasoning.schema_pruning import pruned_catalog
from tarqa.task_spec.serialization import serialize_schema
from tarqa.types import CanonicalAnswer, TaskSpec, Track


@dataclass
class ReasoningResult:
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    answer: CanonicalAnswer
    attempts: int


class ReasoningPipeline:
    def __init__(
        self,
        catalog: SchemaCatalog,
        database_path: str | Path,
        model: Any,
        *,
        add_bridge_tables: bool = True,
        max_repair_attempts: int = 2,
        timeout_seconds: float = 120,
        schema_policy: str = "full",
        schema_renderer=None,
    ) -> None:
        self.catalog = catalog
        self.executor = DuckDBReadOnlyExecutor(database_path, timeout_seconds=timeout_seconds)
        self.model = model
        self.add_bridge_tables = add_bridge_tables
        self.max_repair_attempts = max_repair_attempts
        if schema_policy not in {"full", "tsg_selected"}:
            raise ValueError(f"Unknown Reasoning schema policy: {schema_policy}")
        self.schema_policy = schema_policy
        self.schema_renderer = schema_renderer or serialize_schema

    def catalog_for_spec(self, spec: TaskSpec) -> SchemaCatalog:
        if spec.track != Track.REASONING:
            raise ValueError("ReasoningPipeline received a non-reasoning Task Spec.")
        if self.schema_policy == "tsg_selected":
            return pruned_catalog(self.catalog, spec.relevant_tables, self.add_bridge_tables)
        return self.catalog

    def answer(self, question: str, spec: TaskSpec, answer_nature: str | None = None) -> ReasoningResult:
        input_catalog = self.catalog_for_spec(spec)
        if hasattr(self.model, "generate_candidates"):
            errors = []
            candidates = self.model.generate_candidates(question, input_catalog, self.executor.database_path)
            for attempt, sql in enumerate(candidates, 1):
                try:
                    columns, rows = self.executor.execute(sql)
                    answer = normalize_sql_result(rows, answer_nature=answer_nature)
                    return ReasoningResult(sql, columns, rows, answer, attempt)
                except Exception as exc:
                    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                        raise
                    errors.append(f"{type(exc).__name__}: {exc}")
            raise RuntimeError(f"No executable RESDSQL candidate: {errors[:3]}")
        schema_text = self.schema_renderer(input_catalog)
        sql = self.model.generate(question, schema_text)
        last_error: Exception | None = None
        for attempt in range(self.max_repair_attempts + 1):
            try:
                columns, rows = self.executor.execute(sql)
                answer = normalize_sql_result(rows, answer_nature=answer_nature)
                return ReasoningResult(sql, columns, rows, answer, attempt + 1)
            except Exception as exc:
                if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                    raise
                last_error = exc
                if attempt >= self.max_repair_attempts:
                    break
                sql = self.model.generate(
                    question,
                    schema_text,
                    error=f"{type(exc).__name__}: {exc}",
                    sql=sql,
                )
        raise RuntimeError(f"Text-to-SQL failed after retries: {last_error}")

    def close(self):
        if hasattr(self.model, "release_database"):
            self.model.release_database(self.catalog, self.executor.database_path)
