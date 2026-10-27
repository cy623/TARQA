from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import duckdb

from tarqa.data.metadata import ForeignKey, SchemaCatalog


@dataclass(frozen=True)
class PathStep:
    from_table: str
    to_table: str
    foreign_key: ForeignKey

    @property
    def signature(self) -> str:
        return f"{self.from_table}|{self.foreign_key.signature}|{self.to_table}"


@dataclass(frozen=True)
class RelationPath:
    start_table: str
    end_table: str
    steps: tuple[PathStep, ...]

    @property
    def signature(self) -> str:
        if not self.steps:
            return self.start_table
        return "__".join(step.signature for step in self.steps)


class SchemaGraph:
    def __init__(self, catalog: SchemaCatalog, foreign_keys: list[ForeignKey] | None = None) -> None:
        self.catalog = catalog
        self.foreign_keys = foreign_keys or catalog.foreign_keys
        self.adj: dict[str, list[PathStep]] = defaultdict(list)
        for fk in self.foreign_keys:
            self.adj[fk.child_table].append(
                PathStep(fk.child_table, fk.parent_table, fk)
            )
            self.adj[fk.parent_table].append(
                PathStep(fk.parent_table, fk.child_table, fk)
            )

    def paths(
        self,
        start: str,
        end: str,
        *,
        max_hops: int = 2,
        max_paths: int = 4,
    ) -> list[RelationPath]:
        if start == end:
            return [RelationPath(start, end, ())]
        found: list[RelationPath] = []
        queue: deque[tuple[str, tuple[PathStep, ...], frozenset[str]]] = deque(
            [(start, (), frozenset({start}))]
        )
        while queue and len(found) < max_paths:
            table, steps, visited = queue.popleft()
            if len(steps) >= max_hops:
                continue
            for step in self.adj.get(table, []):
                if step.to_table in visited:
                    continue
                next_steps = (*steps, step)
                if step.to_table == end:
                    found.append(RelationPath(start, end, next_steps))
                    if len(found) >= max_paths:
                        break
                else:
                    queue.append(
                        (step.to_table, next_steps, visited | {step.to_table})
                    )
        return found

    def reachable_tables(self, start: str, max_hops: int = 2) -> set[str]:
        reached = {start}
        queue = deque([(start, 0)])
        while queue:
            table, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for step in self.adj.get(table, []):
                if step.to_table not in reached:
                    reached.add(step.to_table)
                    queue.append((step.to_table, depth + 1))
        return reached


def _normalized_name(name: str) -> str:
    value = name.lower().replace("_", "")
    for prefix in ("w", "l", "src", "dst", "from", "to"):
        if value.startswith(prefix) and len(value) > len(prefix) + 2:
            value = value[len(prefix) :]
    for suffix in ("identifier", "id", "key", "code"):
        if value.endswith(suffix) and len(value) > len(suffix):
            value = value[: -len(suffix)]
    return value


def infer_missing_foreign_keys(
    catalog: SchemaCatalog,
    database_path: str | Path,
    *,
    containment_threshold: float = 0.9,
    max_distinct_values: int = 200_000,
) -> list[ForeignKey]:
    """Conservative FK inference used only when metadata is incomplete.

    Declared keys are always retained. Candidates require a parent primary key,
    normalized-name compatibility, and high child-to-parent value containment.
    """
    declared = list(catalog.foreign_keys)
    declared_signatures = {fk.signature for fk in declared}
    parents: list[tuple[str, str]] = []
    for table in catalog.tables.values():
        parents.extend((table.name, key) for key in table.primary_keys)
    if not parents:
        return declared

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        inferred: list[ForeignKey] = []
        for child in catalog.tables.values():
            for column in child.columns.values():
                if (child.name, column.name) in parents:
                    continue
                child_norm = _normalized_name(column.name)
                if not child_norm:
                    continue
                for parent_table, parent_key in parents:
                    if parent_table == child.name:
                        continue
                    parent_norm = _normalized_name(parent_key)
                    table_norm = _normalized_name(parent_table)
                    if child_norm not in {parent_norm, table_norm}:
                        continue
                    candidate = ForeignKey(child.name, column.name, parent_table, parent_key)
                    if candidate.signature in declared_signatures:
                        continue
                    query = f'''
                        WITH child_values AS (
                            SELECT DISTINCT "{column.name}" AS v
                            FROM "{child.name}"
                            WHERE "{column.name}" IS NOT NULL
                            LIMIT {int(max_distinct_values)}
                        ), parent_values AS (
                            SELECT DISTINCT "{parent_key}" AS v
                            FROM "{parent_table}"
                            WHERE "{parent_key}" IS NOT NULL
                        )
                        SELECT
                            COUNT(*) AS child_n,
                            COUNT(p.v) AS matched_n
                        FROM child_values c
                        LEFT JOIN parent_values p USING(v)
                    '''
                    child_n, matched_n = connection.execute(query).fetchone()
                    if child_n and matched_n / child_n >= containment_threshold:
                        inferred.append(candidate)
                        declared_signatures.add(candidate.signature)
        return [*declared, *inferred]
    finally:
        connection.close()
