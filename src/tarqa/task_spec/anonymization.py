from __future__ import annotations

import random
from dataclasses import dataclass
import re

from tarqa.data.metadata import ColumnSchema, ForeignKey, SchemaCatalog, TableSchema
from tarqa.types import TaskSpec
from tarqa.utils.seed import stable_int


@dataclass(frozen=True)
class TableAnonymization:
    forward: dict[str, str]
    reverse: dict[str, str]

    @classmethod
    def build(cls, table_names: list[str], key: str) -> "TableAnonymization":
        shuffled = sorted(table_names)
        rng = random.Random(stable_int(key))
        rng.shuffle(shuffled)
        forward = {
            real: f"table_{chr(ord('A') + idx)}" if idx < 26 else f"table_{idx + 1}"
            for idx, real in enumerate(shuffled)
        }
        return cls(forward=forward, reverse={v: k for k, v in forward.items()})


    def apply_text(self, text: str) -> str:
        out = text
        for real in sorted(self.forward, key=len, reverse=True):
            out = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(real)}(?![A-Za-z0-9_])",
                self.forward[real],
                out,
                flags=re.IGNORECASE,
            )
        return out

    def apply_spec(self, spec: TaskSpec) -> TaskSpec:
        return spec.model_copy(
            update={"relevant_tables": [self.forward.get(t, t) for t in spec.relevant_tables]}
        )

    def restore_spec(self, spec: TaskSpec) -> TaskSpec:
        return spec.model_copy(
            update={"relevant_tables": [self.reverse.get(t, t) for t in spec.relevant_tables]}
        )

    def apply_catalog(self, catalog: SchemaCatalog) -> SchemaCatalog:
        tables: dict[str, TableSchema] = {}
        for real_name, table in catalog.tables.items():
            anon_name = self.forward[real_name]
            fks = [
                ForeignKey(
                    child_table=self.forward.get(fk.child_table, fk.child_table),
                    child_column=fk.child_column,
                    parent_table=self.forward.get(fk.parent_table, fk.parent_table),
                    parent_column=fk.parent_column,
                )
                for fk in table.foreign_keys
            ]
            tables[anon_name] = TableSchema(
                name=anon_name,
                description=table.description,
                columns={
                    name: ColumnSchema(name=c.name, dtype=c.dtype, description=c.description)
                    for name, c in table.columns.items()
                },
                primary_keys=list(table.primary_keys),
                foreign_keys=fks,
            )
        return SchemaCatalog(domain=catalog.domain, description=catalog.description, tables=tables)
