from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tarqa.utils.io import read_json


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: str
    description: str = ""

    @property
    def is_numeric(self) -> bool:
        t = self.dtype.upper()
        return any(token in t for token in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "NUMERIC", "HUGEINT", "UBIGINT"))

    @property
    def is_temporal(self) -> bool:
        t = self.dtype.upper()
        return any(token in t for token in ("DATE", "TIME", "TIMESTAMP"))

    @property
    def is_boolean(self) -> bool:
        return "BOOL" in self.dtype.upper()


@dataclass(frozen=True)
class ForeignKey:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str

    @property
    def signature(self) -> str:
        return f"{self.child_table}.{self.child_column}->{self.parent_table}.{self.parent_column}"


@dataclass
class TableSchema:
    name: str
    description: str = ""
    columns: dict[str, ColumnSchema] = field(default_factory=dict)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)


@dataclass
class SchemaCatalog:
    domain: str
    description: str
    tables: dict[str, TableSchema]

    @classmethod
    def from_files(
        cls,
        domain: str,
        metadata_path: str | Path,
        profiling_path: str | Path | None = None,
        database_path: str | Path | None = None,
    ) -> "SchemaCatalog":
        metadata = read_json(metadata_path)
        profiling = read_json(profiling_path) if profiling_path else {}

        prof_tables: dict[str, dict[str, Any]] = {}
        if isinstance(profiling, dict):
            for table in profiling.get("tables", []):
                if isinstance(table, dict) and table.get("table_name"):
                    prof_tables[str(table["table_name"])] = table

        tables: dict[str, TableSchema] = {}
        for table_name, table_meta in metadata.get("tables", {}).items():
            prof = prof_tables.get(table_name, {})
            schema_block = prof.get("schema", {}) if isinstance(prof, dict) else {}
            comments = table_meta.get("comments", {}) or {}

            columns: dict[str, ColumnSchema] = {}
            for col in schema_block.get("columns", []):
                name = str(col["name"])
                columns[name] = ColumnSchema(
                    name=name,
                    dtype=str(col.get("type", "UNKNOWN")),
                    description=str(col.get("comment") or comments.get(name, "")),
                )
            for name, description in comments.items():
                if name not in columns:
                    columns[name] = ColumnSchema(name=name, dtype="UNKNOWN", description=str(description))

            fks: list[ForeignKey] = []
            for fk in table_meta.get("foreign_keys", []) or []:
                ref = str(fk.get("references", ""))
                if "." not in ref:
                    continue
                parent_table, parent_column = ref.split(".", 1)
                fks.append(
                    ForeignKey(
                        child_table=table_name,
                        child_column=str(fk["column"]),
                        parent_table=parent_table,
                        parent_column=parent_column,
                    )
                )

            tables[table_name] = TableSchema(
                name=table_name,
                description=str(table_meta.get("description", "")),
                columns=columns,
                primary_keys=[str(x) for x in table_meta.get("primary_key", []) or []],
                foreign_keys=fks,
            )

        if database_path is not None:
            import duckdb
            connection = duckdb.connect(str(database_path), read_only=True)
            try:
                for table in tables.values():
                    quoted = '"' + table.name.replace('"', '""') + '"'
                    actual = connection.execute(f"DESCRIBE SELECT * FROM {quoted}").fetchall()
                    table.columns = {r[0]: ColumnSchema(r[0],r[1],table.columns.get(r[0],ColumnSchema(r[0],r[1])).description) for r in actual}
            finally:
                connection.close()
        return cls(
            domain=domain,
            description=str(metadata.get("database_description", "")),
            tables=tables,
        )

    @property
    def foreign_keys(self) -> list[ForeignKey]:
        return [fk for table in self.tables.values() for fk in table.foreign_keys]

    @property
    def table_names(self) -> list[str]:
        return sorted(self.tables)

    def has_table(self, name: str) -> bool:
        return name in self.tables

    def has_column(self, table: str, column: str) -> bool:
        return table in self.tables and column in self.tables[table].columns

    def primary_keys(self, table: str) -> list[str]:
        return list(self.tables[table].primary_keys)

    def entity_key_columns(self) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for table in self.tables.values():
            out.update((table.name, key) for key in table.primary_keys)
            out.update((fk.child_table, fk.child_column) for fk in table.foreign_keys)
        return out
