from __future__ import annotations

from tarqa.data.metadata import SchemaCatalog
from tarqa.utils.text import normalize_space


def _truncate(text: str, limit: int) -> str:
    text = normalize_space(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def serialize_schema(
    catalog: SchemaCatalog,
    *,
    max_description_chars: int = 360,
    max_column_description_chars: int = 180,
    max_columns_per_table: int | None = None,
) -> str:
    parts = [f"Database: {catalog.domain}"]
    if catalog.description:
        parts.append(f"Database description: {_truncate(catalog.description, 500)}")

    for table_name in catalog.table_names:
        table = catalog.tables[table_name]
        parts.append(f"\n<TABLE name=\"{table_name}\">")
        if table.description:
            parts.append(f"Description: {_truncate(table.description, max_description_chars)}")
        if table.primary_keys:
            parts.append("Primary key: " + ", ".join(table.primary_keys))
        parts.append("Columns:")
        columns = list(table.columns.values())
        if max_columns_per_table is not None:
            columns = columns[:max_columns_per_table]
        for column in columns:
            desc = _truncate(column.description, max_column_description_chars)
            suffix = f" — {desc}" if desc else ""
            parts.append(f"- {column.name}: {column.dtype}{suffix}")
        if table.foreign_keys:
            parts.append("Foreign keys:")
            for fk in table.foreign_keys:
                parts.append(
                    f"- {fk.child_column} -> {fk.parent_table}.{fk.parent_column}"
                )
        parts.append("</TABLE>")
    return "\n".join(parts)
