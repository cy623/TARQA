from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tarqa.prediction.fk_graph import RelationPath
from tarqa.types import EntityAnchor


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class PathQuery:
    from_sql: str
    where_sql: str
    parameters: tuple[Any, ...]
    final_alias: str


def build_path_query(path: RelationPath, anchor: EntityAnchor) -> PathQuery:
    if path.start_table != anchor.table:
        raise ValueError("Relation path does not start at the anchor table.")
    aliases = [f"t{i}" for i in range(len(path.steps) + 1)]
    parts = [f"FROM {quote_identifier(path.start_table)} {aliases[0]}"]
    current_table = path.start_table

    for idx, step in enumerate(path.steps, 1):
        if step.from_table != current_table:
            raise ValueError(f"Invalid path continuity at {step}")
        current_alias = aliases[idx - 1]
        next_alias = aliases[idx]
        fk = step.foreign_key
        if current_table == fk.parent_table and step.to_table == fk.child_table:
            condition = (
                f"{next_alias}.{quote_identifier(fk.child_column)} = "
                f"{current_alias}.{quote_identifier(fk.parent_column)}"
            )
        elif current_table == fk.child_table and step.to_table == fk.parent_table:
            condition = (
                f"{current_alias}.{quote_identifier(fk.child_column)} = "
                f"{next_alias}.{quote_identifier(fk.parent_column)}"
            )
        else:
            raise ValueError(f"Path step does not match FK direction: {step}")
        parts.append(
            f"JOIN {quote_identifier(step.to_table)} {next_alias} ON {condition}"
        )
        current_table = step.to_table

    where_sql = f"{aliases[0]}.{quote_identifier(anchor.key)} = ?"
    return PathQuery(
        from_sql="\n".join(parts),
        where_sql=where_sql,
        parameters=(anchor.value,),
        final_alias=aliases[-1],
    )
