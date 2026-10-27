from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp


WRITE_PATTERN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|ATTACH|DETACH)\b", re.I)


def assert_read_only_sql(sql: str) -> None:
    statements = [node for node in sqlglot.parse(sql, read="duckdb") if node is not None]
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("Exactly one SELECT/CTE/set query is permitted")
    for node in statements[0].walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop,
                             exp.Alter, exp.Command, exp.Copy, exp.Into)):
            raise ValueError("Only read-only database queries are permitted")
        if isinstance(node, exp.Table) and not isinstance(node.this, exp.Identifier):
            raise ValueError("External table functions are not benchmark database tables")


def parse_sql(sql: str) -> exp.Expression:
    assert_read_only_sql(sql)
    return sqlglot.parse_one(sql, read="duckdb")


def referenced_tables(sql: str) -> set[str]:
    tree = parse_sql(sql)
    return {table.name for table in tree.find_all(exp.Table)}


@dataclass(frozen=True)
class ColumnLiteralPredicate:
    table: str | None
    column: str
    operator: str
    value: Any


def _literal_value(node: exp.Expression) -> Any:
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        text = str(node.this)
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    return None


def extract_column_literal_predicates(sql: str) -> list[ColumnLiteralPredicate]:
    tree = parse_sql(sql)
    alias_to_table: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        alias_to_table[table.alias_or_name] = table.name

    out: list[ColumnLiteralPredicate] = []
    operator_map: dict[type[exp.Expression], str] = {
        exp.EQ: "=",
        exp.GT: ">",
        exp.GTE: ">=",
        exp.LT: "<",
        exp.LTE: "<=",
        exp.NEQ: "!=",
    }
    for cls, op in operator_map.items():
        for node in tree.find_all(cls):
            left, right = node.left, node.right
            if isinstance(left, exp.Column):
                value = _literal_value(right)
                if value is not None:
                    out.append(
                        ColumnLiteralPredicate(
                            table=alias_to_table.get(left.table, left.table or None),
                            column=left.name,
                            operator=op,
                            value=value,
                        )
                    )
            elif isinstance(right, exp.Column):
                value = _literal_value(left)
                if value is not None:
                    out.append(
                        ColumnLiteralPredicate(
                            table=alias_to_table.get(right.table, right.table or None),
                            column=right.name,
                            operator=op,
                            value=value,
                        )
                    )
    return out


def normalized_sql_signature(sql: str) -> str:
    """Internal task-analysis signature; not part of Task Spec.

    Dates and numeric literals are abstracted while string category values are retained,
    because category values often define different predictive quantities.
    """
    tree = parse_sql(sql).copy()
    for literal in list(tree.find_all(exp.Literal)):
        text = str(literal.this)
        if literal.is_string and not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T].*)?", text):
            continue
        literal.replace(exp.Var(this="<VALUE>"))
    return tree.sql(dialect="duckdb", pretty=False, normalize=True)
