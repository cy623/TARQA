"""SQL extraction and training-demonstration annotations.

SQL annotation formatting retains the RESDSQL attribution (MIT, 2023 RUC
KBReasoning); see third_party_licenses/RESDSQL.txt.
"""
import re
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope
from tarqa.data.sql_utils import assert_read_only_sql

def extract_sql(text):
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    # A generated SQL block may end with a trailing fence.
    text = text.split("```")[0].strip()
    start = re.search(r"\b(SELECT|WITH)\b", text, re.I)
    if start:
        text = text[start.start():]
    # Preserve semicolons inside strings; the read-only executor rejects extra statements.
    return text

def normalize_sql(sql):
    assert_read_only_sql(sql)
    return sqlglot.parse_one(sql, read="duckdb").sql(dialect="duckdb", normalize=True)


def sql_supervision(sql, catalog):
    """Resolve CTEs, aliases, correlated references and stars against the DB.

    Fail on unresolved gold columns instead of silently assigning false labels.
    The qualified SQL is only used for labels; it never becomes a model input.
    """
    assert_read_only_sql(sql)
    schema = {t.name: {c.name: c.dtype for c in t.columns.values()}
              for t in catalog.tables.values()}
    tree = qualify(sqlglot.parse_one(sql, read="duckdb"), dialect="duckdb",
                   schema=schema, identify=False, validate_qualify_columns=True)
    table_lookup = {t.lower(): t for t in catalog.tables}
    tables, columns = set(), set()
    for scope in traverse_scope(tree):
        sources = {}
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table):
                table = table_lookup.get(source.name.lower())
                if table is None:
                    raise ValueError(f"Gold SQL refers to an unknown table: {source.name}")
                sources[alias.lower()] = table
                tables.add(table)
        for column in scope.columns:
            table = sources.get(column.table.lower())
            if table is None:
                # Correlated outer-column references belong to an ancestor scope.
                parent = scope.parent
                while parent is not None and table is None:
                    source = parent.sources.get(column.table)
                    if isinstance(source, exp.Table):
                        table = table_lookup.get(source.name.lower())
                    parent = parent.parent
            if table is not None and not column.is_star:
                names = {c.lower(): c for c in catalog.tables[table].columns}
                if column.name.lower() not in names:
                    raise ValueError(f"Unresolved gold column: {column.sql()}")
                columns.add((table, names[column.name.lower()]))
    normalized = normalize_sql(sql)
    masked = sqlglot.parse_one(normalized, read="duckdb").transform(
        lambda n: exp.Var(this="_") if isinstance(n, (exp.Column, exp.Literal, exp.Boolean))
        else exp.Identifier(this="_") if isinstance(n, exp.Identifier) else n)
    skeleton = masked.sql(dialect="duckdb").lower()
    # Match the original skeleton simplifications without modifying the real SQL.
    skeleton = re.sub(r"\bon _ = _(?: (?:and|or) _ = _)*", "", skeleton)
    skeleton = re.sub(r"_ (?:join _\s*)+", "_ ", skeleton)
    skeleton = re.sub(r"_\s*,\s*_(?:\s*,\s*_)*", "_", skeleton)
    skeleton = re.sub(r"_ (?:!=|<>|>=|<=|=|>|<) _", "_", skeleton)
    skeleton = re.sub(r"where _(?: (?:and|or) _)+", "where _", skeleton)
    skeleton = " ".join(skeleton.split())
    return dict(table_labels=[int(t in tables) for t in catalog.tables],
                column_labels=[[int((t, c) in columns) for c in table.columns]
                               for t, table in catalog.tables.items()],
                norm_sql=normalized, sql_skeleton=skeleton,
                output_sequence=f"{skeleton} | {normalized}")
