"""Schema/skeleton formatting. Gold SQL is used only to build supervision.

Ranked input format follows RESDSQL/text2sql_data_generator.py (MIT, RUC
KBReasoning, 2023). DuckDB normalization deliberately preserves quoted
identifiers, aliases and literal values; the original SQLite string rewriting
is unsuitable for arbitrary DuckDB queries.
"""
import re
from copy import deepcopy

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from tarqa.data.sql_utils import assert_read_only_sql


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


def public_payload(question, catalog, contents=None):
    """Only question, complete schema and real DB contents; no QA metadata."""
    tables = []
    for name, table in catalog.tables.items():
        tables.append(dict(table_name_original=name,
                           table_name=name.replace("_", " ").lower(),
                           column_names_original=list(table.columns),
                           column_names=[c.replace("_", " ").lower() for c in table.columns],
                           db_contents=[contents.matches(question, name, c) if contents else []
                                        for c in table.columns]))
    return dict(question=question, db_id=catalog.domain, db_schema=tables,
                fk=[dict(source_table_name_original=f.child_table,
                         source_column_name_original=f.child_column,
                         target_table_name_original=f.parent_table,
                         target_column_name_original=f.parent_column)
                    for f in catalog.foreign_keys])


def classifier_words(payload, selections):
    """Pre-split RoBERTa words plus table/column token alignment indices."""
    words, table_ids, column_ids = [payload["question"]], [], []
    fk_columns = set()
    for fk in payload["fk"]:
        fk_columns.add((fk["source_table_name_original"], fk["source_column_name_original"]))
        fk_columns.add((fk["target_table_name_original"], fk["target_column_name_original"]))
    for table_id, columns in selections:
        table = payload["db_schema"][table_id]
        words.extend(["|", table["table_name"], ":"])
        table_ids.append(len(words) - 2)
        for col in columns:
            extra = list(table["db_contents"][col])
            if (table["table_name_original"], table["column_names_original"][col]) in fk_columns:
                extra.append("[FK]")
            text = table["column_names"][col]
            if extra:
                text += " ( " + " , ".join(extra) + " ) "
            column_ids.append(len(words))
            words.extend([text, ","])
        words.pop()
    return words, table_ids, column_ids


def ranker_chunks(payload, tokenizer, max_length=512):
    """Cover every column, including wide tables, without silent truncation."""
    chunks, current = [], []
    for tid, table in enumerate(payload["db_schema"]):
        for cid in range(len(table["column_names"])):
            candidate = deepcopy(current)
            if candidate and candidate[-1][0] == tid:
                candidate[-1][1].append(cid)
            else:
                candidate.append((tid, [cid]))
            words = classifier_words(payload, candidate)[0]
            n = len(tokenizer(words, is_split_into_words=True, truncation=False)["input_ids"])
            if n <= max_length:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = [(tid, [cid])]
                n = len(tokenizer(classifier_words(payload, current)[0],
                                  is_split_into_words=True, truncation=False)["input_ids"])
                if n > max_length:
                    raise ValueError("Question plus one schema column exceeds ranker max_length; "
                                     "shorten DB-content limits or review the question")
    if current:
        chunks.append(current)
    if not chunks:
        raise ValueError("RESDSQL requires a nonempty database schema")
    return chunks


def ranked_input(payload, scores, topk_tables=4, topk_columns=5, *, gold=None):
    """Gold is allowed only for constructing training inputs, never validation.

    All required gold items are retained in training, as in original RESDSQL.
    Additional items use ranker scores. Inference always uses predicted scores.
    """
    schema = payload["db_schema"]
    if len(scores["tables"]) != len(schema):
        raise ValueError("Ranker scores do not cover the full schema")
    def select(probabilities, k, labels=None):
        order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
        selected = [i for i in order if labels is not None and labels[i]]
        limit = len(order) if k <= 0 else max(k, len(selected))
        selected += [i for i in order if i not in selected][:max(0, limit-len(selected))]
        return selected
    tids = select(scores["tables"], topk_tables, gold["table_labels"] if gold else None)
    text = payload["question"]
    selected_names = set()
    columns_by_table = {}
    for tid in tids:
        table = schema[tid]
        name = table["table_name_original"]
        selected_names.add(name)
        cids = select(scores["columns"][tid], topk_columns,
                      gold["column_labels"][tid] if gold else None)
        columns_by_table[name] = [table["column_names_original"][c] for c in cids]
        descriptions = []
        for cid in cids:
            field = f"{name}.{table['column_names_original'][cid]}"
            values = table["db_contents"][cid]
            if values:
                field += " ( " + " , ".join(values) + " )"
            descriptions.append(field)
        text += f" | {name} : " + " , ".join(descriptions)
    for fk in payload["fk"]:
        if {fk["source_table_name_original"], fk["target_table_name_original"]} <= selected_names:
            text += (f" | {fk['source_table_name_original']}.{fk['source_column_name_original']}"
                     f" = {fk['target_table_name_original']}.{fk['target_column_name_original']}")
    return text, dict(tables=[schema[t]["table_name_original"] for t in tids], columns=columns_by_table)


def extract_generated_sql(text, dialect="duckdb"):
    """Remove skeleton at the first unquoted separator; preserve SQL literals."""
    text = text.strip()
    quote, i, delimiter = None, 0, None
    while i < len(text):
        char = text[i]
        if quote:
            if char == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        elif char in ("'", '"', '`'):
            quote = char
        elif char == "|" and text[max(0, i-1):i] != "|" and text[i+1:i+2] != "|":
            delimiter = i
            break
        i += 1
    sql = text[delimiter + 1:].strip() if delimiter is not None else text
    trees = [t for t in sqlglot.parse(sql, read=dialect) if t is not None]
    if len(trees) != 1 or not isinstance(trees[0], exp.Query):
        raise ValueError("RESDSQL did not generate one complete SELECT/CTE query")
    sql = trees[0].sql(dialect="duckdb")
    assert_read_only_sql(sql)
    return sql
