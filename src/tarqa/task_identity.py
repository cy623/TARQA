"""Canonical semantic task identity; SQL defines identity, never model features."""
from datetime import datetime
import re
import hashlib
import json
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.simplify import simplify
from tarqa.data.releases import stamp

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d\d:\d\d:\d\d)?$")


def anchor_columns(domain, anchor):
    return {"stats": {"userid", "owneruserid", "id"},
            "ncaa": {"team_id", "team", "wteam", "lteam"},
            "airline": {"origin", "dest", "uniquecarrier", "carrier", "code"}}.get(domain, {anchor["key"].lower()})


def sql_template(sql, domain, anchor, reference_time, *, replacement_anchor=None,
                 replacement_time=None, canonical=True):
    tree = sqlglot.parse_one(sql, read="duckdb")
    ref = datetime.fromisoformat(reference_time)
    newref = datetime.fromisoformat(replacement_time) if replacement_time else None
    columns = anchor_columns(domain, anchor)
    replaced = 0
    for node in list(tree.find_all(exp.Literal)):
        if node.is_string and DATE.fullmatch(node.this):
            offset = datetime.fromisoformat(node.this) - ref
            if canonical:
                node.replace(exp.Literal.string(f"__REFERENCE_PLUS_{int(offset.total_seconds())}_SECONDS__"))
            else:
                dt = newref + offset
                node.replace(exp.Literal.string(dt.strftime("%Y-%m-%d") if len(node.this) == 10 else stamp(dt)))
            continue
        parent = node.parent
        if isinstance(parent, exp.EQ):
            other = parent.expression if parent.this is node else parent.this
            if isinstance(other, exp.Column):
                col = other.name.lower()
                if domain == "ncaa" and col == "season":
                    if canonical:
                        node.replace(exp.Literal.string("__REFERENCE_SEASON__"))
                    elif newref:
                        node.replace(exp.Literal.number(newref.year))
                    continue
                if col in columns and str(node.this) == str(anchor["value"]):
                    replaced += 1
                    node.replace(exp.Literal.string("__ANCHOR__") if canonical else exp.convert(replacement_anchor))
    if not replaced:
        raise ValueError("No exact anchor equality found; cannot safely bind this label program")
    if canonical:
        tree = qualify(tree, dialect="duckdb", validate_qualify_columns=False,
                       qualify_columns=False, quote_identifiers=False, identify=False,
                       canonicalize_table_aliases=True)
        tree = simplify(tree)
    return tree.sql(dialect="duckdb", pretty=False, normalize=True)


def task_definition(q, domain):
    anchor = q.anchor.model_dump(mode="json")
    template = sql_template(q.golden_sql, domain, anchor, q.reference_time)
    # Preserve anchor role, type, aggregates, columns, filters, thresholds, units
    # implied by target column, all window offsets and strict/inclusive bounds.
    signature = {"domain": domain, "anchor_role": [anchor["table"], anchor["key"]],
                 "target_type": "Classification" if q.answer_nature == "Boolean" else "Regression",
                 "program": template}
    return domain + "_" + digest(signature)[:16], signature


