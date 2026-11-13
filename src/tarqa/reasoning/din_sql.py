"""DIN-SQL stages used by TARQA's frozen SQL executor.

Schema linking, decomposition, generation and self-correction use pinned
instructions and permitted Reasoning training demonstrations. Test gold SQL
and answers never enter generation. See third_party_licenses/DIN-SQL.txt.
"""
import json
from pathlib import Path
import re

import sqlglot
from sqlglot import exp
from sklearn.feature_extraction.text import TfidfVectorizer

from tarqa.reasoning.utils import digest, save_json
from tarqa.utils.io import read_json
from tarqa.reasoning.sql_utils import extract_sql
from tarqa.reasoning.sql_utils import sql_supervision

UPSTREAM_COMMIT = "0474801616130413b024fd008b016a190684649e"
STAGES = {"link": "SCHEMA_LINKING_TEMPLATE", "classify": "CLASSIFICATION_TEMPLATE",
          "EASY": "EASY_CLASS_TEMPLATE", "NON-NESTED": "NON_NESTED_CLASS_TEMPLATE",
          "NESTED": "NESTED_CLASS_TEMPLATE", "correct": "SELF_CORRECTION_PROMPT"}


def official_templates():
    """Load the packaged runtime prompts without importing upstream scripts."""
    record = read_json(Path(__file__).resolve().parent / "prompts/din_sql.json")
    values, provenance = record["templates"], record["upstream"]
    if provenance["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Unexpected DIN-SQL upstream revision")
    if digest(values) != "48615240272368e280385f5913661a57b18785b5f971736766572cf88b6b5c63":
        raise ValueError("DIN-SQL runtime prompts changed")
    if any(f"{role}_{name}" not in values for name in STAGES.values() for role in ("SYSTEM", "HUMAN")):
        raise ValueError("Official DIN-SQL templates are incomplete")
    return values, provenance


def annotate_training_sql(sql, catalog):
    """Annotation uses TRAIN gold only, including real resolved schema identities."""
    supervision = sql_supervision(sql, catalog)
    tree = sqlglot.parse_one(sql, read="duckdb")
    nested = len(list(tree.find_all(exp.Select))) > 1 or bool(list(tree.find_all(exp.SetOperation)))
    joined = bool(list(tree.find_all(exp.Join)))
    label = "NESTED" if nested else "NON-NESTED" if joined else "EASY"
    links = []
    for tid, (table, spec) in enumerate(catalog.tables.items()):
        if supervision["table_labels"][tid]:
            links.append(table)
        links += [f"{table}.{column}" for cid, column in enumerate(spec.columns)
                  if supervision["column_labels"][tid][cid]]
    links += [node.sql(dialect="duckdb") for node in tree.find_all(exp.Literal)]
    links += [join.args["on"].sql(dialect="duckdb") for join in tree.find_all(exp.Join) if join.args.get("on")]
    subqueries = ["Compute the relation defined by: " + node.sql(dialect="duckdb")
                  for node in list(tree.find_all(exp.Select))[1:]]
    # Demonstration candidate only: add one clearly unrequested output literal.
    # The correction target remains the ORIGINAL executed training SQL.
    corrupted = tree.copy()
    outer = corrupted if isinstance(corrupted, exp.Select) else corrupted.find(exp.Select)
    outer.append("expressions", exp.alias_(exp.Literal.number(1), "unrequested_extra_output"))
    return dict(label=label, links=list(dict.fromkeys(links)), sub_questions=subqueries,
                candidate_sql=corrupted.sql(dialect="duckdb"))


class DemoBank:
    def __init__(self, records, catalog, executor, *, num_examples=3):
        if num_examples < 1:
            raise ValueError("DIN-SQL requires at least one training demonstration")
        self.num_examples = int(num_examples)
        self.rows = []
        seen = set()
        for q in records:
            if q.domain != catalog.domain or q.query_type == "Predictive" or q.uid in seen:
                raise ValueError("DIN demonstration bank must contain unique same-domain Reasoning TRAIN QA")
            seen.add(q.uid)
            executor.execute(q.golden_sql)  # Do not fabricate labels or accept invalid gold SQL.
            self.rows.append(dict(uid=q.uid, question=q.primary_question, sql=q.golden_sql,
                                  **annotate_training_sql(q.golden_sql, catalog)))
        if not self.rows:
            raise ValueError("DIN-SQL has no permitted Reasoning training examples")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, token_pattern=r"(?u)\b\w+\b")
        self.matrix = self.vectorizer.fit_transform([r["question"] for r in self.rows])
        self.signature = digest(self.rows)

    def select(self, question, label=None):
        scores = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        indices = sorted(range(len(self.rows)), key=lambda i: (
            0 if label is None or self.rows[i]["label"] == label else 1, -float(scores[i]), self.rows[i]["uid"]))
        return [self.rows[i] for i in indices[:self.num_examples]]


def demonstration(row, stage):
    prefix = "Q: " + row["question"] + "\n"
    links = "Schema_links: " + json.dumps(row["links"], ensure_ascii=False)
    if stage == "link":
        return prefix + links
    if stage == "classify":
        return (prefix + links + "\nThe SQL structure requires " + row["label"] + ".\n" +
                "sub_questions: " + json.dumps(row["sub_questions"], ensure_ascii=False) +
                '\nLabel: "' + row["label"] + '"')
    if stage == "correct":
        return (prefix + "SQL: " + row["candidate_sql"] +
                "\nRemove the additional output that the question did not request.\nRevised_SQL: " + row["sql"])
    analysis = ""
    if stage != "EASY":
        analysis = "\nResolve the listed columns, join conditions and values."
    if stage == "NESTED":
        analysis += "\nsub_questions: " + json.dumps(row["sub_questions"], ensure_ascii=False)
    return prefix + links + analysis + "\nSQL: " + row["sql"]


def parse_links(text):
    match = re.search(r"schema[_ ]links\s*:\s*(\[[\s\S]*?\])", text, flags=re.I)
    if not match:
        raise ValueError("DIN schema-link module did not produce Schema_links: [...] ")
    return match.group(1)


def parse_classification(text):
    match = re.search(r'Label\s*:\s*["\']?(NON-NESTED|NESTED|EASY)\b', text, flags=re.I)
    if not match:
        raise ValueError("DIN classification module did not produce a valid Label")
    label = match.group(1).upper()
    sub = re.search(r"(?:sub[_ -]questions|subquestions|questions)\s*(?:=|:)\s*(\[[\s\S]*?\])", text, flags=re.I)
    if label == "NESTED" and not sub:
        raise ValueError("DIN NESTED classification requires explicit sub_questions")
    return label, sub.group(1) if sub else "[]"


def parse_sql(text, *, corrected=False):
    marker = r"Revised[_ ]SQL\s*:\s*" if corrected else r"(?<![\w_])SQL\s*:\s*"
    positions = list(re.finditer(marker, text, flags=re.I))
    return extract_sql(text[positions[-1].end():] if positions else text)


class DINSQLEngine:
    def __init__(self, client, bank, schema, *, trace_root):
        self.client, self.bank, self.schema = client, bank, schema
        self.templates, self.provenance = official_templates()
        self.trace_root = Path(trace_root)

    def messages(self, stage, question, examples, **fields):
        key = STAGES[stage]
        system = self.templates["SYSTEM_" + key]
        system += "\nUse only read-only DuckDB SQL. The schema below also applies to every demonstration."
        user = "Database schema:\n" + self.schema + "\n\nSame-domain training examples:\n"
        user += "\n\n".join(demonstration(row, stage) for row in examples)
        values = dict(schema=self.schema, columns_descriptions="", question=question, hint="",
                      schema_links="[]", sub_questions="[]", sql_query="", **{})
        values.update(fields)
        user += "\n\nCurrent question:\n" + self.templates["HUMAN_" + key].format(**values)
        if stage == "link":
            user += "\nEnd with Schema_links: [...] ."
        elif stage == "classify":
            user += '\nEnd with sub_questions: ["..."] (empty for non-nested), then Label: "EASY", "NON-NESTED", or "NESTED".'
        else:
            user += "\nEnd with " + ("Revised_SQL:" if stage == "correct" else "SQL:") + " and one complete query in a ```sql``` block."
        return [dict(role="system", content=system), dict(role="user", content=user)]

    def call(self, uid, stage, question, **fields):
        examples = self.bank.select(question, stage if stage in {"EASY", "NON-NESTED", "NESTED"} else None)
        messages = self.messages(stage, question, examples, **fields)
        identity = digest(dict(messages=messages, bank=self.bank.signature, uid=uid, stage=stage))
        path = self.trace_root / digest(uid)[:20] / f"{stage}.json"
        if path.exists():
            record = read_json(path)
            if record["request_sha256"] != identity or record["response_sha256"] != digest(record["response"]):
                raise RuntimeError("DIN stage checkpoint changed; refusing incompatible resume")
        else:
            response = self.client.generate(messages, stage=stage)
            record = dict(uid=uid, stage=stage, request_sha256=identity, response=response,
                          response_sha256=digest(response), demo_ids=[r["uid"] for r in examples], messages=messages)
            save_json(path, record)
        if record["response"].get("truncated"):
            raise ValueError(f"DIN {stage} generation reached the output limit without EOS")
        return record["response"]["text"], record

    def generate(self, uid, question):
        trace = []
        text, record = self.call(uid, "link", question); trace.append(record)
        links = parse_links(text)
        text, record = self.call(uid, "classify", question, schema_links=links); trace.append(record)
        label, sub = parse_classification(text)
        text, record = self.call(uid, label, question, schema_links=links, sub_questions=sub); trace.append(record)
        candidate = parse_sql(text)
        text, record = self.call(uid, "correct", question, sql_query=candidate); trace.append(record)
        return parse_sql(text, corrected=True), dict(predicted_class=label, predicted_links=links,
                                                    candidate_sql=candidate, stages=trace)
