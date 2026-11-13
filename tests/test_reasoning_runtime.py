"""Regression checks for the self-contained SQL runtime; no model weights needed."""
import copy
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import duckdb

from tarqa.data.metadata import ColumnSchema, SchemaCatalog, TableSchema
from tarqa.model import _SharedBaseChat, _SQLModel
from tarqa.reasoning import din_sql
from tarqa.reasoning.pipeline import ReasoningPipeline
from tarqa.reasoning.schema import schema_ddl
from tarqa.types import TaskSpec


class RejectLegacyPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tarqa_baselines" or fullname.startswith("tarqa_baselines."):
            raise ImportError("The standalone runtime cannot use the legacy package")
        return None


class ScriptedSQLClient:
    """Fixed responses exercise routing, stage parsing, caching, and real SQL execution."""
    def __init__(self, truncated=False):
        self.calls = []
        self.truncated = truncated

    def generate(self, messages, *, stage="direct"):
        self.calls.append((stage, messages))
        responses = {
            "direct": "```sql\nSELECT SUM(amount) FROM orders WHERE seller_id = 1;\n```",
            "link": 'Schema_links: ["orders.seller_id", "orders.amount"]',
            "classify": 'sub_questions: []\nLabel: "EASY"',
            "EASY": "SQL: SELECT COUNT(*) + 1 FROM orders WHERE seller_id = 1;",
            "correct": "Revised_SQL: SELECT COUNT(*) FROM orders WHERE seller_id = 1;",
        }
        return {"text": responses[stage], "truncated": self.truncated}


class ReasoningRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocker = RejectLegacyPackage()
        sys.meta_path.insert(0, cls.blocker)

    @classmethod
    def tearDownClass(cls):
        sys.meta_path.remove(cls.blocker)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.database = self.directory / "test.duckdb"
        with duckdb.connect(str(self.database)) as connection:
            connection.execute("CREATE TABLE orders (seller_id BIGINT, amount DOUBLE)")
            connection.execute("INSERT INTO orders VALUES (1, 3.5), (1, 2.5), (2, 99.0)")
        self.catalog = SchemaCatalog("toy", "", {
            "orders": TableSchema("orders", columns={
                "seller_id": ColumnSchema("seller_id", "BIGINT"),
                "amount": ColumnSchema("amount", "DOUBLE"),
            }),
            "unused_marker_table": TableSchema("unused_marker_table", columns={
                "id": ColumnSchema("id", "BIGINT"),
            }),
        })
        self.spec = TaskSpec(track="Reasoning", relevant_tables=["orders"])
        self.bank = self.directory / "demonstrations.json"
        self.bank.write_text(json.dumps({"examples": [{
            "uid": "toy_train_1", "question": "How many orders did seller 2 receive?",
            "sql": "SELECT COUNT(*) FROM orders WHERE seller_id = 2",
            "label": "EASY", "links": ["orders.seller_id", "2"],
            "sub_questions": [],
            "candidate_sql": "SELECT COUNT(*), 1 FROM orders WHERE seller_id = 2",
        }]}))

    def pipeline(self, client, backend):
        sql = _SQLModel(client, backend, self.directory / "stages", self.bank, 1)
        sql.spec, sql.uid = self.spec, "toy_test_1"
        pipeline = ReasoningPipeline(self.catalog, self.database, sql,
            schema_policy="tsg_selected", max_repair_attempts=0,
            schema_renderer=schema_ddl)
        self.addCleanup(pipeline.close)
        return pipeline

    def test_direct_generation_executes_on_real_database(self):
        client = ScriptedSQLClient()
        result = self.pipeline(client, "direct").answer("Seller 1 total revenue?", self.spec)
        self.assertEqual(result.rows, [(6.0,)])
        self.assertEqual([stage for stage, _ in client.calls], ["direct"])
        self.assertNotIn("unused_marker_table", str(client.calls))

    def test_din_stages_correct_sql_and_reuse_validated_trace(self):
        client = ScriptedSQLClient()
        pipeline = self.pipeline(client, "din_sql")
        result = pipeline.answer("Seller 1 order count?", self.spec)
        self.assertEqual(result.rows, [(2,)])
        self.assertEqual([stage for stage, _ in client.calls],
                         ["link", "classify", "EASY", "correct"])
        self.assertNotIn("unused_marker_table", str(client.calls))
        again = pipeline.answer("Seller 1 order count?", self.spec)
        self.assertEqual(again.rows, result.rows)
        self.assertEqual(len(client.calls), 4)

    def test_shared_model_chat_initializes_without_external_package(self):
        generator = SimpleNamespace(model=object(), tokenizer=object())
        chat = _SharedBaseChat(generator, {"max_new_tokens": 32})
        self.assertIs(chat._client.model, generator.model)

    def test_packaged_prompts_are_complete_and_tampering_is_rejected(self):
        templates, _ = din_sql.official_templates()
        self.assertTrue(all(f"{role}_{name}" in templates
                            for name in din_sql.STAGES.values()
                            for role in ("SYSTEM", "HUMAN")))
        record = json.loads((ROOT / "src/tarqa/reasoning/prompts/din_sql.json").read_text())
        changed = copy.deepcopy(record)
        changed["templates"]["SYSTEM_SCHEMA_LINKING_TEMPLATE"] += " changed"
        with patch.object(din_sql, "read_json", return_value=changed):
            with self.assertRaisesRegex(ValueError, "prompts changed"):
                din_sql.official_templates()

    def test_truncated_generation_is_reported(self):
        with self.assertRaisesRegex(ValueError, "output token limit"):
            self.pipeline(ScriptedSQLClient(truncated=True), "direct").answer(
                "Seller 1 revenue?", self.spec)


if __name__ == "__main__":
    unittest.main()
