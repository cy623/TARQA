"""Published-data contract; model inputs never contain label SQL or future answers."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import hashlib
import math
from pathlib import Path

from tarqa.data.qa_loader import QARecord
from tarqa.data.task_spec_labels import build_gold_task_spec
from tarqa.types import EntityAnchor, TaskSpec, TaskType, Track
from tarqa.utils.io import read_json

SPLITS = ("train", "validation", "test")
SUPPORT_FIELDS = {"sample_id", "domain", "parent_question_id", "parent_split", "question",
                  "anchor", "reference_time", "task_type", "relevant_tables", "label"}


@lru_cache(maxsize=256)
def _file_hash(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hash(path):
    info = Path(path).stat()
    return _file_hash(str(Path(path).resolve()), info.st_size, info.st_mtime_ns)


def stamp(value):
    return datetime.fromisoformat(str(value)).isoformat(sep=" ", timespec="seconds")


def scalar_label(value, task_type):
    if isinstance(value, dict) and "values" in value:
        if value.get("is_truncated"):
            raise ValueError("A scalar prediction target cannot be truncated")
        value = value["values"]
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, str) and task_type == "Classification":
        value = {"true": 1, "false": 0}.get(value.lower(), value)
    if isinstance(value, (list, dict, tuple)) or value is None:
        raise ValueError(f"Expected an observed scalar label, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite label")
    if task_type == "Classification" and result not in (0, 1):
        raise ValueError(f"Invalid Boolean label: {value}")
    return result


class ReleaseDataset:
    def __init__(self, paths, *, validate_rows=True):
        if paths.release_root is None:
            raise ValueError("Published QA/support release required")
        self.paths = paths
        self.validate_rows = validate_rows
        self.root = Path(paths.release_root)
        self.domain = paths.domain.lower()
        self.config = read_json(self.root / "config.json")
        self._qa, self._support, self._times = {}, {}, {}
        times = self.root / "qa_reference_times.json"
        if times.exists():
            self._times = {r["parent_question_id"]: r["model_reference_time"] for r in read_json(times)}
        elif self.domain == "ncaa":
            raise FileNotFoundError("NCAA requires qa_reference_times.json; never guess its cutoff")

    def qa(self, split):
        if split not in SPLITS:
            raise ValueError(split)
        if split not in self._qa:
            rows = read_json(self.root / "qa" / f"{split}.json")
            if self.validate_rows and not isinstance(rows, list):
                raise ValueError("QA must be a JSON array")
            records = []
            for raw in rows:
                row = deepcopy(raw)
                if row["metadata"]["query_type"] == "Predictive":
                    ref = self._times.get(row["instance_id"], row["metadata"].get("reference_time"))
                    if self.domain == "ncaa" and row["instance_id"] not in self._times:
                        raise ValueError(f"Missing NCAA canonical time: {row['instance_id']}")
                    row["metadata"]["reference_time"] = stamp(ref)
                records.append(QARecord(self.domain, row))
            self._qa[split] = records
        return self._qa[split]

    def support(self, split):
        if split not in SPLITS:
            raise ValueError(split)
        if split not in self._support:
            path = self.root / "support" / f"{split}.json"
            if not path.exists():
                if self.config.get("has_prediction_tasks") is False:
                    self._support[split] = []
                    return []
                raise FileNotFoundError(path)
            rows = read_json(path)
            if not self.validate_rows:
                self._support[split] = rows
                return rows
            if not isinstance(rows, list):
                raise ValueError("support must be a JSON array")
            parents = {r.instance_id: r for r in self.qa(split)}
            for row in rows:
                if set(row) != SUPPORT_FIELDS:
                    raise ValueError(f"Unexpected support fields: {row.get('sample_id')}")
                if row["domain"] != self.domain or row["parent_split"] != split:
                    raise ValueError(f"Wrong support domain/split: {row['sample_id']}")
                parent = parents.get(row["parent_question_id"])
                if parent is None or parent.query_type != "Predictive":
                    raise ValueError(f"Orphan support: {row['sample_id']}")
                spec = build_gold_task_spec(parent)
                if row["task_type"] != spec.task_type.value:
                    raise ValueError("Support/parent task type mismatch")
                if set(row["relevant_tables"]) != set(spec.relevant_tables):
                    raise ValueError("Support/parent relevant tables mismatch")
                anchor = EntityAnchor.model_validate(row["anchor"])
                if (anchor.table, anchor.key) != (parent.anchor.table, parent.anchor.key):
                    raise ValueError("Support changed the parent's anchor role")
                if split == "test" and self.config.get("support_test_usage") == "evaluation_only":
                    minimum = self.config["support_test_reference_time_min"]
                    if stamp(row["reference_time"]) < stamp(minimum):
                        raise ValueError("Evaluation support precedes training/validation label availability")
                elif stamp(row["reference_time"]) >= parent.reference_time:
                    raise ValueError("Support must precede its original QA cutoff")
                scalar_label(row["label"], row["task_type"])
            self._support[split] = rows
        return self._support[split]

    def adaptation_support(self, split="test"):
        """Keep evaluation-only outcome labels out of episodic adaptation."""
        if split == "test" and self.config.get("support_test_usage") == "evaluation_only":
            raise ValueError("This support/test release is evaluation-only; use support/train for adaptation")
        return self.prediction_rows("support/" + split)

    def prediction_rows(self, role):
        if role.startswith("support/"):
            split = role.split("/")[1]
            rows = []
            for s in self.support(split):
                r = dict(s)
                r["role"] = role
                evaluation_only = split == "test" and self.config.get("support_test_usage") == "evaluation_only"
                r["uid"] = f"{self.domain}::{s['sample_id'] if evaluation_only else s['parent_question_id']}"
                if evaluation_only:
                    r["parent_uid"] = f"{self.domain}::{s['parent_question_id']}"
                r["spec"] = TaskSpec(relevant_tables=s["relevant_tables"],
                    reference_time=stamp(s["reference_time"]), task_type=TaskType(s["task_type"]),
                    track=Track.PREDICTION).model_dump(mode="json")
                r["label"] = scalar_label(s["label"], s["task_type"])
                rows.append(r)
            return rows
        if role != "qa/test":
            raise ValueError("Head data can only use support/* or original qa/test")
        rows = []
        for q in self.qa("test"):
            if q.query_type != "Predictive":
                continue
            spec = build_gold_task_spec(q)
            rows.append(dict(sample_id=q.uid, uid=q.uid, domain=self.domain,
                parent_question_id=q.instance_id, parent_split="test", role=role,
                question=q.primary_question, anchor=q.anchor.model_dump(mode="json"),
                reference_time=q.reference_time, task_type=spec.task_type.value,
                relevant_tables=spec.relevant_tables, spec=spec.model_dump(mode="json"),
                label=scalar_label(q.raw_answer, spec.task_type.value)))
        return rows

    def validate(self, *, require_checksums=True):
        seen_qa, seen_support = set(), set()
        mapping = {(r["parent_split"], r["parent_question_id"]): r
                   for r in read_json(self.root / "question_to_support.json")}
        report = {"domain": self.domain, "qa": {}, "support": {}, "prediction_qa": {}}
        for split in SPLITS:
            qa, support = self.qa(split), self.support(split)
            for q in qa:
                if q.uid in seen_qa:
                    raise ValueError(f"QA split overlap: {q.uid}")
                seen_qa.add(q.uid)
                build_gold_task_spec(q)
                if q.query_type == "Predictive":
                    _ = q.anchor
            counts = Counter(r["parent_question_id"] for r in support)
            ids_by_parent = {}
            for s in support:
                if s["sample_id"] in seen_support:
                    raise ValueError("Duplicate support sample_id")
                seen_support.add(s["sample_id"])
                ids_by_parent.setdefault(s["parent_question_id"], set()).add(s["sample_id"])
            for q in qa:
                m = mapping[(split, q.instance_id)]
                if counts[q.instance_id] != m["support_count"]:
                    raise ValueError(f"Wrong support count: {q.uid}")
                if set(m["sample_ids"]) != ids_by_parent.get(q.instance_id, set()):
                    raise ValueError("Support mapping IDs differ")
                minimum = 0 if split == "test" and self.config.get("support_test_allow_missing_parents") else 1
                if q.query_type == "Predictive" and not minimum <= counts[q.instance_id] <= 10:
                    raise ValueError("Retained prediction parent must have 1–10 supports")
            report["qa"][split] = len(qa)
            report["support"][split] = len(support)
            report["prediction_qa"][split] = sum(q.query_type == "Predictive" for q in qa)
        if len(mapping) != len(seen_qa):
            raise ValueError("Mapping contains unknown QA parents")
        if seen_support:
            audit_path = self.root / "audit/sample_executions.json"
            if not audit_path.exists():
                raise FileNotFoundError("Use the full release: execution/availability audit is required")
            audits = {r["sample_id"]: r for r in read_json(audit_path)}
            for split in SPLITS:
                parents = {r.instance_id: r for r in self.qa(split)}
                for row in self.support(split):
                    audit = audits[row["sample_id"]]
                    if (audit["parent_question_id"], audit["parent_split"]) != (row["parent_question_id"], split):
                        raise ValueError("Support execution audit has a different parent/split")
                    label = scalar_label(row["label"], row["task_type"])
                    if label != scalar_label(audit["label"], row["task_type"]):
                        raise ValueError("Label differs from execution audit")
                    if label != scalar_label(audit["raw_result"], row["task_type"]):
                        raise ValueError("Label differs from recorded raw SQL result")
                    available = stamp(audit["label_available_at"])
                    cutoff = stamp(audit["label_observation_cutoff"])
                    evaluation_only = split == "test" and self.config.get("support_test_usage") == "evaluation_only"
                    if available > cutoff or (not evaluation_only and cutoff > parents[row["parent_question_id"]].reference_time):
                        raise ValueError("Support label unavailable at its allowed training/adaptation cutoff")
            report["support_execution_audits_checked"] = len(seen_support)
        files = [self.root / "config.json", self.paths.metadata, self.root / "question_to_support.json"]
        if seen_support:
            files.append(self.root / "audit/sample_executions.json")
        files += [self.root / "qa" / f"{s}.json" for s in SPLITS]
        files += [self.root / "support" / f"{s}.json" for s in SPLITS if (self.root / "support" / f"{s}.json").exists()]
        if self._times:
            files.append(self.root / "qa_reference_times.json")
        report["input_sha256"] = {str(p.relative_to(self.root)): file_hash(p) for p in files}
        actual_db_hash = file_hash(self.paths.database)
        report["database_sha256"] = actual_db_hash
        checksums_path = self.root / "checksums.json"
        if not require_checksums and not checksums_path.exists():
            report["checksum_validation"] = "skipped_missing_manifest"
            return report
        expected = read_json(checksums_path)
        for name, digest in report["input_sha256"].items():
            if name not in expected or expected[name]["sha256"] != digest:
                raise ValueError(f"Published release file changed: {name}")
        database_entries = [value for name, value in expected.items() if name.endswith(".duckdb")]
        if len(database_entries) != 1 or database_entries[0]["sha256"] != actual_db_hash:
            raise ValueError("Database does not match the audited release snapshot")
        report["checksum_validation"] = "verified"
        return report
