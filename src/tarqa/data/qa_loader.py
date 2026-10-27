from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tarqa.types import EntityAnchor
from tarqa.utils.io import read_json


@dataclass(frozen=True)
class QARecord:
    domain: str
    payload: dict[str, Any]

    @property
    def instance_id(self) -> str:
        return str(self.payload["instance_id"])

    @property
    def uid(self) -> str:
        return f"{self.domain}::{self.instance_id}"

    @property
    def query_type(self) -> str:
        return str(self.payload["metadata"]["query_type"])

    @property
    def answer_nature(self) -> str:
        return str(self.payload["metadata"]["answer_nature"])

    @property
    def relevant_tables(self) -> list[str]:
        return list(self.payload["metadata"]["complexity"].get("tables_involved", []))

    @property
    def reference_time(self) -> str | None:
        value = self.payload["metadata"].get("reference_time")
        return str(value) if value not in (None, "") else None

    @property
    def primary_question(self) -> str:
        return str(self.payload["question"]["primary_nl"])

    @property
    def variants(self) -> list[str]:
        return [str(x) for x in self.payload["question"].get("nl_variants", [])]

    @property
    def all_questions(self) -> list[str]:
        return [self.primary_question, *self.variants]

    @property
    def golden_sql(self) -> str:
        return str(self.payload["evidence"]["golden_sql"])

    @property
    def raw_answer(self) -> Any:
        return self.payload["ground_truth"].get("raw_value")

    @property
    def target_instances(self) -> list[EntityAnchor]:
        raw = self.payload.get("target_instances", [])
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise ValueError(f"target_instances must be a list: {self.uid}")
        return [EntityAnchor.model_validate(x) for x in raw]

    @property
    def anchor(self) -> EntityAnchor:
        anchors = self.target_instances
        if self.query_type != "Predictive":
            raise ValueError(f"Only Predictive records have anchors: {self.uid}")
        if len(anchors) != 1:
            raise ValueError(
                f"Final Prediction scope requires exactly one target_instances entry: "
                f"{self.uid} has {len(anchors)}"
            )
        return anchors[0]


def load_qa_records(path: str | Path, domain: str) -> list[QARecord]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"QA file must contain a JSON list: {path}")
    return [QARecord(domain=domain, payload=row) for row in data]


def iter_qa_records(path: str | Path, domain: str) -> Iterator[QARecord]:
    yield from load_qa_records(path, domain)
