from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tarqa.data.discovery import DomainPaths
from tarqa.data.releases import ReleaseDataset
from tarqa.utils.io import write_jsonl


VERBALIZER_SYSTEM = (
    "Convert a canonical relational-database answer into one concise natural-language answer. "
    "Never alter, round, omit, or invent values."
)


def build_verbalizer_dataset(paths: dict[str, DomainPaths], output_path: str | Path, *, split="train") -> int:
    if split not in {"train", "validation"}:
        raise ValueError("Verbalizer training cannot read qa/test")
    rows: list[dict[str, Any]] = []
    for domain, domain_paths in paths.items():
        for record in ReleaseDataset(domain_paths).qa(split):
            if record.query_type == "Predictive":
                continue  # original QA future answers are never a training source
            target = record.payload["ground_truth"].get("formatted_nl_answer")
            if not target:
                continue
            user = (
                f"Question: {record.primary_question}\n"
                f"Canonical answer: {json.dumps(record.raw_answer, ensure_ascii=False, default=str)}"
            )
            rows.append(
                {
                    "uid": record.uid,
                    "domain": domain,
                    "system_prompt": VERBALIZER_SYSTEM,
                    "user_prompt": user,
                    "assistant_text": str(target),
                }
            )
    write_jsonl(output_path, rows)
    return len(rows)
