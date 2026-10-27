from __future__ import annotations

import json
from dataclasses import dataclass

from tarqa.types import TaskSpec


SYSTEM_PROMPT = """You are the Task Spec Generator for a relational-database QA system.
Read the natural-language question and the complete database schema. Return exactly one JSON object and no explanation.
The model-facing Task Spec contains exactly four fields:
1. relevant_tables: schema tables needed by the question;
2. reference_time: required only for Prediction, otherwise null;
3. task_type: Classification or Regression for Prediction, otherwise null;
4. track: Reasoning or Prediction.
Reasoning means the answer is deterministically obtainable from the currently observed database state. Prediction means the answer concerns an unobserved future state.
For predictive questions, Boolean answers map to Classification and numerical scalar answers map to Regression.
Never invent a table not present in the schema.
"""


@dataclass(frozen=True)
class Demonstration:
    question: str
    task_spec: TaskSpec


def render_user_prompt(
    schema_text: str,
    question: str,
    demonstrations: list[Demonstration] | None = None,
    generation_strategy: str = "track_last",
) -> str:
    parts: list[str] = []
    if demonstrations:
        parts.append("Examples:")
        for idx, demo in enumerate(demonstrations, 1):
            spec = json.dumps(
                demo.task_spec.ordered_dict(generation_strategy),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            parts.append(f"Example {idx}\nQuestion: {demo.question}\nTask Spec: {spec}")
    parts.append("Schema:\n" + schema_text)
    parts.append("Question:\n" + question)
    if generation_strategy == "track_last":
        parts.append(
            "Generate fields in this order: relevant_tables, reference_time, task_type, track."
        )
    elif generation_strategy == "track_first":
        parts.append(
            "Generate fields in this order: track, relevant_tables, reference_time, task_type."
        )
    else:
        parts.append("Generate the complete JSON object in one pass.")
    parts.append("Task Spec:")
    return "\n\n".join(parts)
