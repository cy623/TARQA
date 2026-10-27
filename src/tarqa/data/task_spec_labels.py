from __future__ import annotations

from tarqa.data.qa_loader import QARecord
from tarqa.types import TaskSpec, TaskType, Track


REASONING_QUERY_TYPES = {"Retrieval", "Analytical", "Relational Reasoning"}


def track_from_query_type(query_type: str) -> Track:
    if query_type == "Predictive":
        return Track.PREDICTION
    if query_type in REASONING_QUERY_TYPES:
        return Track.REASONING
    raise ValueError(f"Unsupported query_type: {query_type}")


def task_type_from_answer_nature(answer_nature: str, track: Track) -> TaskType | None:
    if track == Track.REASONING:
        return None
    normalized = answer_nature.strip().lower()
    if normalized in {"boolean", "booleans"}:
        return TaskType.CLASSIFICATION
    if normalized in {"scalar", "scalars"}:
        return TaskType.REGRESSION
    raise ValueError(
        f"Predictive answer_nature={answer_nature!r} is outside the final paper scope "
        "(only Boolean and Scalars are supported)."
    )


def build_gold_task_spec(record: QARecord) -> TaskSpec:
    track = track_from_query_type(record.query_type)
    task_type = task_type_from_answer_nature(record.answer_nature, track)
    reference_time = record.reference_time if track == Track.PREDICTION else None
    spec = TaskSpec(
        relevant_tables=record.relevant_tables,
        reference_time=reference_time,
        task_type=task_type,
        track=track,
    )
    errors = spec.validate_dependencies()
    if errors:
        raise ValueError(f"Invalid gold Task Spec for {record.uid}: {errors}")
    return spec
