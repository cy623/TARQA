from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime, timezone


class Track(str, Enum):
    REASONING = "Reasoning"
    PREDICTION = "Prediction"


class TaskType(str, Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"


class TaskSpec(BaseModel):
    """The four-field model-facing Task Spec used by the paper."""

    model_config = ConfigDict(extra="forbid")
    relevant_tables: list[str] = Field(default_factory=list)
    reference_time: str | None = None
    task_type: TaskType | None = None
    track: Track

    @field_validator("reference_time")
    @classmethod
    def canonical_time(cls, value):
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed.isoformat(sep=" ", timespec="seconds")

    @field_validator("relevant_tables")
    @classmethod
    def unique_tables(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    def validate_dependencies(self) -> list[str]:
        errors: list[str] = []
        if self.track == Track.PREDICTION:
            if self.reference_time is None:
                errors.append("Prediction requires reference_time.")
            if self.task_type is None:
                errors.append("Prediction requires task_type.")
        else:
            if self.task_type is not None:
                errors.append("Reasoning must have task_type=null.")
            if self.reference_time is not None:
                errors.append("Reasoning must have reference_time=null.")
        return errors

    def ordered_dict(self, strategy: str = "track_last") -> dict[str, Any]:
        fields: dict[str, Any] = {
            "relevant_tables": self.relevant_tables,
            "reference_time": self.reference_time,
            "task_type": self.task_type.value if self.task_type else None,
            "track": self.track.value,
        }
        if strategy in {"track_last", "direct"}:
            return fields
        if strategy == "track_first":
            return {
                "track": fields["track"],
                "relevant_tables": fields["relevant_tables"],
                "reference_time": fields["reference_time"],
                "task_type": fields["task_type"],
            }
        raise ValueError(f"Unknown generation strategy: {strategy}")


class EntityAnchor(BaseModel):
    """Offline benchmark anchor metadata; never part of TaskSpec."""

    table: str
    key: str
    value: Any


class BindingResult(BaseModel):
    """Resolved single-anchor binding consumed by Prediction aggregation."""

    anchors: list[EntityAnchor]
    source: str = "dataset_target_instances"
    confidence: float = 1.0
    notes: list[str] = Field(default_factory=list)

    @field_validator("anchors")
    @classmethod
    def exactly_one_anchor(cls, value: list[EntityAnchor]) -> list[EntityAnchor]:
        if len(value) != 1:
            raise ValueError("The final Prediction scope requires exactly one Anchor Entity.")
        return value


class CanonicalAnswer(BaseModel):
    answer_type: str
    value: Any
    probability: float | None = None
    unit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
