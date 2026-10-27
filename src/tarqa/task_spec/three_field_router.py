"""Unadapted LLM routing interface, without learned table selection."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tarqa.types import TaskSpec, TaskType, Track
from tarqa.utils.text import extract_json_object


SYSTEM_PROMPT = """You are the task router for a relational-database QA system.
Read the natural-language question and the complete database schema. Return exactly one JSON object and no explanation.
The model-facing routing decision contains exactly three fields:
1. reference_time: required only for Prediction, otherwise null;
2. task_type: Classification or Regression for Prediction, otherwise null;
3. track: Reasoning or Prediction.
Reasoning means the answer is deterministically obtainable from the currently observed database state. Prediction means the answer concerns an unobserved future state.
For predictive questions, Boolean answers map to Classification and numerical scalar answers map to Regression.
Do not output relevant_tables or any other field. The downstream executor uses the complete schema.
"""


class ThreeFieldDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_time: str | None
    task_type: TaskType | None
    track: Track

    @field_validator("reference_time")
    @classmethod
    def canonical_time(cls, value):
        return TaskSpec.canonical_time(value)

    @model_validator(mode="after")
    def check_dependencies(self):
        errors = self.graph_spec().validate_dependencies()
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def graph_spec(self):
        # Empty relevant_tables is an INTERNAL adapter for the existing graph API.
        # full_schema ignores this list; no table list is predicted or supplied as gold.
        return TaskSpec(relevant_tables=[], reference_time=self.reference_time,
                        task_type=self.task_type, track=self.track)


def parse_decision(text):
    payload = extract_json_object(text)
    if set(payload) != {"reference_time", "task_type", "track"}:
        raise ValueError("Router must generate exactly reference_time, task_type, track")
    return ThreeFieldDecision.model_validate(payload)


def user_prompt(schema, question):
    return (f"Schema:\n{schema}\n\nQuestion:\n{question}\n\n"
            "Generate fields in this order: reference_time, task_type, track.\n\nRouting decision:")


def question_representation_input(question, decision):
    return (f"Question: {question}\nReference time: {decision.reference_time}\n"
            f"Task type: {decision.task_type.value if decision.task_type else None}\n"
            f"Track: {decision.track.value}")
