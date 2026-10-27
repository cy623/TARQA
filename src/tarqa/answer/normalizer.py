from __future__ import annotations

from typing import Any

from tarqa.types import CanonicalAnswer


def _unwrap_singleton(rows: list[tuple[Any, ...]]) -> Any:
    if len(rows) == 1 and len(rows[0]) == 1:
        return rows[0][0]
    return [list(row) for row in rows]


def normalize_sql_result(
    rows: list[tuple[Any, ...]],
    *,
    answer_nature: str | None = None,
) -> CanonicalAnswer:
    nature = (answer_nature or "").strip().lower()
    if nature in {"boolean", "booleans"}:
        value = _unwrap_singleton(rows)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"yes", "true", "1"}:
                value = True
            elif lowered in {"no", "false", "0"}:
                value = False
        return CanonicalAnswer(answer_type="Boolean", value=bool(value))
    if nature in {"scalar", "scalars"}:
        return CanonicalAnswer(answer_type="Scalar", value=_unwrap_singleton(rows))
    if nature in {"ranking", "rankings"}:
        return CanonicalAnswer(answer_type="Ranking", value=[list(row) for row in rows])
    if nature in {"set", "sets"}:
        return CanonicalAnswer(answer_type="Set", value=[list(row) for row in rows])

    if len(rows) == 1 and len(rows[0]) == 1:
        value = rows[0][0]
        if isinstance(value, bool):
            return CanonicalAnswer(answer_type="Boolean", value=value)
        return CanonicalAnswer(answer_type="Scalar", value=value)
    return CanonicalAnswer(answer_type="Set", value=[list(row) for row in rows])


def normalize_prediction(
    *,
    value: Any,
    task_type: str,
    probability: float | None = None,
    unit: str | None = None,
) -> CanonicalAnswer:
    if task_type == "Classification":
        return CanonicalAnswer(
            answer_type="Boolean",
            value=bool(value),
            probability=probability,
            unit=unit,
        )
    if task_type == "Regression":
        return CanonicalAnswer(answer_type="Scalar", value=float(value), unit=unit)
    raise ValueError(f"Unsupported task type for prediction normalization: {task_type}")
