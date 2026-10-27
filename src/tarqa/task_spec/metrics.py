from __future__ import annotations

from collections import defaultdict
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from tarqa.types import TaskSpec, Track


def _table_scores(gold: list[list[str]], pred: list[list[str]]) -> dict[str, float]:
    labels = sorted({x for rows in gold + pred for x in rows})
    if not labels:
        return {"table_precision": 1.0, "table_recall": 1.0, "table_f1": 1.0, "table_exact": 1.0}
    # Count table memberships directly; a one-column indicator matrix can
    # otherwise be interpreted as binary classification and count true negatives.
    hits = sum(len(set(g) & set(p)) for g, p in zip(gold, pred))
    predicted = sum(len(set(row)) for row in pred)
    expected = sum(len(set(row)) for row in gold)
    p = hits / predicted if predicted else 0.0
    r = hits / expected if expected else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    exact = sum(set(g) == set(p_) for g, p_ in zip(gold, pred)) / len(gold)
    return {"table_precision": p, "table_recall": r, "table_f1": f, "table_exact": exact}


def task_spec_metrics(gold: list[TaskSpec], pred: list[TaskSpec | None]) -> dict[str, Any]:
    if len(gold) != len(pred):
        raise ValueError("Gold and prediction lengths differ.")
    if not gold:
        raise ValueError("Cannot evaluate an empty Task Spec dataset.")
    # A malformed or inconsistent generation must never become a correct
    # Reasoning prediction merely because a fallback fills in null fields.
    pred = [p if p is not None and not p.validate_dependencies() else None for p in pred]
    result = _table_scores(
        [x.relevant_tables for x in gold], [x.relevant_tables if x else [] for x in pred]
    )
    result["table_exact"] = sum(
        p is not None and set(g.relevant_tables) == set(p.relevant_tables)
        for g, p in zip(gold, pred)
    ) / len(gold)
    if not any(p is not None for p in pred):
        result.update(table_precision=0.0, table_recall=0.0, table_f1=0.0)
    gold_track = [x.track.value for x in gold]
    pred_track = [x.track.value if x else "INVALID" for x in pred]
    result["track_accuracy"] = accuracy_score(gold_track, pred_track)
    result["track_macro_f1"] = f1_score(
        gold_track, pred_track,
        labels=sorted(set(gold_track) | (set(pred_track) - {"INVALID"})),
        average="macro", zero_division=0,
    )
    gold_type = [x.task_type.value if x.task_type else "null" for x in gold]
    pred_type = ["INVALID" if x is None else (x.task_type.value if x.task_type else "null") for x in pred]
    result["task_type_accuracy"] = accuracy_score(gold_type, pred_type)
    result["reference_time_accuracy"] = sum(
        p is not None and g.reference_time == p.reference_time for g, p in zip(gold, pred)
    ) / len(gold)
    result["full_spec_exact"] = sum(
        (p is not None and set(g.relevant_tables)==set(p.relevant_tables) and g.reference_time==p.reference_time
         and g.track==p.track and g.task_type==p.task_type) for g,p in zip(gold,pred)
    ) / len(gold)
    prediction_pairs = [(g, p) for g, p in zip(gold, pred) if g.track == Track.PREDICTION]
    for field in ("task_type", "reference_time"):
        result[f"prediction_{field}_accuracy"] = (
            sum(p is not None and getattr(g, field) == getattr(p, field) for g, p in prediction_pairs)
            / len(prediction_pairs) if prediction_pairs else None
        )
    result = {key: float(value) if value is not None else None for key, value in result.items()}
    result["num_examples"] = len(gold)
    result["num_prediction_examples"] = len(prediction_pairs)
    result["invalid_generation_rate"] = sum(p is None for p in pred) / len(pred)
    return result


def paraphrase_consistency(rows: list[dict[str, Any]]) -> float | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(row)
    valid = [items for items in grouped.values() if len(items) > 1]
    if not valid:
        return None
    consistent = 0
    for items in valid:
        first = items[0]["prediction"]
        consistent += (
            first is not None
            and all(not item.get("dependency_errors") and item["prediction"] == first for item in items)
        )
    return consistent / len(valid)
