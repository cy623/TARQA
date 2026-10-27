"""Execution evaluation shared by epoch validation and final testing."""
from collections import defaultdict
from decimal import Decimal
import math

from tarqa.reasoning.executor import DuckDBReadOnlyExecutor


def canonical(value):
    if isinstance(value, (tuple, list)):
        return tuple(canonical(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, canonical(v)) for k, v in value.items()))
    if isinstance(value, (float, Decimal)):
        value = float(value)
        if math.isnan(value):
            return ("nan",)
        return round(value, 8)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def same_rows(actual, expected, answer_nature):
    actual, expected = canonical(actual), canonical(expected)
    if answer_nature.lower() in {"set", "sets"}:
        return set(actual) == set(expected)
    return actual == expected


def select_executable(candidates, executor):
    errors = []
    for attempt, sql in enumerate(candidates, 1):
        try:
            columns, rows = executor.execute(sql)
            return sql, columns, rows, attempt
        except Exception as exc:
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                raise
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError(f"All candidates failed execution: {errors[:3]}")


def evaluate_ranked(rows, input_texts, model, database_paths, *, timeout_seconds=120, gold_cache=None):
    """Only predicted schema rankings are supplied by the caller for validation.

    Gold execution is scoring only and cannot influence candidate selection.
    Bad gold SQL raises a data error instead of being hidden as a model failure.
    """
    if len(rows) != len(input_texts):
        raise ValueError("Validation input count differs from QA count")
    outputs = []
    for record, text in zip(rows, input_texts):
        executor = DuckDBReadOnlyExecutor(database_paths[record["domain"]], timeout_seconds=timeout_seconds)
        expected = gold_cache[record["uid"]] if gold_cache is not None else executor.execute(record["golden_sql"])[1]
        item = dict(uid=record["uid"], domain=record["domain"], correct=False, executable=False)
        try:
            sql, _, actual, attempts = select_executable(model.generate_from_input(text), executor)
            item.update(sql=sql, executable=True, attempts=attempts,
                        correct=same_rows(actual, expected, record["answer_nature"]))
        except Exception as exc:
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                raise
            item["error"] = f"{type(exc).__name__}: {exc}"
        outputs.append(item)
    return summarize(outputs), outputs


def summarize(outputs):
    def metrics(items):
        n = len(items)
        return dict(total=n, correct=sum(r["correct"] for r in items),
                    execution_accuracy=sum(r["correct"] for r in items)/n if n else None,
                    executable_rate=sum(r.get("executable", False) for r in items)/n if n else None,
                    failure_count=sum(not r.get("executable", False) for r in items))
    groups = defaultdict(list)
    for item in outputs:
        groups[item["domain"]].append(item)
    result = metrics(outputs)
    result["metrics_by_domain"] = {d: metrics(items) for d, items in groups.items()}
    scores = [m["execution_accuracy"] for m in result["metrics_by_domain"].values() if m["total"]]
    result["domain_macro_execution_accuracy"] = sum(scores)/len(scores) if scores else None
    result["schema_policy"] = "full_schema_to_executor; no TSG table pruning"
    result["metric_protocol"] = "execution_v1_sets_unordered_other_answers_ordered_round8"
    return result
