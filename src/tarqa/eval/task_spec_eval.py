from __future__ import annotations

from pathlib import Path
from typing import Any

from tarqa.task_spec.anonymization import TableAnonymization
from tarqa.task_spec.inference import LocalTaskSpecGenerator, TaskSpecOutputError
from tarqa.task_spec.metrics import paraphrase_consistency, task_spec_metrics
from tarqa.types import TaskSpec
from tarqa.utils.io import read_jsonl, write_json, write_jsonl
from tqdm.auto import tqdm


def save_task_spec_results(
    result_rows: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Any]:
    """Score saved generations, including invalid outputs, without another model run."""
    gold_specs = [TaskSpec.model_validate(row["gold"]) for row in result_rows]
    pred_specs = [
        TaskSpec.model_validate(row["prediction"])
        if row["prediction"] is not None and not row.get("dependency_errors") else None
        for row in result_rows
    ]
    metrics = task_spec_metrics(gold_specs, pred_specs)
    metrics["metric_protocol"] = "task_spec_v2_invalid_is_error"
    metrics["paraphrase_consistency"] = paraphrase_consistency(result_rows)
    metrics["paraphrase_status"] = (
        "not_evaluated_no_variants" if metrics["paraphrase_consistency"] is None else "evaluated"
    )
    output_dir = Path(output_dir)
    write_jsonl(output_dir / "predictions.jsonl", result_rows)
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def evaluate_task_spec_generator(
    generator: LocalTaskSpecGenerator,
    test_path: str | Path,
    output_dir: str | Path,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []

    rows = list(read_jsonl(test_path))
    if not rows:
        raise ValueError("Cannot evaluate an empty Task Spec dataset.")
    for row in tqdm(rows, desc="Task Spec generation", disable=not show_progress):
        anonymization = None
        if row.get("table_map"):
            anonymization = TableAnonymization(
                forward=dict(row["table_map"]), reverse=dict(row["reverse_table_map"])
            )
        gold = TaskSpec.model_validate(row["real_gold_spec"])
        if gold.validate_dependencies():
            raise ValueError(f"Invalid gold Task Spec: {row['uid']}")
        try:
            generated = generator.generate(
                row["system_prompt"],
                row["user_prompt"],
                anonymization=anonymization,
            )
            pred = generated.spec
            raw = generated.raw_text
            errors = generated.dependency_errors
        except TaskSpecOutputError as exc:
            pred = None
            raw = exc.raw_text
            errors = [f"{type(exc).__name__}: {exc}"]
        result_rows.append(
            {
                "uid": row["uid"],
                "domain": row["domain"],
                "question": row["question"],
                "gold": gold.model_dump(mode="json"),
                "prediction": pred.model_dump(mode="json") if pred is not None else None,
                "raw_output": raw,
                "dependency_errors": errors,
            }
        )

    return save_task_spec_results(result_rows, output_dir)
