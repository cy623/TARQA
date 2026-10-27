from pathlib import Path

from tarqa.data.metadata import SchemaCatalog
from tarqa.data.releases import ReleaseDataset
from tarqa.data.task_spec_labels import build_gold_task_spec
from tarqa.reasoning.factory import build_reasoning_model
from tarqa.reasoning.pipeline import ReasoningPipeline
from tarqa.reasoning.resdsql.evaluation import same_rows, summarize
from tarqa.utils.io import write_json, write_jsonl


def evaluate_reasoning(paths, model_cfg, execution_cfg, output_dir, *, model=None,
                       split="test", return_outputs=False, expected_release=None):
    """Known Reasoning subset, full schema; script 08 measures generated routing."""
    release = ReleaseDataset(paths)
    release_report = release.validate()
    if expected_release is not None and release_report != expected_release:
        raise ValueError("Reasoning evaluation release differs from the checkpoint snapshot")
    catalog = SchemaCatalog.from_files(paths.domain, paths.metadata, paths.profiling,
                                       database_path=paths.database)
    model = model if model is not None else build_reasoning_model(model_cfg)
    pipeline = ReasoningPipeline(catalog, paths.database, model,
        max_repair_attempts=int(execution_cfg.get("max_repair_attempts", 2)),
        timeout_seconds=float(execution_cfg.get("timeout_seconds", 120)))
    outputs = []
    try:
        for record in release.qa(split):
            if record.query_type == "Predictive":
                continue
            # Gold is scorer-only. No golden SQL/answer is passed to either model.
            gold_rows = pipeline.executor.execute(record.golden_sql)[1]
            item = dict(uid=record.uid, domain=paths.domain, parent_question_id=record.instance_id,
                        question=record.primary_question, correct=False, executable=False)
            try:
                result = pipeline.answer(record.primary_question, build_gold_task_spec(record),
                                          answer_nature=record.answer_nature)
                item.update(sql=result.sql, prediction=result.answer.value, attempts=result.attempts,
                    executable=True, correct=same_rows(result.rows, gold_rows, record.answer_nature))
                if getattr(model, "backend", None) == "resdsql":
                    item["resdsql"] = dict(model.last_details)
            except Exception as exc:
                if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                    raise
                item["error"] = f"{type(exc).__name__}: {exc}"
            outputs.append(item)
    finally:
        pipeline.close()
    metrics = dict(summarize(outputs), domain=paths.domain, qa_split=split,
                   backend=model_cfg.get("backend", "causal_lm"),
                   routing="known_reasoning_subset; use 08 for generated-router evaluation")
    output_dir = Path(output_dir)
    write_json(output_dir / "predictions.json", outputs)
    write_jsonl(output_dir / "predictions.jsonl", outputs)
    write_json(output_dir / "metrics.json", metrics)
    return (metrics, outputs) if return_outputs else metrics
